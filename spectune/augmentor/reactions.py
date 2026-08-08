"""Reaction context for a molecule: measured precedents, plausible routes, verification.

What the available resources actually deliver was measured before this module
was designed, on 491 distinct NMRexp test molecules:

- **Exact product lookup** in the local corpora is precise but sparse. A full
  scan of Pistachio (15.7M reactions) plus ChemPile-lift (452k) matched only
  30/491 molecules (6.1%) -- expected, since NMRexp molecules come from recent
  papers while those corpora are patent data. It is still worth doing: a hit
  gives real reactants, real agents/solvents, and a named reaction class, and
  a full parallel scan costs about a minute on this machine.
- **Murcko-scaffold matching** against the same corpora was tried as a way to
  raise coverage and rejected: it matched 151/355 scaffolds but the matches are
  dominated by generic cores (a bare benzene ring matches 1.3M reactions), so
  it carries almost no information while costing 10x the scan time.
- **RDKit retro-templates** cover nearly every molecule and propose chemically
  standard disconnections with their usual conditions.
- **ASKCOS forward prediction** then closes the loop: feeding a template's
  reactants back to ASKCOS says whether they really reconstruct the target. In
  spot checks the target came back rank 1 for straightforward esters, amides,
  and biaryls, which is what makes the generated routes safe to quote.

Everything here is best-effort: a molecule with no precedent, no applicable
template, and no ASKCOS access simply gets an empty reaction context.
"""

from __future__ import annotations

import asyncio
import csv
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectune.tools import AskcosReactionForwardPredictTool, ReactionLocalIndexSearchTool

from .config import EnrichmentConfig

JsonDict = dict[str, Any]

_ATOM_MAP_RE = re.compile(r":\d+\]")
_REACTION_SMILES_RE = re.compile(r"([^\s]+>>[^\s]+)")

_SCAN_TARGETS: frozenset[str] = frozenset()
_RDKIT_QUIETED = False


def _quiet_rdkit() -> None:
    """Silence RDKit's C++ logger, which writes straight to stderr.

    Several templates intentionally emit a fixed reagent with no atom maps
    (Boc anhydride, a phosphorus ylide), which RDKit reports on every single
    call unless the logger is turned off.
    """
    global _RDKIT_QUIETED
    if _RDKIT_QUIETED:
        return
    from rdkit import RDLogger  # type: ignore[import-not-found]

    RDLogger.DisableLog("rdApp.*")
    _RDKIT_QUIETED = True


@dataclass(frozen=True, slots=True)
class RetroTemplate:
    """One standard disconnection, written product-to-reactants."""

    key: str
    smarts: str
    zh: str
    en: str
    conditions_zh: str
    conditions_en: str


RETRO_TEMPLATES: tuple[RetroTemplate, ...] = (
    RetroTemplate(
        "esterification",
        # The carbon must carry an explicit carbon substituent besides the
        # carbonyl O and the ester O. Without it, "acyl carbon with an sp3-O
        # neighbour" also matches Boc/Cbz carbamates (O:3's neighbour is a
        # tert-butyl/benzyl carbon, same as a real ester) and dialkyl
        # carbonates, inventing a carbamic acid that was never a reagent.
        "[#6:5][C:1](=[O:2])[O;H0;!$(O[a]):3][C;X4:4]>>[#6:5][C:1](=[O:2])[OH:3].[OH][C:4]",
        "酯化反应",
        "esterification",
        "羧酸与醇，酸催化或 DCC/DMAP 缩合",
        "carboxylic acid + alcohol, acid catalysis or DCC/DMAP",
    ),
    RetroTemplate(
        "aryl_esterification",
        "[#6:5][C:1](=[O:2])[O;H0:3][c:4]>>[#6:5][C:1](=[O:2])[OH:3].[OH][c:4]",
        "芳基酯化反应",
        "aryl esterification",
        "羧酸与苯酚，DCC/DMAP 缩合或先制酰氯再与酚成酯",
        "carboxylic acid + phenol, DCC/DMAP coupling or via the acid chloride",
    ),
    RetroTemplate(
        "amide_coupling",
        # N:3 is, by construction, always bonded to the carbonyl carbon C:1 --
        # a naive "N not bonded to a C=O" filter matches that very bond and
        # excludes every amide, including this template's whole intended
        # scope. The filter instead has to name the *other* carbonyl
        # neighbour explicitly, which only a genuine imide/di-acyl nitrogen
        # has. The filter on C:1 rules out carbamates and ureas: their
        # carbonyl carbon has a second heteroatom (O or N) besides the
        # reacting N, so without this a Boc/Cbz carbamate or a urea gets
        # "split" into an amine plus an invented carbamic acid that is not a
        # real reagent. Formamide (R = H there) is unaffected, since the
        # exclusion needs two distinct heavy neighbours to fire.
        "[C;!$([CX3](=O)(N)[#7,#8]):1](=[O:2])[NX3;!$([N](C(=O))C(=O));!$(N[a]):3]>>[C:1](=[O:2])[OH].[N:3]",
        "酰胺缩合",
        "amide coupling",
        "羧酸与胺，HATU/EDCI 与 DIPEA",
        "carboxylic acid + amine with HATU/EDCI and DIPEA",
    ),
    RetroTemplate(
        "suzuki",
        "[c:1]-[c:2]>>[c:1]B(O)O.[c:2]Br",
        "Suzuki 偶联",
        "Suzuki-Miyaura coupling",
        "芳基硼酸与芳基卤代物，钯催化、碱性条件",
        "aryl boronic acid + aryl halide, Pd catalyst and base",
    ),
    RetroTemplate(
        "buchwald_hartwig",
        "[c:1][NX3;H0,H1;!$(NC=O):2]>>[c:1]Br.[N:2]",
        "Buchwald-Hartwig 胺化",
        "Buchwald-Hartwig amination",
        "芳基卤代物与胺，钯/膦配体、强碱",
        "aryl halide + amine, Pd/phosphine ligand and strong base",
    ),
    RetroTemplate(
        "sonogashira",
        "[c:1][CX2:2]#[CX2:3]>>[c:1]I.[CX2;H1:2]#[CX2:3]",
        "Sonogashira 偶联",
        "Sonogashira coupling",
        "芳基卤代物与端炔，Pd/CuI、胺碱",
        "aryl halide + terminal alkyne, Pd/CuI and amine base",
    ),
    RetroTemplate(
        "heck",
        "[c:1][CX3:2]=[CX3:3]>>[c:1]Br.[CX3;H2:2]=[CX3:3]",
        "Heck 反应",
        "Heck reaction",
        "芳基卤代物与烯烃，钯催化、碱",
        "aryl halide + alkene, Pd catalyst and base",
    ),
    RetroTemplate(
        "reductive_amination",
        "[C;X4;H1,H2:1][NX3;H0,H1;!$(NC=O):2]>>[C:1]=O.[N:2]",
        "还原胺化",
        "reductive amination",
        "醛/酮与胺，NaBH(OAc)3 还原",
        "aldehyde/ketone + amine with NaBH(OAc)3",
    ),
    RetroTemplate(
        "williamson_ether",
        "[c:1][OX2][CX4;!$(C=O):2]>>[c:1][OH].[C:2]Br",
        "Williamson 醚合成",
        "Williamson ether synthesis",
        "酚与卤代烷，K2CO3/DMF",
        "phenol + alkyl halide, K2CO3 in DMF",
    ),
    RetroTemplate(
        "sulfonamide",
        "[SX4:1](=[O:2])(=[O:3])[NX3:4]>>[S:1](=[O:2])(=[O:3])Cl.[N:4]",
        "磺酰胺化",
        "sulfonamide formation",
        "磺酰氯与胺，吡啶或三乙胺",
        "sulfonyl chloride + amine, pyridine or Et3N",
    ),
    RetroTemplate(
        "boc_protection",
        "[NX3:1][CX3](=O)[OX2]C([CH3])([CH3])[CH3]>>[N:1].O=C(OC(C)(C)C)OC(C)(C)C",
        "Boc 保护",
        "Boc protection",
        "胺与 Boc 酸酐，碱性条件",
        "amine + Boc anhydride under base",
    ),
    RetroTemplate(
        "cbz_protection",
        "[NX3:1][CX3](=O)[OX2]Cc1ccccc1>>[N:1].ClC(=O)OCc1ccccc1",
        "Cbz 保护",
        "Cbz protection",
        "胺与氯甲酸苄酯，碱性条件",
        "amine + benzyl chloroformate under base",
    ),
    RetroTemplate(
        "silyl_protection",
        "[OX2:1][Si:2]>>[O:1][H].Cl[Si:2]",
        "硅醚保护",
        "silyl ether protection",
        "醇与硅氯化物，咪唑/DMF",
        "alcohol + silyl chloride, imidazole in DMF",
    ),
    RetroTemplate(
        "click_triazole",
        "[cX3:1]1[cX3:2][nX2][nX2][nX3:3]1>>[C:1]#[C:2].[N:3]=[N+]=[N-]",
        "叠氮-炔环加成(click)",
        "azide-alkyne cycloaddition",
        "端炔与叠氮，CuI 催化",
        "terminal alkyne + azide, Cu(I) catalysis",
    ),
    RetroTemplate(
        "grignard_addition",
        # The carbon that becomes the Grignard reagent must be a plain alkyl
        # carbon: a real reagent would not survive a neighbouring OH/NH.
        "[C;X4;!$(C[O,N,S,F,Cl,Br,I]):1][CX4;H1,H2:2][OX2H:3]>>[C:1][Mg]Br.[C:2]=[O:3]",
        "格氏加成",
        "Grignard addition",
        "格氏试剂与醛酮，无水四氢呋喃",
        "Grignard reagent + carbonyl in dry THF",
    ),
    RetroTemplate(
        "wittig",
        "[CX3;!$(C=O):1]=[CX3;!$(C=O):2]>>[C:1]=O.[C:2]=P(c1ccccc1)(c1ccccc1)c1ccccc1",
        "Wittig 烯化",
        "Wittig olefination",
        "醛酮与磷叶立德",
        "carbonyl + phosphorus ylide",
    ),
    RetroTemplate(
        "oxime_formation",
        # O:3 is required to carry an H (a free oxime). An O-alkylated oxime
        # ether would not match, which is deliberate: that is made by a
        # different step (O-alkylation of the oxime), not straight off the
        # carbonyl. An aromatic isoxazole ring's C=N-O is a different atom
        # environment (lower-case n/o, no matching =N with an explicit
        # single-bonded [OH]) and is not touched by this pattern.
        "[CX3:1]=[NX2:2][OH:3]>>[C:1]=O.[N:2][O:3]",
        "缩合成肟",
        "oxime formation",
        "醛/酮与盐酸羟胺，醇/水中碱性条件",
        "aldehyde/ketone + hydroxylamine hydrochloride, base in alcohol/water",
    ),
    RetroTemplate(
        "hydrazone_formation",
        # Deliberately not filtered on what N:3 carries further -- a plain
        # hydrazine, an aryl hydrazine (phenylhydrazone), a sulfonylhydrazine
        # (tosylhydrazone) and a semicarbazide all condense with a carbonyl
        # the same way, so no single downstream substituent should exclude
        # the match. A pyrazole ring's C=N-N is aromatic (lower-case c/n)
        # and is a different atom environment, not matched here.
        "[CX3:1]=[NX2:2][NX3:3]>>[C:1]=O.[N:2][N:3]",
        "缩合成腙",
        "hydrazone formation",
        "醛/酮与肼/芳基肼/氨基脲，醇中酸催化",
        "aldehyde/ketone + hydrazine/arylhydrazine/semicarbazide, acid catalysis in alcohol",
    ),
    RetroTemplate(
        "urea_formation",
        # Requires *both* flanking atoms to be N, which already keeps this
        # disjoint from amide_coupling (one N) and from carbamates/esters
        # (one O). Cuts one C-N bond and restores the C=N-C=O of an
        # isocyanate rather than inventing a free "carbamic acid" -- that is
        # the standard way to draw a urea's retrosynthesis.
        "[NX3:1][CX3:2](=[O:3])[NX3:4]>>[N:1]=[C:2]=[O:3].[N:4]",
        "脲化反应",
        "urea formation",
        "异氰酸酯与胺，或胺与 CDI/三光气缩合",
        "isocyanate + amine, or amine + CDI/triphosgene",
    ),
    RetroTemplate(
        "friedel_crafts_acylation",
        # The far side of the carbonyl is pinned to carbon so this cannot
        # collide with amide_coupling (N there) or (aryl_)esterification (O
        # there) on the very same bond -- each keeps its own carbonyl
        # neighbour disjoint from the others'.
        "[c:1][CX3:2](=[O:3])[#6:4]>>[c:1].Cl[C:2](=[O:3])[#6:4]",
        "Friedel-Crafts 酰基化",
        "Friedel-Crafts acylation",
        "芳烃与酰氯，AlCl3 催化",
        "arene + acyl chloride, AlCl3 catalysis",
    ),
    RetroTemplate(
        "thioether_formation",
        # Mirrors williamson_ether's own scoping choice: only the aryl-S
        # case is templated, because a thiophenolate is as reliable a
        # nucleophile/leaving-group setup as a phenolate is, whereas a bare
        # dialkyl sulfide (both sides sp3) is common in natural products for
        # reasons that have nothing to do with a simple SN2 alkylation.
        "[c:1][SX2][CX4;!$(C=O):2]>>[c:1][SH].[C:2]Br",
        "硫醚合成",
        "thioether formation",
        "硫酚与卤代烷，K2CO3/DMF",
        "thiophenol + alkyl halide, K2CO3 in DMF",
    ),
    RetroTemplate(
        "disulfide_formation",
        "[#6:3][SX2:1][SX2:2][#6:4]>>[#6:3][SH:1].[SH:2][#6:4]",
        "二硫键氧化偶联",
        "disulfide formation",
        "两个硫醇，I2 或空气氧化偶联",
        "two thiols, oxidative coupling with I2 or air",
    ),
    RetroTemplate(
        "glycosylation",
        # Every pyranose ring atom is mapped (not just the anomeric carbon
        # and the exocyclic O/aglycone) so the sugar's other substituents
        # (its -OH groups, -CH2OH) are carried over onto the donor fragment
        # instead of being silently dropped. A free reducing sugar (bare
        # anomeric -OH, no exocyclic C) does not match.
        "[C;X4;R:1]1([OX2:2][#6;!$([#6]=[OX1]):3])[OX2;R:4][CX4;R:5][CX4;R:6][CX4;R:7][CX4;R:8]1"
        ">>[C:1]1([OH:2])[O:4][C:5][C:6][C:7][C:8]1.[OH][#6:3]",
        "糖苷化",
        "glycosylation",
        "糖基供体(异头位卤代/三氯乙亚胺酸酯等活化)与配基醇/酚,Lewis 酸(TMSOTf/BF3·OEt2)促进",
        "activated glycosyl donor (anomeric halide/trichloroacetimidate etc.)"
        " + acceptor alcohol/phenol, Lewis-acid promoter (TMSOTf/BF3\u00b7OEt2)",
    ),
)

# Canonical SMILES of agents that show up constantly in patent reaction records,
# so retrieved conditions can be written in Chinese instead of as raw SMILES.
AGENT_NAMES_ZH: Mapping[str, str] = {
    "ClCCl": "二氯甲烷",
    "ClC(Cl)Cl": "氯仿",
    "CO": "甲醇",
    "CCO": "乙醇",
    "CCOC(C)=O": "乙酸乙酯",
    "CC#N": "乙腈",
    "CN(C)C=O": "DMF",
    "CS(C)=O": "DMSO",
    "C1CCOC1": "四氢呋喃",
    "Cc1ccccc1": "甲苯",
    "c1ccccc1": "苯",
    "O": "水",
    "CC(C)=O": "丙酮",
    "CCOCC": "乙醚",
    "CC(=O)O": "乙酸",
    "Cl": "盐酸",
    "O=S(=O)(O)O": "硫酸",
    "[OH-]": "氢氧根",
    "O[Na]": "氢氧化钠",
    "O[K]": "氢氧化钾",
    "CC(C)(C)O[K]": "叔丁醇钾",
    "CC(C)(C)O[Na]": "叔丁醇钠",
    "O=C([O-])[O-]": "碳酸盐",
    "CCN(CC)CC": "三乙胺",
    "CC(C)N(C(C)C)C(C)C": "DIPEA",
    "c1ccncc1": "吡啶",
    "[Pd]": "钯催化剂",
    "[Cu]": "铜催化剂",
    "[Ni]": "镍催化剂",
    "[H][H]": "氢气",
    "[Li]CCCC": "正丁基锂",
    "[Al+3]": "铝试剂",
    "O=C1CCC(=O)N1Br": "NBS",
    "CC(C)(C#N)N=NC(C)(C)C#N": "AIBN",
}

# Monomer motifs, used to flag that a molecule plausibly sits in a polymer context.
_POLYMER_MOTIFS: tuple[tuple[str, str, str, str, str, int], ...] = (
    ("acrylate", "C=CC(=O)O", "丙烯酸酯类单体", "acrylate monomer", "自由基聚合", 1),
    ("styrene", "C=Cc1ccccc1", "苯乙烯类单体", "styrenic monomer", "自由基/阴离子聚合", 1),
    ("vinyl_ether", "C=C[OX2][#6]", "乙烯基醚单体", "vinyl ether monomer", "阳离子聚合", 1),
    ("epoxide", "[CX4;R]1[OX2;R][CX4;R]1", "环氧单体", "epoxide monomer", "开环聚合", 1),
    ("lactone", "[CX3;R](=O)[OX2;R]", "内酯单体", "lactone monomer", "开环聚合成聚酯", 1),
    ("lactam", "[NX3;R][CX3;R](=O)", "内酰胺单体", "lactam monomer", "开环聚合成聚酰胺", 1),
    ("isocyanate", "[NX2]=[CX2]=[OX1]", "异氰酸酯单体", "isocyanate monomer", "聚氨酯加成聚合", 1),
    ("siloxane", "[Si][OX2][Si]", "硅氧烷单元", "siloxane unit", "聚硅氧烷", 1),
    ("norbornene", "C1=CC2CCC1C2", "降冰片烯单体", "norbornene monomer", "开环易位聚合(ROMP)", 1),
    ("diol", "[OX2H][CX4]", "二元醇", "diol", "缩聚成聚酯/聚氨酯", 2),
    ("diacid", "[CX3](=O)[OX2H1]", "二元酸", "dicarboxylic acid", "缩聚成聚酯/聚酰胺", 2),
    ("diamine", "[NX3;H2;!$(NC=O)]", "二元胺", "diamine", "缩聚成聚酰胺", 2),
    ("dihalo_arene", "[F,Cl,Br,I][c]", "二卤代芳香单体", "dihaloarene monomer", "偶联型共轭高分子聚合", 2),
)


def _strip_atom_maps(text: str) -> str:
    return _ATOM_MAP_RE.sub("]", text)


def _canonical(smiles: str) -> str | None:
    from rdkit import Chem  # type: ignore[import-not-found]

    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule) if molecule is not None else None


def _flat_canonical(smiles: str) -> str | None:
    """Canonical SMILES with stereochemistry removed, for tolerant comparison.

    Forward-prediction models routinely return the right skeleton without
    stereocentres, which should still count as "this route reaches the target".
    """
    from rdkit import Chem  # type: ignore[import-not-found]

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, isomericSmiles=False)


def describe_agents(agents: Sequence[str]) -> list[str]:
    """Translate agent SMILES into Chinese names, keeping unknown ones verbatim."""
    described: list[str] = []
    for agent in agents:
        name = AGENT_NAMES_ZH.get(agent)
        if name is None:
            canonical = _canonical(agent) if agent else None
            name = AGENT_NAMES_ZH.get(canonical or "", agent)
        if name and name not in described:
            described.append(name)
    return described


# Local corpus scanning


def _initialize_scan_worker(targets: frozenset[str]) -> None:
    from rdkit import RDLogger  # type: ignore[import-not-found]

    RDLogger.DisableLog("rdApp.*")
    global _SCAN_TARGETS
    _SCAN_TARGETS = targets


def _pistachio_hits(lines: Sequence[str]) -> list[tuple[str, JsonDict]]:
    hits: list[tuple[str, JsonDict]] = []
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        # The reaction field may carry an appended extended-SMILES block
        # (" |f:2.3|"), which is metadata about component grouping.
        reaction = fields[0].split(None, 1)[0] if fields else ""
        parts = reaction.split(">")
        if len(parts) != 3:
            continue
        reactant_mix, agent_mix, product_mix = parts
        for product in product_mix.split("."):
            if len(product) < 6:
                continue
            key = _canonical(_strip_atom_maps(product))
            if key is None or key not in _SCAN_TARGETS:
                continue
            hits.append(
                (
                    key,
                    {
                        "source": "pistachio",
                        "reactants": _canonical_components(reactant_mix),
                        "agents": _canonical_components(agent_mix),
                        "reaction_class": fields[4] if len(fields) > 4 else "",
                        "reaction_class_code": fields[3] if len(fields) > 3 else "",
                        "reference": fields[1] if len(fields) > 1 else "",
                    },
                )
            )
    return hits


def _chempile_hits(texts: Sequence[str]) -> list[tuple[str, JsonDict]]:
    hits: list[tuple[str, JsonDict]] = []
    for text in texts:
        match = _REACTION_SMILES_RE.search(text)
        if not match:
            continue
        reaction = match.group(1).strip().rstrip(".,;:")
        if ">>" not in reaction:
            continue
        reactant_mix, product_mix = reaction.split(">>", 1)
        for product in product_mix.split("."):
            key = _canonical(product) if len(product) >= 6 else None
            if key is None or key not in _SCAN_TARGETS:
                continue
            hits.append(
                (
                    key,
                    {
                        "source": "chempile",
                        "reactants": _canonical_components(reactant_mix),
                        "agents": [],
                        "reaction_class": "",
                        "reaction_class_code": "",
                        "reference": "",
                    },
                )
            )
    return hits


def _canonical_components(mixture: str) -> list[str]:
    components: list[str] = []
    for token in mixture.split("."):
        token = token.strip()
        if not token:
            continue
        canonical = _canonical(_strip_atom_maps(token))
        if canonical and canonical not in components:
            components.append(canonical)
    return components


def _chunks(items: Iterable[Any], size: int, limit: int = 0) -> Iterator[list[Any]]:
    buffer: list[Any] = []
    total = 0
    for item in items:
        buffer.append(item)
        total += 1
        if len(buffer) >= size:
            yield buffer
            buffer = []
        if limit and total >= limit:
            break
    if buffer:
        yield buffer


def _uspto_hits(path: Path, targets: frozenset[str]) -> list[tuple[str, JsonDict]]:
    hits: list[tuple[str, JsonDict]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for product in str(row.get("product") or "").split("."):
                key = _canonical(product) if product else None
                if key is None or key not in targets:
                    continue
                hits.append(
                    (
                        key,
                        {
                            "source": "uspto",
                            "reactants": _canonical_components(str(row.get("reactant") or "")),
                            "agents": [],
                            "reaction_class": "",
                            "reaction_class_code": "",
                            "reference": "",
                        },
                    )
                )
    return hits


def scan_product_precedents(
    targets: Sequence[str],
    config: EnrichmentConfig,
    *,
    progress: bool = True,
) -> dict[str, list[JsonDict]]:
    """Find local reactions whose product is one of ``targets``.

    One streaming pass per configured corpus, parallelized across processes:
    Pistachio is 6.9 GB of atom-mapped reaction SMILES that no useful index
    ships with, so every product is re-canonicalized and tested for membership
    in the target set rather than the other way round. Results are bounded by
    ``config.max_precedents_per_molecule`` and de-duplicated by reactant set.
    """
    from spectune.dataloader.base import progress_iter

    _quiet_rdkit()
    target_set = frozenset(targets)
    found: dict[str, list[JsonDict]] = {}
    if not target_set or not config.max_precedents_per_molecule:
        return found

    index = config.reaction_index
    workers = config.reaction_scan_workers
    chunk_size = config.reaction_scan_chunk_size

    def absorb(hits: Iterable[tuple[str, JsonDict]]) -> None:
        for key, precedent in hits:
            bucket = found.setdefault(key, [])
            if len(bucket) >= config.max_precedents_per_molecule:
                continue
            signature = (precedent["source"], tuple(precedent["reactants"]))
            if any((item["source"], tuple(item["reactants"])) == signature for item in bucket):
                continue
            bucket.append(precedent)

    if "uspto" in config.reaction_sources and index.uspto_csv_path and Path(index.uspto_csv_path).exists():
        absorb(_uspto_hits(Path(index.uspto_csv_path), target_set))

    parallel_jobs: list[tuple[str, Any, Any]] = []
    if "pistachio" in config.reaction_sources and index.pistachio_smi_path:
        path = Path(index.pistachio_smi_path)
        if path.exists():
            parallel_jobs.append(("pistachio", path, _pistachio_hits))
    if "chempile" in config.reaction_sources and index.chempile_parquet_path:
        path = Path(index.chempile_parquet_path)
        if path.exists():
            parallel_jobs.append(("chempile", path, _chempile_hits))

    for name, path, worker in parallel_jobs:
        if name == "pistachio":
            handle = path.open(encoding="utf-8", errors="replace")
            items: Iterable[Any] = handle
        else:
            handle = None
            items = _read_chempile_texts(path)
        try:
            chunks = _chunks(items, chunk_size, config.reaction_scan_max_lines)
            labelled = progress_iter(
                chunks,
                label=f"Augmentor/reaction scan ({name}, {workers} workers, chunks)",
                enabled=progress,
            )
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_scan_worker,
                initargs=(target_set,),
            ) as executor:
                pending: list[Any] = []
                for chunk in labelled:
                    pending.append(executor.submit(worker, chunk))
                    if len(pending) >= workers * 3:
                        for future in pending:
                            absorb(future.result())
                        pending = []
                for future in pending:
                    absorb(future.result())
        finally:
            if handle is not None:
                handle.close()
    return found


def _read_chempile_texts(path: Path) -> Iterator[str]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read the ChemPile parquet; install spectune[reaction]") from exc

    frame = pd.read_parquet(path, columns=["text"])
    yield from frame["text"].fillna("").astype(str)


# Template-derived routes

_TEMPLATE_REACTION_CACHE: dict[str, tuple[Any, int, int]] = {}


def _heavy_unmapped_atom_count(template_mol: Any) -> int:
    """Count heavy atoms in a reaction template that carry no atom map.

    An atom map ties a template atom to a specific input atom, so it is
    "reused" verbatim in whatever product it ends up in. An unmapped atom has
    no such link: on the reactant side it names matched substructure the
    reaction discards outright (e.g. the whole Boc anhydride's leaving-group
    atoms); on the product side it names a brand-new atom the template
    introduces (a fresh -OH oxygen, a halide, a whole second reagent).
    """
    return sum(1 for atom in template_mol.GetAtoms() if atom.GetAtomMapNum() == 0 and atom.GetAtomicNum() > 1)


def _compiled_reaction(template: RetroTemplate) -> tuple[Any, int, int]:
    """Build (and cache) a template's reaction plus its expected heavy-atom delta.

    ``removed`` and ``added`` (see :func:`_heavy_unmapped_atom_count`) let a
    caller predict, for any input molecule, exactly how many heavy atoms a
    correct application of this template must produce:
    ``molecule_heavy_atoms - removed + added``. That prediction is what makes
    it possible to catch a template firing on a bond that turns out to sit
    inside a ring of the target: RDKit does not represent that as a
    ring-opening (one molecule), it duplicates the untouched ring atoms into
    *both* product fragments, which this heavy-atom count will not match.
    """
    from rdkit.Chem import AllChem  # type: ignore[import-not-found]

    cached = _TEMPLATE_REACTION_CACHE.get(template.key)
    if cached is not None:
        return cached
    reaction = AllChem.ReactionFromSmarts(template.smarts)
    removed = _heavy_unmapped_atom_count(reaction.GetReactantTemplate(0))
    added = sum(
        _heavy_unmapped_atom_count(reaction.GetProductTemplate(i)) for i in range(reaction.GetNumProductTemplates())
    )
    result = (reaction, removed, added)
    _TEMPLATE_REACTION_CACHE[template.key] = result
    return result


def retro_routes(smiles: str, *, max_routes: int = 4) -> list[JsonDict]:
    """Apply the standard disconnections in :data:`RETRO_TEMPLATES` to ``smiles``.

    Each route is a plausible way the molecule could have been made, with the
    conditions that normally accompany that transformation. Routes are ranked
    by how much of the molecule the disconnection actually cuts (balanced
    fragment sizes first), so a full-scale coupling outranks stripping off a
    methyl group.
    """
    from rdkit import Chem  # type: ignore[import-not-found]

    _quiet_rdkit()
    molecule = Chem.MolFromSmiles(str(smiles or "").strip())
    if molecule is None:
        return []
    target = Chem.MolToSmiles(molecule)
    molecule_heavy = molecule.GetNumHeavyAtoms()

    routes: list[JsonDict] = []
    seen: set[tuple[str, ...]] = set()
    for template in RETRO_TEMPLATES:
        try:
            reaction, removed_heavy, added_heavy = _compiled_reaction(template)
            expected_heavy = molecule_heavy - removed_heavy + added_heavy
            outcomes = reaction.RunReactants((molecule,))
        except Exception:
            continue
        for group in outcomes:
            reactants: list[str] = []
            valid = True
            produced_heavy = 0
            for fragment in group:
                try:
                    Chem.SanitizeMol(fragment)
                    # Templates that write an explicit [OH] or [NH2] leave those
                    # hydrogens as real atoms, which would surface as "[H]OC..."
                    # in the quoted reactant.
                    fragment = Chem.RemoveHs(fragment)
                except Exception:
                    valid = False
                    break
                if fragment.GetNumHeavyAtoms() == 0:
                    valid = False
                    break
                produced_heavy += fragment.GetNumHeavyAtoms()
                reactants.append(Chem.MolToSmiles(fragment))
            if not valid or not reactants:
                continue
            if produced_heavy != expected_heavy:
                # The matched bond sits inside a ring: RDKit duplicated the
                # untouched ring atoms into both fragments instead of
                # opening the ring into a single molecule. Not a real route.
                continue
            key = tuple(sorted(reactants))
            if key in seen or target in key:
                continue
            seen.add(key)
            sizes = sorted(len(part) for part in reactants)
            routes.append(
                {
                    "template": template.key,
                    "zh": template.zh,
                    "en": template.en,
                    "conditions_zh": template.conditions_zh,
                    "conditions_en": template.conditions_en,
                    "reactants": sorted(reactants),
                    "balance": sizes[0] / max(sizes[-1], 1),
                    "askcos": None,
                }
            )
    routes.sort(key=lambda route: route["balance"], reverse=True)
    return routes[: max(max_routes, 0)]


def polymer_context(smiles: str) -> JsonDict:
    """Flag monomer-like motifs so polymer background can be mentioned when apt.

    Motifs needing two copies (a diol, a diacid, a dihaloarene) are only
    reported when the molecule really carries two, which is what distinguishes
    a polycondensation monomer from an ordinary alcohol.
    """
    from rdkit import Chem  # type: ignore[import-not-found]

    _quiet_rdkit()
    molecule = Chem.MolFromSmiles(str(smiles or "").strip())
    if molecule is None:
        return {"monomer_classes": [], "is_probable_monomer": False}

    classes: list[JsonDict] = []
    for key, smarts, zh, en, polymerization_zh, minimum in _POLYMER_MOTIFS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        matches = molecule.GetSubstructMatches(pattern, uniquify=True, maxMatches=8)
        if len(matches) < minimum:
            continue
        classes.append(
            {
                "key": key,
                "zh": zh,
                "en": en,
                "polymerization_zh": polymerization_zh,
                "count": len(matches),
            }
        )
    return {"monomer_classes": classes, "is_probable_monomer": bool(classes)}


# External verification


async def verify_routes_with_askcos(
    target_smiles: str,
    routes: Sequence[JsonDict],
    config: EnrichmentConfig,
    *,
    tool: AskcosReactionForwardPredictTool | None = None,
) -> None:
    """Annotate the top routes in place with ASKCOS forward-prediction evidence.

    Only ``config.askcos_verify_top_routes`` routes per molecule are checked:
    ASKCOS is a public service and each call is a remote round trip, so this
    stays deliberately small and is skipped entirely when the knob is 0.
    """
    if not routes or config.askcos_verify_top_routes < 1:
        return
    tool = tool or AskcosReactionForwardPredictTool(config.askcos)
    target = _canonical(target_smiles) or target_smiles
    flat_target = _flat_canonical(target_smiles)
    semaphore = asyncio.Semaphore(config.askcos_max_concurrency)

    async def verify(route: JsonDict) -> None:
        async with semaphore:
            result = await tool.execute({"reactants": route["reactants"], "topk": 10})
        candidates = result.data.get("candidates") or []
        predicted = [str(item.get("canonical_smiles") or "") for item in candidates]
        rank: int | None = None
        match: str | None = None
        if target in predicted:
            rank, match = predicted.index(target) + 1, "exact"
        elif flat_target:
            flattened = [_flat_canonical(item) for item in predicted]
            if flat_target in flattened:
                rank, match = flattened.index(flat_target) + 1, "ignoring_stereochemistry"
        route["askcos"] = {
            "status": result.status,
            "target_rank": rank,
            "match": match,
            "verified": rank is not None,
            "top_prediction": predicted[0] if predicted else None,
        }

    await asyncio.gather(*(verify(route) for route in routes[: config.askcos_verify_top_routes]))


async def search_reactant_precedents(
    routes: Sequence[JsonDict],
    config: EnrichmentConfig,
    *,
    tool: ReactionLocalIndexSearchTool | None = None,
) -> list[JsonDict]:
    """Look up local precedents that start from the same reactants as ``routes``.

    This is the reactant-side counterpart of :func:`scan_product_precedents`,
    delegated to the ``reaction_local_index_search`` tool. It is off by default
    (``local_index_topk = 0``) because the tool reads only a bounded prefix of
    each corpus, so its recall on a specific molecule is low.
    """
    if not routes or config.local_index_topk < 1:
        return []
    tool = tool or ReactionLocalIndexSearchTool(config.reaction_index)
    result = await tool.execute({"reactants": routes[0]["reactants"], "topk": config.local_index_topk})
    candidates = result.data.get("candidates") or []
    return [
        {
            "smiles": candidate.get("canonical_smiles"),
            "score": candidate.get("score"),
            "source": candidate.get("source"),
            "reaction_class": candidate.get("reaction_evidence", {}).get("reaction_class_name", ""),
            "matched_reactants": candidate.get("reaction_evidence", {}).get("matched_reactants", []),
        }
        for candidate in candidates
    ]


__all__ = [
    "AGENT_NAMES_ZH",
    "RETRO_TEMPLATES",
    "RetroTemplate",
    "describe_agents",
    "polymer_context",
    "retro_routes",
    "scan_product_precedents",
    "search_reactant_precedents",
    "verify_routes_with_askcos",
]
