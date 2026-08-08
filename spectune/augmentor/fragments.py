"""Bilingual structural-fragment detection on top of RDKit substructure matching.

The augmentor needs to say things like "结构里应该有个三氟甲基" or "是个三萜骨架",
which requires a Chinese name for each detected fragment. Web search was
measured to be unreliable for this (a query keyed on a fragment SMILES tends to
return a *similar but different* structure, which would inject wrong Chinese
names straight into training data), so the authoritative source here is a
curated SMARTS lexicon with both names attached. Everything is offline,
deterministic, and cheap enough to run per row.

``specificity`` ranks how much a fragment actually tells a reader: a benzene
ring (1) is nearly free information, a Boc group or a triterpene skeleton (5-6)
narrows the structure a lot. Detection returns the most specific fragments
first so the query builder can quote one or two useful hints instead of a pile
of generic ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spectune.tools.utils import has_rdkit

JsonDict = dict[str, Any]


@dataclass(frozen=True, slots=True)
class FragmentDefinition:
    """One lexicon entry: a SMARTS pattern plus its bilingual names."""

    key: str
    smarts: str
    zh: str
    en: str
    category: str
    specificity: int


# ``[NX3]`` on its own matches amide, sulfonamide, nitro, imine, hydrazine and
# amidine nitrogens just as happily as it matches an amine, so every amine-like
# entry below shares this guard: no multiple bond on the nitrogen, no heteroatom
# neighbour (hydrazine, hydroxylamine, sulfenamide, phosphoramide), and no
# neighbouring carbon that is itself doubly bonded to O/S/N (amide, carbamate,
# urea, amidine, guanidine) or triply bonded to N (cyanamide).
_AMINE_N = "!$(N=*);!$(N#*);!$(N[!#6;!#1]);!$(N[#6]=[O,S,N]);!$(N[#6]#[#7])"

# Salts written as [NH3+]/[NH2+]/[NH+] are everywhere in vendor and ChEMBL-style
# SMILES, so the protonated forms have to be spelled out next to the neutral
# ones. Otherwise a hydrochloride silently loses its amine label -- and, before
# ``quaternary_ammonium`` was pinned to H0, was reported as a quaternary salt.
_PRIMARY_AMINE_SMARTS = f"[$([NX3;H2;{_AMINE_N}]),$([NX4+;H3;{_AMINE_N}])]"
_SECONDARY_AMINE_SMARTS = f"[$([NX3;H1;{_AMINE_N}]),$([NX4+;H2;{_AMINE_N}])]"
_TERTIARY_AMINE_SMARTS = f"[$([NX3;H0;{_AMINE_N}]),$([NX4+;H1;{_AMINE_N}])]"

# Both the pentavalent and the charge-separated spelling of a nitro group; RDKit
# accepts either on input. ``[#7+][O-]`` matches both of them too, which is why
# the N-oxide entry has to be written narrowly rather than generically.
_NITRO_N = "$([NX3](=O)=O),$([NX3+](=O)[O-])"

# A bare oxane ring. It only becomes a *pyranose* once ``_pyranose_rings``
# confirms the hydroxylation pattern, so the lexicon entry keyed on this SMARTS
# is named for what the SMARTS actually says and nothing more.
_OXANE_SMARTS = "[CX4;R]1[OX2;R][CX4;R][CX4;R][CX4;R][CX4;R]1"

# N-Calpha-C(=O)-N. Barring both nitrogens from being imide nitrogens is what
# separates a real backbone amide from a hydantoin or an N-acyl phthalimide.
_PEPTIDE_N = "NX3;!$(N=*);!$(N[!#6;!#1]);!$(N(C=O)C=O)"
_PEPTIDE_BOND_SMARTS = f"[{_PEPTIDE_N}][CX4][CX3](=O)[{_PEPTIDE_N}]"


# Ordered loosely by category; detection sorts by specificity, not by position.
FRAGMENT_LEXICON: tuple[FragmentDefinition, ...] = (
    # Aromatic and heteroaromatic rings
    FragmentDefinition("benzene", "c1ccccc1", "苯环", "benzene ring", "ring", 1),
    FragmentDefinition("naphthalene", "c1ccc2ccccc2c1", "萘环", "naphthalene", "ring", 3),
    FragmentDefinition("anthracene", "c1ccc2cc3ccccc3cc2c1", "蒽环", "anthracene", "ring", 5),
    FragmentDefinition("biphenyl", "c1ccc(-c2ccccc2)cc1", "联苯", "biphenyl", "ring", 3),
    FragmentDefinition("pyridine", "c1ccncc1", "吡啶环", "pyridine", "ring", 3),
    FragmentDefinition("pyrimidine", "c1cncnc1", "嘧啶环", "pyrimidine", "ring", 4),
    FragmentDefinition("pyrazine", "c1cnccn1", "吡嗪环", "pyrazine", "ring", 4),
    FragmentDefinition("pyrrole", "c1cc[nH]c1", "吡咯环", "pyrrole", "ring", 4),
    FragmentDefinition("furan", "c1ccoc1", "呋喃环", "furan", "ring", 4),
    FragmentDefinition("thiophene", "c1ccsc1", "噻吩环", "thiophene", "ring", 4),
    FragmentDefinition("imidazole", "c1cnc[nH]1", "咪唑环", "imidazole", "ring", 4),
    FragmentDefinition("pyrazole", "c1cn[nH]c1", "吡唑环", "pyrazole", "ring", 4),
    FragmentDefinition("oxazole", "c1ocnc1", "噁唑环", "oxazole", "ring", 4),
    FragmentDefinition("thiazole", "c1scnc1", "噻唑环", "thiazole", "ring", 4),
    FragmentDefinition("triazole", "c1cnnn1", "三氮唑环", "triazole", "ring", 5),
    FragmentDefinition("tetrazole", "c1nnn[nH]1", "四氮唑环", "tetrazole", "ring", 5),
    FragmentDefinition("indole", "c1ccc2[nH]ccc2c1", "吲哚环", "indole", "ring", 5),
    FragmentDefinition("quinoline", "c1ccc2ncccc2c1", "喹啉环", "quinoline", "ring", 5),
    FragmentDefinition("isoquinoline", "c1ccc2cnccc2c1", "异喹啉环", "isoquinoline", "ring", 5),
    FragmentDefinition("carbazole", "c1ccc2c(c1)[nH]c1ccccc12", "咔唑环", "carbazole", "ring", 5),
    FragmentDefinition("benzofuran", "c1ccc2occc2c1", "苯并呋喃", "benzofuran", "ring", 5),
    FragmentDefinition("benzothiophene", "c1ccc2sccc2c1", "苯并噻吩", "benzothiophene", "ring", 5),
    FragmentDefinition("isoxazole", "c1cnoc1", "异噁唑环", "isoxazole", "ring", 4),
    FragmentDefinition("isothiazole", "c1cnsc1", "异噻唑环", "isothiazole", "ring", 4),
    FragmentDefinition("pyridazine", "c1ccnnc1", "哒嗪环", "pyridazine", "ring", 4),
    FragmentDefinition("oxadiazole_134", "c1nnco1", "1,3,4-噁二唑环", "1,3,4-oxadiazole", "ring", 5),
    FragmentDefinition("oxadiazole_124", "c1ncon1", "1,2,4-噁二唑环", "1,2,4-oxadiazole", "ring", 5),
    FragmentDefinition("thiadiazole", "c1nncs1", "噻二唑环", "1,3,4-thiadiazole", "ring", 5),
    FragmentDefinition("benzimidazole", "c1ccc2ncnc2c1", "苯并咪唑环", "benzimidazole", "ring", 5),
    FragmentDefinition("benzoxazole", "c1ccc2ocnc2c1", "苯并噁唑环", "benzoxazole", "ring", 5),
    FragmentDefinition("benzothiazole", "c1ccc2scnc2c1", "苯并噻唑环", "benzothiazole", "ring", 5),
    FragmentDefinition("indazole", "c1ccc2nncc2c1", "吲唑环", "indazole", "ring", 5),
    FragmentDefinition("quinazoline", "c1ccc2ncncc2c1", "喹唑啉环", "quinazoline", "ring", 5),
    FragmentDefinition("quinoxaline", "c1ccc2nccnc2c1", "喹喔啉环", "quinoxaline", "ring", 5),
    FragmentDefinition("purine", "c1ncc2ncnc2n1", "嘌呤环", "purine", "ring", 5),
    FragmentDefinition("coumarin", "O=c1ccc2ccccc2o1", "香豆素骨架", "coumarin", "ring", 5),
    # Saturated rings
    FragmentDefinition("cyclopropane", "[CX4;R]1[CX4;R][CX4;R]1", "环丙烷", "cyclopropane", "ring", 4),
    FragmentDefinition("cyclobutane", "[CX4;R]1[CX4;R][CX4;R][CX4;R]1", "环丁烷", "cyclobutane", "ring", 3),
    FragmentDefinition("cyclopentane", "[CX4;R]1[CX4;R][CX4;R][CX4;R][CX4;R]1", "环戊烷", "cyclopentane", "ring", 2),
    FragmentDefinition(
        "cyclohexane", "[CX4;R]1[CX4;R][CX4;R][CX4;R][CX4;R][CX4;R]1", "环己烷", "cyclohexane", "ring", 2
    ),
    FragmentDefinition("oxetane", "[CX4;R]1[CX4;R][OX2;R][CX4;R]1", "氧杂环丁烷", "oxetane", "ring", 4),
    FragmentDefinition("azetidine", "[CX4;R]1[CX4;R][NX3;R][CX4;R]1", "氮杂环丁烷", "azetidine", "ring", 4),
    FragmentDefinition("piperidine", "C1CCNCC1", "哌啶环", "piperidine", "ring", 4),
    FragmentDefinition("piperazine", "C1CNCCN1", "哌嗪环", "piperazine", "ring", 4),
    FragmentDefinition("morpholine", "C1COCCN1", "吗啉环", "morpholine", "ring", 4),
    FragmentDefinition("pyrrolidine", "C1CCNC1", "吡咯烷环", "pyrrolidine", "ring", 4),
    FragmentDefinition("tetrahydrofuran", "C1CCOC1", "四氢呋喃环", "tetrahydrofuran", "ring", 3),
    FragmentDefinition("tetrahydropyran", _OXANE_SMARTS, "四氢吡喃环", "tetrahydropyran (oxane) ring", "ring", 2),
    FragmentDefinition("epoxide", "[CX4;R]1[OX2;R][CX4;R]1", "环氧乙烷", "epoxide", "group", 5),
    # Carbonyl-centred groups
    FragmentDefinition("carboxylic_acid", "[CX3](=O)[OX2H1]", "羧基", "carboxylic acid", "group", 2),
    # The leading [#6] keeps carbamates (Boc!) and carbonates out of "ester".
    FragmentDefinition("ester", "[#6][CX3](=O)[OX2H0][#6]", "酯基", "ester", "group", 2),
    # -@ forces the C-O bond itself to be a ring bond, so an acyclic ester whose
    # two halves happen to sit in unrelated rings is not called a lactone.
    FragmentDefinition("lactone", "[CX3;R](=O)-@[OX2H0;R]", "内酯", "lactone", "group", 4),
    FragmentDefinition("amide", "[NX3][CX3](=O)[#6]", "酰胺键", "amide", "group", 2),
    FragmentDefinition("lactam", "[NX3;R]-@[CX3;R](=O)", "内酰胺", "lactam", "group", 4),
    FragmentDefinition("aldehyde", "[CX3H1](=O)[#6]", "醛基", "aldehyde", "group", 3),
    FragmentDefinition("ketone", "[#6][CX3](=O)[#6]", "酮羰基", "ketone", "group", 2),
    FragmentDefinition("anhydride", "[CX3](=O)[OX2][CX3](=O)", "酸酐", "anhydride", "group", 5),
    FragmentDefinition("acyl_chloride", "[CX3](=O)[Cl]", "酰氯", "acyl chloride", "group", 5),
    FragmentDefinition("carbamate", "[NX3][CX3](=O)[OX2]", "氨基甲酸酯", "carbamate", "group", 4),
    FragmentDefinition("urea", "[NX3][CX3](=O)[NX3]", "脲基", "urea", "group", 4),
    # Acetals, orthoesters, and cyclic imides
    FragmentDefinition("acetal", "[CX4]([OX2][#6])[OX2][#6]", "缩醛/缩酮", "acetal or ketal", "group", 3),
    # [OX2] also matches -OH, so the unqualified pattern called a gem-triol an
    # orthoester; all three oxygens have to carry a carbon.
    FragmentDefinition("orthoester", "[CX4]([OX2H0][#6])([OX2H0][#6])[OX2H0][#6]", "原酸酯", "orthoester", "group", 5),
    FragmentDefinition("cyclic_imide", "O=[CX3;R][NX3;R][CX3;R]=O", "环状酰亚胺", "cyclic imide", "group", 4),
    # Nitrogen groups
    FragmentDefinition("nitrile", "[NX1]#[CX2]", "腈基", "nitrile", "group", 4),
    # The trailing [#6] is what makes this a C-nitro group; without it a nitrate
    # ester (R-O-NO2) was reported as a nitro group.
    FragmentDefinition("nitro", f"[{_NITRO_N}][#6]", "硝基", "nitro group", "group", 4),
    FragmentDefinition("nitrate_ester", f"[OX2][{_NITRO_N}]", "硝酸酯基", "nitrate ester", "group", 5),
    FragmentDefinition("azide", "[NX2]=[NX2+]=[NX1-]", "叠氮基", "azide", "group", 5),
    FragmentDefinition("primary_amine", _PRIMARY_AMINE_SMARTS, "伯胺", "primary amine", "group", 3),
    FragmentDefinition("secondary_amine", _SECONDARY_AMINE_SMARTS, "仲胺", "secondary amine", "group", 2),
    FragmentDefinition("tertiary_amine", _TERTIARY_AMINE_SMARTS, "叔胺", "tertiary amine", "group", 2),
    FragmentDefinition("aniline", f"[NX3;{_AMINE_N}][c]", "芳胺", "aniline nitrogen", "group", 3),
    # H0 plus four carbons. [NX4+] alone counts hydrogens in X, so it matched
    # every protonated amine salt as well as amine N-oxides.
    FragmentDefinition(
        "quaternary_ammonium", "[NX4+;H0]([#6])([#6])([#6])[#6]", "季铵盐", "quaternary ammonium", "group", 5
    ),
    # A Schiff base: neither the carbon nor the nitrogen may carry a heteroatom,
    # which is what keeps amidines, guanidines, oximes and hydrazones out.
    FragmentDefinition("imine", "[CX3;!$(C[!#6;!#1])]=[NX2;!$(N[!#6;!#1])]", "亚胺", "imine", "group", 4),
    FragmentDefinition("amidine", "[NX3][CX3]=[NX2]", "脒基", "amidine", "group", 4),
    FragmentDefinition("guanidine", "[NX3][CX3](=[NX2])[NX3]", "胍基", "guanidine", "group", 5),
    FragmentDefinition("hydrazone", "[NX3][NX2]=[CX3]", "腙", "hydrazone", "group", 5),
    FragmentDefinition("hydrazide", "[CX3](=O)[NX3][NX3]", "酰肼", "hydrazide", "group", 5),
    FragmentDefinition(
        "hydrazine",
        "[NX3;!$(N[#6]=[O,S,N])][NX3;!$(N[#6]=[O,S,N])]",
        "肼基",
        "hydrazine",
        "group",
        4,
    ),
    FragmentDefinition("oxime", "[CX3]=[NX2][OX2H]", "肟", "oxime", "group", 5),
    # Amine and azine oxides only. "[#7+][O-]" also matches the charge-separated
    # spelling of every nitro group, which is how nitroaromatics ended up
    # labelled as N-oxides.
    FragmentDefinition(
        "n_oxide",
        "[$([NX4+;H0]([#6])([#6])([#6])[OX1-]),$([nX3+][OX1-])]",
        "N-氧化物",
        "N-oxide",
        "group",
        4,
    ),
    FragmentDefinition("diazo", "[CX3]=[N+]=[N-]", "重氮基", "diazo group", "group", 5),
    FragmentDefinition("isocyanate", "[NX2]=[CX2]=[OX1]", "异氰酸酯基", "isocyanate", "group", 5),
    FragmentDefinition("isothiocyanate", "[NX2]=[CX2]=[SX1]", "异硫氰酸酯基", "isothiocyanate", "group", 5),
    FragmentDefinition("carbodiimide", "[NX2]=[CX2]=[NX2]", "碳二亚胺", "carbodiimide", "group", 5),
    FragmentDefinition("peptide_bond", _PEPTIDE_BOND_SMARTS, "肽键", "peptide bond", "group", 5),
    # Oxygen groups
    FragmentDefinition("phenol", "[OX2H][c]", "酚羟基", "phenol", "group", 3),
    FragmentDefinition("alcohol", "[OX2H][CX4]", "醇羟基", "alcohol", "group", 2),
    FragmentDefinition("aryl_methoxy", "[c][OX2][CH3]", "芳环甲氧基", "aryl methoxy", "group", 3),
    FragmentDefinition("ether", "[OD2]([#6])[#6]", "醚键", "ether", "group", 1),
    # Halogen and fluorinated groups
    FragmentDefinition("trifluoromethyl", "[CX4](F)(F)F", "三氟甲基", "trifluoromethyl", "group", 4),
    FragmentDefinition("trifluoromethoxy", "[OX2][CX4](F)(F)F", "三氟甲氧基", "trifluoromethoxy", "group", 5),
    FragmentDefinition("difluoromethyl", "[CX4;H1](F)F", "二氟甲基", "difluoromethyl", "group", 5),
    FragmentDefinition("aryl_fluoride", "[F][c]", "芳环氟取代", "aryl fluoride", "group", 3),
    FragmentDefinition("aryl_chloride", "[Cl][c]", "芳环氯取代", "aryl chloride", "group", 3),
    FragmentDefinition("aryl_bromide", "[Br][c]", "芳环溴取代", "aryl bromide", "group", 3),
    FragmentDefinition("aryl_iodide", "[I][c]", "芳环碘取代", "aryl iodide", "group", 4),
    FragmentDefinition("alkyl_halide", "[CX4;!$(C(F)(F)F)][F,Cl,Br,I]", "卤代烷基", "alkyl halide", "group", 2),
    # Sulfur and phosphorus
    FragmentDefinition("sulfonamide", "[SX4](=O)(=O)[NX3]", "磺酰胺", "sulfonamide", "group", 4),
    FragmentDefinition("sulfone", "[#6][SX4](=O)(=O)[#6]", "砜基", "sulfone", "group", 4),
    FragmentDefinition("sulfoxide", "[#6][SX3](=O)[#6]", "亚砜", "sulfoxide", "group", 4),
    FragmentDefinition("thioether", "[#6][SX2][#6]", "硫醚", "thioether", "group", 3),
    FragmentDefinition("thiol", "[SX2H]", "巯基", "thiol", "group", 4),
    FragmentDefinition("thioamide", "[NX3][CX3]=[SX1]", "硫代酰胺", "thioamide", "group", 4),
    FragmentDefinition("thiourea", "[NX3][CX3](=[SX1])[NX3]", "硫脲基", "thiourea", "group", 4),
    FragmentDefinition("disulfide", "[#16X2][#16X2]", "二硫键", "disulfide", "group", 4),
    FragmentDefinition("sulfonic_acid", "[SX4](=O)(=O)[OX2H1]", "磺酸基", "sulfonic acid", "group", 4),
    FragmentDefinition("sulfonate_ester", "[SX4](=O)(=O)[OX2H0][#6]", "磺酸酯", "sulfonate ester", "group", 3),
    FragmentDefinition("triflate", "[OX2][SX4](=O)(=O)C(F)(F)F", "三氟甲磺酸酯(OTf)", "triflate", "group", 5),
    FragmentDefinition("sulfonyl_chloride", "[SX4](=O)(=O)[Cl]", "磺酰氯", "sulfonyl chloride", "group", 5),
    # Phosphorus is only distinguishable by counting its substituents: three
    # oxygens is a phosphate, two oxygens plus a carbon a phosphonate, three
    # carbons a phosphine oxide. The anionic [OX1-] form matters because
    # nucleotide phosphates are routinely drawn deprotonated.
    FragmentDefinition(
        "phosphate_ester",
        "[PX4](=O)([OX2,OX1-])([OX2,OX1-])[OX2,OX1-]",
        "磷酸(酯)基",
        "phosphate ester",
        "group",
        4,
    ),
    FragmentDefinition("phosphonate", "[PX4](=O)([#6])([OX2,OX1-])[OX2,OX1-]", "膦酸酯", "phosphonate", "group", 4),
    FragmentDefinition("phosphine_oxide", "[PX4](=O)([#6])([#6])[#6]", "膦氧基", "phosphine oxide", "group", 4),
    FragmentDefinition("phosphine", "[PX3]([#6])([#6])[#6]", "膦", "phosphine", "group", 4),
    # Alkyl / unsaturation
    FragmentDefinition("tert_butyl", "[CX4]([CH3])([CH3])[CH3]", "叔丁基", "tert-butyl", "group", 3),
    FragmentDefinition("isopropyl", "[CX4;H1]([CH3])[CH3]", "异丙基", "isopropyl", "group", 2),
    FragmentDefinition("terminal_alkyne", "[CX2;H1]#[CX2]", "端炔", "terminal alkyne", "group", 5),
    FragmentDefinition("alkyne", "[CX2]#[CX2]", "炔键", "alkyne", "group", 4),
    FragmentDefinition("alkene", "[CX3]=[CX3]", "碳碳双键", "alkene", "group", 2),
    # Protecting and organometallic groups
    FragmentDefinition("boc", "CC(C)(C)OC(=O)[NX3]", "叔丁氧羰基(Boc)", "Boc group", "protecting group", 5),
    FragmentDefinition("cbz", "O=C(OCc1ccccc1)[NX3]", "苄氧羰基(Cbz)", "Cbz group", "protecting group", 5),
    # "[CH2]c1ccccc1" matches any benzylic methylene -- the middle of a
    # phenylpropyl chain, the CH2 of a tetralin -- and calling those a benzyl
    # group is wrong. A Bn group sits on a heteroatom or a halide (Bn ether,
    # amine, thioether, ester, halide); the bare benzylic CH2 gets its own
    # honest entry below.
    FragmentDefinition(
        "benzyl",
        "[OX2,NX3,NX4+,SX2,F,Cl,Br,I][CH2]c1ccccc1",
        "苄基",
        "benzyl",
        "protecting group",
        3,
    ),
    # "OC" would have accepted any O-alkyl, so a piperonyl (methylenedioxy)
    # group was reported as PMB; the methoxy has to be spelled out.
    FragmentDefinition(
        "pmb",
        "[OX2,NX3,NX4+,SX2][CH2]c1ccc([OX2][CH3])cc1",
        "对甲氧苄基(PMB)",
        "PMB group",
        "protecting group",
        3,
    ),
    FragmentDefinition("benzylic_ch2", "[CX4;H2]c1ccccc1", "苄位亚甲基", "benzylic CH2", "group", 1),
    FragmentDefinition(
        "trityl",
        "[CX4](c1ccccc1)(c1ccccc1)c1ccccc1",
        "三苯甲基(Trityl)",
        "trityl group",
        "protecting group",
        4,
    ),
    FragmentDefinition(
        "fmoc",
        "O=C(OCC1c2ccccc2-c2ccccc21)[NX3]",
        "9-芴基甲氧羰基(Fmoc)",
        "Fmoc group",
        "protecting group",
        5,
    ),
    FragmentDefinition("tosyl", "Cc1ccc(cc1)[SX4](=O)(=O)", "对甲苯磺酰基(Ts)", "tosyl", "protecting group", 5),
    FragmentDefinition(
        "tbs", "[Si]([CH3])([CH3])C(C)(C)C", "叔丁基二甲基硅基(TBS)", "TBS group", "protecting group", 5
    ),
    FragmentDefinition("tms", "[Si]([CH3])([CH3])[CH3]", "三甲基硅基(TMS)", "TMS group", "protecting group", 4),
    FragmentDefinition("silyl", "[Si]", "硅基", "silyl group", "group", 3),
    FragmentDefinition("boronate", "[BX3]([OX2])[OX2]", "硼酸/硼酸酯", "boronic acid or ester", "group", 5),
    # Bare [Fe] says nothing about ferrocene specifically -- a haem iron matches
    # it too -- so the name claims only what the SMARTS can see.
    FragmentDefinition("iron_complex", "[Fe]", "含铁配合物", "iron-containing complex", "group", 5),
)

_STEROID_SMARTS = "[#6]1~[#6]~[#6]~[#6]2~[#6](~[#6]1)~[#6]~[#6]~[#6]1~[#6]~2~[#6]~[#6]~[#6]2~[#6]~1~[#6]~[#6]~[#6]~2"

_MACROCYCLE_MIN_RING_SIZE = 12
_MAX_MATCHES = 64

# How many of the five ring carbons of an oxane must carry an exocyclic oxygen
# before the ring may be called a sugar. Three keeps the deoxy and amino sugars
# (rhamnose, glucosamine, N-acetylglucosamine) while rejecting THP ethers and
# spiroketals, which carry at most one.
_MIN_SUGAR_OXYGENS = 3

# A triterpene is a C30 skeleton; ring topology alone cannot tell one from any
# other saturated pentacyclic carbon cage, so the count has to be checked before
# the name is used. Oleanane-type skeletons put 22 carbons in the ring system.
_MIN_TRITERPENE_CARBONS = 25
_MIN_TRITERPENE_RING_CARBONS = 20
_MIN_TETRACYCLIC_RING_CARBONS = 16

_PATTERN_CACHE: dict[str, Any] = {}


def _compiled(smarts: str) -> Any:
    """Compile ``smarts`` once; RDKit SMARTS parsing is far from free per row."""
    from rdkit import Chem  # type: ignore[import-not-found]

    if smarts not in _PATTERN_CACHE:
        _PATTERN_CACHE[smarts] = Chem.MolFromSmarts(smarts)
    return _PATTERN_CACHE[smarts]


def _fused_ring_systems(molecule: Any) -> list[list[int]]:
    """Group ring indices into fused systems (rings sharing at least one atom)."""
    rings = [set(ring) for ring in molecule.GetRingInfo().AtomRings()]
    systems: list[list[int]] = []
    assigned: set[int] = set()
    for index in range(len(rings)):
        if index in assigned:
            continue
        system = [index]
        assigned.add(index)
        changed = True
        while changed:
            changed = False
            for other in range(len(rings)):
                if other in assigned:
                    continue
                if any(rings[other] & rings[member] for member in system):
                    system.append(other)
                    assigned.add(other)
                    changed = True
        systems.append(system)
    return systems


def _exocyclic_oxygens(molecule: Any, atom_index: int, ring_atoms: frozenset[int]) -> list[Any]:
    """Oxygen neighbours of ``atom_index`` that lie outside ``ring_atoms``."""
    atom = molecule.GetAtomWithIdx(atom_index)
    return [
        neighbour
        for neighbour in atom.GetNeighbors()
        if neighbour.GetSymbol() == "O" and neighbour.GetIdx() not in ring_atoms
    ]


def _pyranose_rings(molecule: Any) -> list[tuple[frozenset[int], tuple[int, ...]]]:
    """Oxane rings that are actually sugars, with their anomeric carbons.

    ``_OXANE_SMARTS`` on its own is only a tetrahydropyran: a THP-protected
    alcohol and a steroidal spiroketal match it just as well as glucose does, so
    calling every hit a pyranose invents a sugar that is not in the molecule. A
    ring is accepted only if it is hydroxylated like a sugar -- at least
    ``_MIN_SUGAR_OXYGENS`` of its five carbons carry an exocyclic oxygen -- and
    if a carbon next to the ring oxygen carries one too, which is the anomeric
    centre. Returns ``(ring atoms, anomeric carbons)`` for each accepted ring.
    """
    pattern = _compiled(_OXANE_SMARTS)
    if pattern is None:
        return []
    accepted: list[tuple[frozenset[int], tuple[int, ...]]] = []
    seen: set[frozenset[int]] = set()
    for match in molecule.GetSubstructMatches(pattern, uniquify=True, maxMatches=_MAX_MATCHES):
        # A symmetric ring matches several times over; the atom set identifies it
        # uniquely. The SMARTS atom order is C, O, C, C, C, C, so position 1 is
        # always the ring oxygen and positions 0 and 2 its two ring carbons.
        ring_atoms = frozenset(match)
        if ring_atoms in seen:
            continue
        seen.add(ring_atoms)
        oxygenated = {index for index in match if index != match[1] and _exocyclic_oxygens(molecule, index, ring_atoms)}
        if len(oxygenated) < _MIN_SUGAR_OXYGENS:
            continue
        anomeric = tuple(index for index in (match[0], match[2]) if index in oxygenated)
        if not anomeric:
            continue
        accepted.append((ring_atoms, anomeric))
    return accepted


def _glycosidic_oxygens(molecule: Any, pyranoses: list[tuple[frozenset[int], tuple[int, ...]]]) -> set[int]:
    """Anomeric oxygens that bridge a confirmed sugar ring to an aglycone.

    A reducing sugar's anomeric carbon carries a free -OH, whose only heavy
    neighbour is the ring carbon itself. A glycoside bonds that oxygen onward to
    a further carbon: the aglycone, or a second ring in a di-/oligosaccharide.
    An acyl carbon there makes it a 1-O-acyl ester rather than a glycoside, and a
    phosphorus makes it a sugar phosphate, so both are rejected. Oxygens are
    returned as a set because a 1->1 linkage is anomeric on both sides but is
    still a single glycosidic bond.
    """
    linkages: set[int] = set()
    for ring_atoms, anomeric in pyranoses:
        for carbon_index in anomeric:
            for oxygen in _exocyclic_oxygens(molecule, carbon_index, ring_atoms):
                for partner in oxygen.GetNeighbors():
                    if partner.GetIdx() == carbon_index or partner.GetSymbol() != "C":
                        continue
                    if any(
                        bond.GetBondTypeAsDouble() == 2.0 and bond.GetOtherAtom(partner).GetSymbol() in ("O", "N", "S")
                        for bond in partner.GetBonds()
                    ):
                        continue
                    linkages.add(oxygen.GetIdx())
    return linkages


def _skeleton_hints(molecule: Any) -> list[JsonDict]:
    """Detect large fused-carbocycle skeletons (steroid / terpenoid families).

    These have no single reliable SMARTS -- a triterpene is defined by its
    fused-ring topology and saturation rather than by one substructure -- so
    they are recognised from ring-system shape instead: how many rings are
    fused together, how large they are, and how saturated the system is.
    """
    from rdkit import Chem  # type: ignore[import-not-found]

    hints: list[JsonDict] = []
    steroid = _compiled(_STEROID_SMARTS)
    if steroid is not None and molecule.HasSubstructMatch(steroid):
        hints.append(
            {
                "key": "steroid_skeleton",
                "zh": "甾体骨架",
                "en": "steroid skeleton",
                "category": "skeleton",
                "specificity": 6,
                "count": 1,
            }
        )

    rings = molecule.GetRingInfo().AtomRings()
    total_carbons = sum(1 for atom in molecule.GetAtoms() if atom.GetSymbol() == "C")
    for system in _fused_ring_systems(molecule):
        if len(system) < 4:
            continue
        atoms = {atom for index in system for atom in rings[index]}
        carbons = [molecule.GetAtomWithIdx(index) for index in atoms]
        if any(atom.GetIsAromatic() for atom in carbons):
            continue
        if not all(atom.GetSymbol() == "C" for atom in carbons):
            continue
        six_membered = sum(1 for index in system if len(rings[index]) == 6)
        # Topology alone cannot name a terpene class: the carbon count has to
        # back it up, otherwise any saturated polycyclic cage becomes a
        # "triterpene". Anything that does not clear the counts falls through to
        # the purely descriptive name.
        if (
            len(system) >= 5
            and six_membered >= 4
            and len(atoms) >= _MIN_TRITERPENE_RING_CARBONS
            and total_carbons >= _MIN_TRITERPENE_CARBONS
        ):
            zh, en, key = "三萜骨架", "triterpene skeleton", "triterpene_skeleton"
        elif len(system) == 4 and six_membered >= 3 and len(atoms) >= _MIN_TETRACYCLIC_RING_CARBONS:
            zh, en, key = "四环二萜/甾体类骨架", "tetracyclic terpenoid skeleton", "tetracyclic_terpenoid"
        else:
            zh, en, key = "多环稠合碳骨架", "fused polycyclic carbon skeleton", "fused_polycycle"
        if any(hint["key"] == key for hint in hints):
            continue
        hints.append({"key": key, "zh": zh, "en": en, "category": "skeleton", "specificity": 6, "count": 1})

    if any(len(ring) >= _MACROCYCLE_MIN_RING_SIZE for ring in rings):
        hints.append(
            {
                "key": "macrocycle",
                "zh": "大环骨架",
                "en": "macrocyclic ring",
                "category": "skeleton",
                "specificity": 6,
                "count": 1,
            }
        )

    peptide_bond = _compiled(_PEPTIDE_BOND_SMARTS)
    if peptide_bond is not None:
        num_peptide_bonds = len(molecule.GetSubstructMatches(peptide_bond, uniquify=True, maxMatches=_MAX_MATCHES))
        if num_peptide_bonds >= 2:
            hints.append(
                {
                    "key": "peptide_chain",
                    "zh": "多肽链",
                    "en": "peptide chain",
                    "category": "skeleton",
                    "specificity": 6,
                    "count": num_peptide_bonds,
                }
            )

    # Both of these live here rather than in the lexicon because "is this oxane a
    # sugar" is a hydroxylation count, not a substructure.
    pyranoses = _pyranose_rings(molecule)
    if pyranoses:
        hints.append(
            {
                "key": "pyranose",
                "zh": "吡喃糖环",
                "en": "pyranose (sugar) ring",
                "category": "ring",
                "specificity": 4,
                "count": len(pyranoses),
            }
        )
        glycosidic = _glycosidic_oxygens(molecule, pyranoses)
        if glycosidic:
            hints.append(
                {
                    "key": "glycoside",
                    "zh": "糖苷键",
                    "en": "glycosidic bond",
                    "category": "skeleton",
                    "specificity": 5,
                    "count": len(glycosidic),
                }
            )

    scaffold = _murcko_scaffold(molecule, Chem)
    if scaffold:
        hints.append(
            {
                "key": "murcko_scaffold",
                "zh": "分子骨架(Murcko)",
                "en": "Murcko scaffold",
                "category": "skeleton",
                "specificity": 0,
                "count": 1,
                "smiles": scaffold,
            }
        )
    return hints


def _murcko_scaffold(molecule: Any, chem: Any) -> str:
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore[import-not-found]

        core = MurckoScaffold.GetScaffoldForMol(molecule)
    except Exception:
        return ""
    if core is None or core.GetNumAtoms() == 0:
        return ""
    return str(chem.MolToSmiles(core))


def detect_fragments(smiles: str, *, max_fragments: int = 8) -> list[JsonDict]:
    """Return the most informative bilingual fragments present in ``smiles``.

    Results are sorted by ``specificity`` (descending) then match count, and
    the Murcko scaffold is always appended last with specificity 0 so callers
    can keep it for analysis without it ever being chosen as a query hint.
    """
    if not has_rdkit():
        return []
    from rdkit import Chem  # type: ignore[import-not-found]

    molecule = Chem.MolFromSmiles(str(smiles or "").strip())
    if molecule is None:
        return []

    found: list[JsonDict] = []
    for definition in FRAGMENT_LEXICON:
        pattern = _compiled(definition.smarts)
        if pattern is None:
            continue
        matches = molecule.GetSubstructMatches(pattern, uniquify=True, maxMatches=_MAX_MATCHES)
        if not matches:
            continue
        found.append(
            {
                "key": definition.key,
                "zh": definition.zh,
                "en": definition.en,
                "category": definition.category,
                "specificity": definition.specificity,
                "count": len(matches),
            }
        )
    found.extend(_skeleton_hints(molecule))
    found.sort(key=lambda item: (item["specificity"], item["count"]), reverse=True)
    scaffold = [item for item in found if item["key"] == "murcko_scaffold"]
    ranked = [item for item in found if item["key"] != "murcko_scaffold"][: max(max_fragments, 0)]
    return ranked + scaffold


def molecule_properties(smiles: str) -> JsonDict:
    """Formula, weight, and coarse shape descriptors for one molecule.

    Returns ``{}`` without RDKit or on an unparseable SMILES, which callers
    treat as "molecular weight unavailable" rather than as an error.
    """
    if not has_rdkit():
        return {}
    from rdkit import Chem  # type: ignore[import-not-found]
    from rdkit.Chem import Descriptors, rdMolDescriptors  # type: ignore[import-not-found]

    molecule = Chem.MolFromSmiles(str(smiles or "").strip())
    if molecule is None:
        return {}
    return {
        "canonical_smiles": Chem.MolToSmiles(molecule),
        "molecular_formula": str(rdMolDescriptors.CalcMolFormula(molecule)),
        "molecular_weight": round(float(Descriptors.MolWt(molecule)), 3),
        "exact_mass": round(float(Descriptors.ExactMolWt(molecule)), 4),
        "heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
        "ring_count": int(rdMolDescriptors.CalcNumRings(molecule)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "murcko_scaffold": _murcko_scaffold(molecule, Chem),
    }


__all__ = ["FRAGMENT_LEXICON", "FragmentDefinition", "detect_fragments", "molecule_properties"]
