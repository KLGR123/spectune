## Augmentor

Turns clustered truth rows into user-style queries. 

```python
from spectune import Augmentor, AugmentorConfig, EnrichmentConfig, LlmConfig

augmentor = Augmentor(
    AugmentorConfig(
        split="train",
        sample_size=1_000,                 # 0 = whole split; balanced across clusters
        information_mix={                  # shares must sum to 1
            "none": 0.30,                  # spectrum only
            "formula": 0.20,               # + molecular formula
            "structure": 0.20,             # + hedged SMILES / Chinese / English name
            "reaction": 0.15,              # + reactants, conditions, polymer background
            "fragment": 0.15,              # + structural fragment hints
        },
        followup_probability=0.4,          # else the information is stated up front
        use_llm_rewrite=False,             # True → call LLM to rewrite; False (default) → template only
        raw_query_ratio=0.3,               # (only used when use_llm_rewrite=True) share left as plain template
        nmr_noise_ratio=0.2,               # share whose spectrum is corrupted
        nmr_noise_strength=0.3,            # digits_only / strip_punctuation / perturb_values / drop_peaks
        formula_noise_ratio=0.1,           # share of formula-condition rows given a nearby formula
        llm=LlmConfig(max_concurrency=16),                    # reads SPECTUNE_LLM_* by default
        enrichment=EnrichmentConfig(resolve_reactions=True),  # disable stages you don't need
    )
)
dataset = augmentor.build()  # /nmrexp_augmented_train.jsonl
print(augmentor.last_summary)
for sample in dataset.take(3):
    print(sample["num_of_queries"], sample["augmentation"]["information_type"], sample["turns"][0]["content"])
```

Enrichment reuses whatever the cache already covers, so a cheap first pass and a later full pass cost only the new stage.

```python
from spectune import Enricher, EnrichmentConfig

Enricher(EnrichmentConfig(resolve_names=False, resolve_reactions=False)).enrich(truth_train)  # structure only
Enricher(EnrichmentConfig()).enrich(truth_train)  # adds names + reactions, keeps the rest
```

## Reaction Context Status

Last updated: 2026-07-30

Reaction enrichment is deliberately conservative: a missing route is preferable
to a chemically false route. The output keeps measured precedents and generated
retrosynthetic suggestions separate; a template route is a plausible synthesis
hypothesis, not evidence that the reported compound was actually made that way.

### Evidence levels

1. `precedents` are exact-product matches from configured Pistachio, ChemPile,
   or USPTO data. They are the strongest evidence but are sparse: a measured
   full scan matched 30/491 NMRexp test molecules (6.1%).
2. `routes` are generated from the 23 narrow RDKit templates in
   `RETRO_TEMPLATES`. They supply standard reactants and conditions only where
   the product contains the complete reaction motif.
3. `routes[*].askcos` records optional forward-prediction verification. By
   default only the top route is checked (`askcos_verify_top_routes=1`);
   `verified=true` means ASKCOS regenerated the target in its top 10, either
   exactly or after removing stereochemistry. `askcos=None`, a failed status,
   or `verified=false` must not be read as positive evidence.
4. `reactant_precedents` are disabled by default (`local_index_topk=0`) because
   the local reactant search reads only a bounded corpus prefix and therefore
   has low recall for a specific route.

Murcko-scaffold reaction matching is intentionally not used. In the measured
corpora it raised apparent coverage but generic scaffolds dominated the hits
(a bare benzene ring matched about 1.3 million reactions), making the result
too weak to quote as reaction evidence.

### Retrosynthetic template coverage

The current 23 templates cover:

- ester and aryl-ester formation; amide and urea formation;
- Suzuki, Buchwald-Hartwig, Sonogashira, Heck, and
  Friedel-Crafts acylation;
- reductive amination, Williamson ether synthesis, oxime and hydrazone
  formation;
- sulfonamide, aryl thioether, and disulfide formation;
- Boc, Cbz, and silyl protection;
- azide-alkyne click chemistry, Grignard addition, Wittig olefination, and
  pyranose O-glycosylation.

Routes are ranked by disconnection balance, not by historical probability or
confidence. ASKCOS verification and exact precedents remain separate fields.

### Correctness safeguards

- Every generated outcome is sanitized and must conserve the heavy-atom count
  implied by its atom-mapped template. This rejects RDKit outcomes where a
  template cuts a bond inside a ring and duplicates the untouched ring atoms
  into two fake reactants.
- A route containing the unchanged target as one of its reactants is rejected,
  and duplicate reactant sets are collapsed.
- Carbonyl templates are mutually narrowed where confusion would fabricate a
  reagent: ordinary esterification excludes carbamates and carbonates; amide
  coupling excludes carbamates, ureas, imides, and anilide nitrogens; ureas
  disconnect to an isocyanate plus an amine instead of to a fictitious free
  carbamic acid.
- Glycosylation requires a confirmed pyranose anomeric C-O-C linkage and maps
  every ring atom, preserving the sugar's hydroxyl and hydroxymethyl
  substituents. A free reducing sugar does not match.
- Narrow templates intentionally leave unsupported chemistry empty. Notably,
  there is no generic retro-Diels-Alder, dialkyl ether/thioether, Negishi,
  Stille, Kumada, Mitsunobu, or C-glycoside template: the product alone does
  not identify those histories reliably enough.

If no exact precedent, valid template, or usable external verification exists,
the reaction context remains empty. This is expected behaviour, not an error.

## Fragment Lexicon Status

Last updated: 2026-07-30

`FRAGMENT_LEXICON` in `fragments.py` — 131 SMARTS entries + 7 skeleton heuristics (`_skeleton_hints`).
All entries validated against RDKit 2025.09.3; `pytest tests/test_augmentor.py` passes 91/91.

A detected name is quoted verbatim into the training query, so the lexicon is
written to under-report rather than to guess: every pattern is narrow enough that
a hit means the group is really there, even where that costs coverage.

Results are ranked by `specificity` descending, then by match count, and the
augmentor quotes only the leading one or two entries (`max_fragment_items=2`)
and only their `zh` name; the count itself is used for ranking and never
rendered. A false positive with high specificity therefore reaches the training
data, while a missed group only leaves the query thinner — which is why the
patterns are biased towards silence. `murcko_scaffold` carries specificity 0 and
is appended last, so it stays available for analysis without ever being quoted.

### SMARTS Lexicon (131 entries)


| Category                            | Count | Representative keys                                                                                                                                                                                                                |
| ----------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Aromatic / heteroaromatic rings     | 40    | `benzene` `pyridine` `imidazole` `indole` `quinoline` `benzimidazole` `oxadiazole_134` `purine` `coumarin` …                                                                                                                       |
| Saturated rings                     | 13    | `cyclopropane` `cyclobutane` `cyclopentane` `cyclohexane` `oxetane` `azetidine` `piperidine` `morpholine` `tetrahydropyran` …                                                                                                      |
| Carbonyl groups                     | 9     | `carboxylic_acid` `ester` `lactone` `amide` `lactam` `aldehyde` `ketone` `anhydride` `acyl_chloride`                                                                                                                               |
| Acetals, orthoesters, cyclic imides | 5     | `carbamate` `urea` `acetal` `orthoester` `cyclic_imide`                                                                                                                                                                            |
| Nitrogen groups                     | 21    | `nitrile` `nitro` `nitrate_ester` `azide` `primary_amine` `secondary_amine` `tertiary_amine` `aniline` `imine` `amidine` `guanidine` `hydrazone` `hydrazide` `hydrazine` `oxime` `n_oxide` `diazo` `isocyanate` `peptide_bond` … |
| Oxygen groups                       | 4     | `phenol` `alcohol` `aryl_methoxy` `ether`                                                                                                                                                                                          |
| Halogens / fluorinated              | 8     | `trifluoromethyl` `trifluoromethoxy` `difluoromethyl` `aryl_fluoride/chloride/bromide/iodide` `alkyl_halide`                                                                                                                       |
| Sulfur / phosphorus                 | 14    | `sulfonamide` `sulfone` `sulfoxide` `thioether` `thiol` `thioamide` `thiourea` `disulfide` `sulfonic_acid` `sulfonate_ester` `triflate` `sulfonyl_chloride` `phosphate_ester` `phosphonate` `phosphine_oxide` `phosphine`          |
| Alkyl / unsaturation                | 5     | `tert_butyl` `isopropyl` `terminal_alkyne` `alkyne` `alkene`                                                                                                                                                                       |
| Protecting groups / organometallic  | 13    | `boc` `cbz` `benzyl` `benzylic_ch2` `pmb` `trityl` `fmoc` `tosyl` `tbs` `tms` `silyl` `boronate` `iron_complex`                                                                                                                    |




### Skeleton-level heuristics (`_skeleton_hints`, not SMARTS entries)


| key | Trigger condition |
| --- | --- |
| `steroid_skeleton` | Matches the steroid-core SMARTS |
| `triterpene_skeleton` / `tetracyclic_terpenoid` / `fused_polycycle` | All-carbon saturated fused ring system; distinguished by ring count, six-membered ring ratio, and carbon count. Without the carbon count adamantane reads as a tetracyclic terpenoid, so anything that fails it falls back to the descriptive `fused_polycycle` |
| `macrocycle` | Any single ring with ≥ 12 atoms |
| `peptide_chain` | ≥ 2 peptide bonds detected |
| `pyranose` | An oxane ring hydroxylated like a sugar: ≥ 3 of its 5 carbons carry an exocyclic O, one of them next to the ring O. The bare oxane is reported as `tetrahydropyran` instead, since a THP ether and a spiroketal match the ring on its own |
| `glycoside` | The anomeric O of a confirmed `pyranose` bonded onward to a non-acyl carbon. An acyl carbon makes it a 1-O-acyl ester and a phosphorus a sugar phosphate, so neither counts |
| `murcko_scaffold` | Computed for every molecule; specificity = 0, always last in results |




### Disambiguation safeguards

Every entry below was once written loosely enough to report a chemically
unrelated group under the wrong name. They are pinned narrow on purpose:
loosening one re-introduces a false Chinese name, not merely extra hits.

- `n_oxide` accepts only trialkylamine and azine oxides. A generic `[#7+][O-]`
  also matches the charge-separated spelling of a nitro group, which RDKit
  produces from both `[N+](=O)[O-]` and `N(=O)=O` input, so every nitroaromatic
  read as an N-oxide as well.
- `quaternary_ammonium` requires `H0` and four carbons, because `X` counts
  hydrogens: `[NX4+]` matched every `[NH3+]` salt, every `[NH4+]` counterion and
  every amine oxide. The amine entries spell out the protonated forms
  (`[NX4+;H3]` / `H2` / `H1`) so a hydrochloride keeps its amine label rather
  than losing it together with the false quaternary one.
- The three amine entries and `aniline` share one guard: no multiple bond on the
  nitrogen, no heteroatom neighbour, and no neighbouring carbon doubly bonded to
  O/S/N. That keeps amides, sulfonamides, hydrazines, hydroxylamines, amidines
  and guanidines out of the amine names, and stops a nitro or phosphoramide
  nitrogen on an aromatic ring from reading as an aromatic amine.
- `nitro` requires a carbon on the nitrogen; a nitrate ester (`R-O-NO2`) has its
  own entry instead of borrowing the nitro name.
- Phosphorus is named by counting substituents: three oxygens a phosphate, two
  oxygens plus a carbon a phosphonate, three carbons a phosphine oxide. A bare
  `[PX4](=O)` called every nucleotide phosphate a phosphine oxide, and a bare
  `[PX3]` called every phosphoramidite a phosphine.
- Sugar recognition is a hydroxyl count in `_pyranose_rings`, not a SMARTS,
  because the oxane pattern alone also fits a THP ether and a spiroketal. The
  bare ring is reported as `tetrahydropyran`.
- `benzyl` requires the CH2 to sit on a heteroatom or a halide. `[CH2]c1ccccc1`
  also matched the middle of a phenylpropyl chain and the CH2 of a tetralin;
  those are `benzylic_ch2` at specificity 1 now. `pmb` spells the methoxy out,
  since `OC` accepted the methylenedioxy of a piperonyl group.
- `peptide_bond` bars both nitrogens from being imide nitrogens, which is what
  separates a backbone amide from a hydantoin or an N-acyl phthalimide.
- `ester` requires a carbon on the carbonyl, keeping carbamates (Boc) and
  carbonates out. `lactone` and `lactam` use a `-@` ring bond so an acyclic
  ester whose two halves sit in unrelated rings is not called cyclic.
- `orthoester` requires all three oxygens to carry a carbon, since `[OX2]` also
  matches `-OH` and accepted a gem-triol.
- `imine` bars a heteroatom on both atoms of the C=N, so oximes, hydrazones,
  amidines and guanidines keep their own names instead of also reading as
  imines. `hydrazine` excludes acylated nitrogens, which are `hydrazide`.
- `iron_complex` claims only what `[Fe]` can see. The former `ferrocene` name
  applied just as readily to a haem iron.

### Known limitations

- Five-membered rings with `[nH]` in their SMARTS (`pyrrole`, `imidazole`, `pyrazole`, `tetrazole`, `indole`, `carbazole`) **do not match N-substituted forms** (e.g. N-methylpyrrole). `benzimidazole`, `indazole`, and `purine` were already switched to a generic `n` atom; the rest are left as-is to avoid inconsistency churn.
- `cyclopentane` / `cyclohexane` SMARTS will fire on saturated sub-rings inside fused systems (e.g. steroid D-ring), producing acceptable overlap with skeleton hints.
- Two pre-existing `ruff E501` violations (lines > 120 chars): `cyclohexane`, `tbs` — not fixed.

### Deliberate non-detections

These are known blind spots, kept blind because the narrow pattern that avoids a
false name cannot also cover them. Silence is the intended behaviour.

| Not detected | Why |
| --- | --- |
| 1,2,4-triazole, 1,2,3-thiadiazole, 1,2,5-oxadiazole | `triazole` / `thiadiazole` / `oxadiazole_*` each pin one isomer's heteroatom order; the other isomers need their own entries |
| Carboxylate, phenoxide, thiolate, sulfonate anions | `carboxylic_acid`, `phenol`, `alcohol`, `thiol`, `sulfonic_acid` all require an explicit H |
| Azo, nitroso, phosphinate, trifluoroborate | No entry |
| N-sulfonyl and N-acyl imines | Collateral of the `imine` heteroatom guard described above |
| Protonated guanidines and amidines | `guanidine` / `amidine` require a neutral `[NX2]` |
| C-glycosides | `glycoside` needs an anomeric O to bridge |

### Verification

`tests/test_augmentor.py` parses every lexicon SMARTS, because a typo makes
`_compiled` return `None` and drops that entry silently for good, and asserts
both the expected and the forbidden keys for each look-alike pair listed above.

The narrowing was measured against the 1,653 unique molecules in
`outputs/datasets/enrichment.jsonl`: 675 labels were withdrawn across
528 of them and the leading quoted hint changed for 170. Every withdrawal was
then re-checked with probes written independently of the lexicon; none of them
removed a true ester, N-oxide, aromatic amine or phosphine oxide. Fragment
enrichment already written under `outputs/` predates these patterns and still
carries the withdrawn labels.


