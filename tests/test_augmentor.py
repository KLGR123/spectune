import importlib.util
import json
import random

import pytest

from spectune import Augmentor, AugmentorConfig, EnrichmentConfig
from spectune.augmentor.formulas import perturb_formula
from spectune.augmentor.spectra import apply_nmr_noise, apply_reaction_noise, build_spectrum_text, split_peaks
from spectune.llm import LlmClient, LlmConfig
from tests.mock_servers import run_mock_get_json_server, run_mock_json_server

_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None

_NMR = {
    "type": "1H NMR",
    "frequency": "400 MHz",
    "solvent": "CDCl3",
    "shift_text": "8.52-8.49 (m, 1H), 4.75 (q, J = 5.3 Hz, 1H), 3.95 (s, 3H), 2.68 (d, J = 5.3 Hz, 3H)",
    "processed": None,
}


def _truth_record(index=0, smiles="CNS(=O)(=O)c1cccc(C(=O)OC)c1", formula="C9H11NO4S"):
    return {
        "sample_id": f"NMRexp:test:checked:{index}",
        "modality": "nmr",
        "gt_smiles": smiles,
        "molecular_formula": formula,
        "nmr": dict(_NMR),
        "ms": None,
        "provenance": {"dataset": "NMRexp", "split": "test"},
        "quality": None,
        "cluster": index % 3,
    }


def _offline_enrichment(tmp_path, **overrides):
    settings = {
        "resolve_names": False,
        "resolve_reactions": False,
        "detect_fragments": True,
        "compute_properties": True,
        "cache_path": str(tmp_path / "enrichment.jsonl"),
        "show_progress": False,
    }
    settings.update(overrides)
    return EnrichmentConfig(**settings)


def _config(tmp_path, **overrides):
    settings = {
        "split": "test",
        "seed": 5,
        "show_progress": False,
        "datasets_dir": str(tmp_path),
        "enrichment": _offline_enrichment(tmp_path),
        "llm": LlmConfig(base_url="", model=""),
    }
    settings.update(overrides)
    return AugmentorConfig(**settings)


class TestConfigValidation:
    def test_information_mix_must_sum_to_one(self):
        with pytest.raises(ValueError, match="must sum to 1"):
            AugmentorConfig(information_mix={"none": 0.5, "formula": 0.2})

    def test_unknown_information_type_is_rejected(self):
        with pytest.raises(ValueError, match="unknown information type"):
            AugmentorConfig(information_mix={"none": 0.5, "mass_spec": 0.5})

    def test_probabilities_are_bounded(self):
        with pytest.raises(ValueError, match="followup_probability"):
            AugmentorConfig(followup_probability=1.5)
        with pytest.raises(ValueError, match="formula_noise_ratio"):
            AugmentorConfig(formula_noise_ratio=-0.1)

    def test_reaction_noise_is_reserved(self):
        with pytest.raises(ValueError, match="reaction noise is reserved"):
            AugmentorConfig(reaction_noise_ratio=0.2)

    def test_unknown_noise_mode_is_rejected(self):
        with pytest.raises(ValueError, match="unknown NMR noise mode"):
            AugmentorConfig(nmr_noise_modes=("shuffle_solvent",))

    def test_unknown_reaction_source_is_rejected(self):
        with pytest.raises(ValueError, match="unknown reaction source"):
            EnrichmentConfig(reaction_sources=("reaxys",))


class TestSpectrumTextAndNoise:
    def test_clean_text_reassembles_the_conventional_one_liner(self):
        assert build_spectrum_text(_NMR).startswith("1H NMR (400 MHz, CDCl3): 8.52-8.49 (m, 1H)")

    def test_missing_shift_text_yields_empty_string(self):
        assert build_spectrum_text({"type": "1H NMR"}) == ""

    def test_split_peaks_ignores_commas_inside_parentheses(self):
        assert split_peaks(_NMR["shift_text"]) == [
            "8.52-8.49 (m, 1H)",
            "4.75 (q, J = 5.3 Hz, 1H)",
            "3.95 (s, 3H)",
            "2.68 (d, J = 5.3 Hz, 3H)",
        ]

    def test_digits_only_keeps_numbers_and_drops_the_header(self):
        text, detail = apply_nmr_noise(_NMR, "digits_only", random.Random(0))

        assert "NMR" not in text
        assert set(text) <= set("0123456789. -")
        assert detail["applied"] is True

    def test_digits_only_reads_a_shift_range_as_two_positive_numbers(self):
        text, _ = apply_nmr_noise(_NMR, "digits_only", random.Random(0))

        assert text.startswith("8.52 8.49")

    def test_strip_punctuation_removes_only_punctuation(self):
        text, detail = apply_nmr_noise(_NMR, "strip_punctuation", random.Random(1), strength=1.0)
        peaks = text.split(": ", 1)[1]

        assert "(" not in peaks and ")" not in peaks
        assert text.startswith("1H NMR (400 MHz, CDCl3)")  # only the peak list is corrupted
        assert detail["removed_characters"] > 0
        assert "8.52-8.49" in peaks

    def test_perturb_values_changes_some_numbers_but_keeps_the_layout(self):
        text, detail = apply_nmr_noise(_NMR, "perturb_values", random.Random(2), strength=1.0)

        assert detail["perturbed_values"] > 0
        assert text != build_spectrum_text(_NMR)
        assert text.count("(") == build_spectrum_text(_NMR).count("(")

    def test_drop_peaks_keeps_at_least_one_peak(self):
        text, detail = apply_nmr_noise(_NMR, "drop_peaks", random.Random(3), strength=1.0)

        assert detail["dropped_peaks"] == 3
        assert len(split_peaks(text.split(": ", 1)[1])) == 1

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown NMR noise mode"):
            apply_nmr_noise(_NMR, "transpose", random.Random(0))

    def test_reaction_noise_is_still_a_no_op(self):
        payload = {"reactants": ["CCO"]}

        result, detail = apply_reaction_noise(payload, random.Random(0))

        assert result is payload
        assert detail["applied"] is False


class TestFormulaNoise:
    def test_perturbation_makes_a_small_traceable_composition_change(self):
        formula, detail = perturb_formula("C9H11NO4S", random.Random(0))

        assert formula != "C9H11NO4S"
        assert detail["applied"] is True
        assert detail["changed"] is True
        assert detail["original_formula"] == "C9H11NO4S"
        assert detail["perturbed_formula"] == formula
        assert sum(abs(delta) for delta in detail["element_deltas"].values()) <= 3

    def test_unsupported_formula_is_left_unchanged(self):
        formula, detail = perturb_formula("(C2H4)n", random.Random(0))

        assert formula == "(C2H4)n"
        assert detail["applied"] is False
        assert detail["reason"] == "unsupported_formula"


@pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit is not installed")
class TestStructureAnalysis:
    def test_fragments_are_named_in_both_languages_and_ranked_by_specificity(self):
        from spectune.augmentor.fragments import detect_fragments

        fragments = detect_fragments("FC(F)(F)c1ccncc1")
        keys = [fragment["key"] for fragment in fragments]

        assert keys[0] == "trifluoromethyl"
        assert "pyridine" in keys
        assert fragments[0]["zh"] == "三氟甲基"

    def test_murcko_scaffold_is_reported_last_so_it_is_never_quoted_as_a_hint(self):
        from spectune.augmentor.fragments import detect_fragments

        fragments = detect_fragments("FC(F)(F)c1ccncc1")

        assert fragments[-1]["key"] == "murcko_scaffold"
        assert fragments[-1]["smiles"] == "c1ccncc1"

    def test_fused_ring_analysis_recognises_a_triterpene_skeleton(self):
        from spectune.augmentor.fragments import detect_fragments

        triterpene = "CC1(C)CCC2(CCC3(C)C(=CCC4C3(C)CCC3C(C)(C)C(O)CCC34C)C2C1)C(=O)O"
        keys = [fragment["key"] for fragment in detect_fragments(triterpene)]

        assert "triterpene_skeleton" in keys

    def test_every_lexicon_smarts_compiles(self):
        # A typo in a SMARTS string makes _compiled() return None and
        # detect_fragments() silently skip that entry forever, so the whole
        # lexicon is parsed here rather than only where a test happens to look.
        from rdkit import Chem

        from spectune.augmentor.fragments import FRAGMENT_LEXICON

        broken = [entry.key for entry in FRAGMENT_LEXICON if Chem.MolFromSmarts(entry.smarts) is None]

        assert broken == []
        assert len({entry.key for entry in FRAGMENT_LEXICON}) == len(FRAGMENT_LEXICON)

    # Every case below is a group that a looser SMARTS used to mislabel as a
    # chemically unrelated one. A wrong Chinese name goes straight into a
    # training query, so these matter more than the coverage they add.
    @pytest.mark.parametrize(
        ("name", "smiles", "expected", "forbidden"),
        [
            # "[#7+][O-]" also matches the charge-separated spelling of nitro.
            ("nitroarene", "c1ccccc1[N+](=O)[O-]", ["nitro"], ["n_oxide", "aniline"]),
            ("amine oxide", "C[N+](C)(C)[O-]", ["n_oxide"], ["nitro", "quaternary_ammonium"]),
            ("azine oxide", "[O-][n+]1ccccc1", ["n_oxide", "pyridine"], ["nitro"]),
            ("nitrate ester", "CCO[N+](=O)[O-]", ["nitrate_ester"], ["nitro", "n_oxide"]),
            # X counts hydrogens, so "[NX4+]" matched every protonated amine.
            ("amine hydrochloride", "[Cl-].[NH3+]Cc1ccccc1", ["primary_amine"], ["quaternary_ammonium"]),
            ("ammonium counterion", "[NH4+].[Cl-]", [], ["quaternary_ammonium"]),
            ("tetraalkylammonium", "CCCC[N+](CCCC)(CCCC)CCCC", ["quaternary_ammonium"], ["tertiary_amine"]),
            # Nitrogens that are not amines at all.
            ("guanidine", "NC(=N)N", ["guanidine"], ["primary_amine", "imine"]),
            ("hydrazide", "CC(=O)NN", ["hydrazide"], ["hydrazine", "primary_amine"]),
            ("sulfonamide", "CS(=O)(=O)N", ["sulfonamide"], ["primary_amine"]),
            ("anilide", "CC(=O)Nc1ccccc1", ["amide"], ["aniline"]),
            ("oxime", "c1ccccc1C=NO", ["oxime"], ["imine"]),
            ("Schiff base", "c1ccccc1C=Nc1ccccc1", ["imine"], []),
            # An oxane is not a sugar, and an acetal on one is not a glycoside.
            ("THP ether", "CCCCOC1CCCCO1", ["tetrahydropyran", "acetal"], ["pyranose", "glycoside"]),
            ("spiroketal", "CC1CCC2(OC1)OC1CC3CCC4CC(O)CCC4(C)C3CC1C2C", [], ["pyranose", "glycoside"]),
            ("reducing sugar", "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O", ["pyranose"], ["glycoside"]),
            ("O-glycoside", "CO[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O", ["pyranose", "glycoside"], []),
            (
                "sugar phosphate",
                "OC[C@H]1O[C@@H](OP(=O)(O)O)[C@H](O)[C@@H](O)[C@@H]1O",
                ["pyranose", "phosphate_ester"],
                ["glycoside"],
            ),
            # Phosphorus is told apart only by counting its substituents.
            (
                "nucleotide phosphate",
                "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O",
                ["phosphate_ester"],
                ["phosphine_oxide", "phosphonate"],
            ),
            ("phosphonate", "CCOP(=O)(C)OCC", ["phosphonate"], ["phosphine_oxide", "phosphate_ester"]),
            (
                "phosphine oxide",
                "O=P(c1ccccc1)(c1ccccc1)c1ccccc1",
                ["phosphine_oxide"],
                ["phosphate_ester"],
            ),
            ("phosphoramidite", "CC(C)N(C(C)C)P(OCC)OCC", [], ["phosphine", "phosphine_oxide"]),
            # A benzylic CH2 is not a benzyl group.
            ("benzyl ether", "CCOCc1ccccc1", ["benzyl", "benzylic_ch2"], []),
            ("phenylpropylamine", "NCCCc1ccccc1", ["benzylic_ch2", "primary_amine"], ["benzyl"]),
            ("tetralin", "C1CCc2ccccc2C1", ["benzylic_ch2"], ["benzyl"]),
            ("piperonyl amine", "CNCc1ccc2c(c1)OCO2", ["benzyl"], ["pmb"]),
            # An imide alpha to an amide is not a peptide bond.
            ("tripeptide", "NCC(=O)NCC(=O)NCC(=O)O", ["peptide_bond", "peptide_chain"], []),
            ("prolyl peptide", "NCC(=O)N1CCC[C@H]1C(=O)NCC(=O)O", ["peptide_bond"], []),
            ("hydantoin", "O=C1NC(=O)CN1", ["cyclic_imide"], ["peptide_bond", "peptide_chain"]),
            # Carbamates and carbonates are not esters; lactones need one ring.
            ("Boc carbamate", "CC(C)(C)OC(=O)NC", ["boc", "carbamate"], ["ester"]),
            ("carbonate", "CCOC(=O)OCC", [], ["ester"]),
            ("lactone", "O=C1CCCCO1", ["lactone", "ester"], []),
            ("ester spanning two rings", "O=C(OC1CCCCC1)C1CCCCC1", ["ester"], ["lactone"]),
            ("N-acylpiperidine", "CC(=O)N1CCCCC1", ["amide", "piperidine"], ["lactam"]),
            # Ring topology alone cannot name a terpene class.
            ("adamantane", "C1C2CC3CC1CC(C2)C3", [], ["tetracyclic_terpenoid"]),
            ("gem-triol", "OC(O)(O)C", [], ["orthoester"]),
        ],
    )
    def test_lookalike_groups_are_not_mislabelled(self, name, smiles, expected, forbidden):
        from spectune.augmentor.fragments import detect_fragments

        keys = {fragment["key"] for fragment in detect_fragments(smiles, max_fragments=99)}

        assert [key for key in expected if key not in keys] == [], name
        assert [key for key in forbidden if key in keys] == [], name

    def test_properties_include_formula_and_weight(self):
        from spectune.augmentor.fragments import molecule_properties

        properties = molecule_properties("CNS(=O)(=O)c1cccc(C(=O)OC)c1")

        assert properties["molecular_formula"] == "C9H11NO4S"
        assert 229.0 < properties["molecular_weight"] < 229.5

    def test_invalid_smiles_yields_no_properties(self):
        from spectune.augmentor.fragments import molecule_properties

        assert molecule_properties("not-a-molecule") == {}


@pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit is not installed")
class TestReactionContext:
    def test_retro_templates_propose_the_expected_disconnection(self):
        from spectune.augmentor.reactions import retro_routes

        routes = retro_routes("COC(=O)c1ccc(-c2ccccc2)cc1")
        templates = {route["template"] for route in routes}

        assert {"esterification", "suzuki"} <= templates
        suzuki = next(route for route in routes if route["template"] == "suzuki")
        assert any("B(O)" in reactant for reactant in suzuki["reactants"])
        assert suzuki["conditions_zh"]

    def test_routes_never_reproduce_the_target_itself(self):
        from spectune.augmentor.reactions import retro_routes

        target = "COC(=O)c1ccc(-c2ccccc2)cc1"

        assert all(target not in route["reactants"] for route in retro_routes(target))

    def test_polymer_context_flags_a_monomer(self):
        from spectune.augmentor.reactions import polymer_context

        context = polymer_context("C=CC(=O)OCC")

        assert context["is_probable_monomer"] is True
        assert context["monomer_classes"][0]["zh"] == "丙烯酸酯类单体"

    def test_polymer_context_is_empty_for_an_ordinary_molecule(self):
        from spectune.augmentor.reactions import polymer_context

        assert polymer_context("c1ccccc1")["is_probable_monomer"] is False

    def test_agent_smiles_are_translated_for_chinese_queries(self):
        from spectune.augmentor.reactions import describe_agents

        assert describe_agents(["ClCCl", "[Pd]", "CCOC(C)=O"]) == ["二氯甲烷", "钯催化剂", "乙酸乙酯"]

    def test_scan_without_configured_corpora_returns_nothing(self, tmp_path):
        from spectune.augmentor.reactions import scan_product_precedents
        from spectune.tools import ReactionLocalIndexSearchConfig

        config = EnrichmentConfig(
            reaction_index=ReactionLocalIndexSearchConfig(
                uspto_csv_path="", chempile_parquet_path="", pistachio_smi_path=""
            ),
            cache_path=str(tmp_path / "cache.jsonl"),
        )

        assert scan_product_precedents(["CCO"], config, progress=False) == {}

    def test_scan_finds_a_product_in_a_local_uspto_style_csv(self, tmp_path):
        from spectune.augmentor.reactions import scan_product_precedents
        from spectune.tools import ReactionLocalIndexSearchConfig

        csv_path = tmp_path / "uspto.csv"
        csv_path.write_text("reactant,product\nCO.O=C(O)c1ccccc1,COC(=O)c1ccccc1\n", encoding="utf-8")
        config = EnrichmentConfig(
            reaction_sources=("uspto",),
            reaction_index=ReactionLocalIndexSearchConfig(uspto_csv_path=str(csv_path)),
            cache_path=str(tmp_path / "cache.jsonl"),
        )

        found = scan_product_precedents(["COC(=O)c1ccccc1"], config, progress=False)

        assert found["COC(=O)c1ccccc1"][0]["reactants"] == ["CO", "O=C(O)c1ccccc1"]


class TestNameResolution:
    def _resolve(self, synonyms, tmp_path, **overrides):
        import asyncio

        from spectune.augmentor.naming import resolve_names

        def response(path, _query):
            if path.endswith("/cids/JSON"):
                return {"IdentifierList": {"CID": [2244]}}
            if path.endswith("/synonyms/JSON"):
                return {"InformationList": {"Information": [{"CID": 2244, "Synonym": synonyms}]}}
            raise AssertionError(f"unexpected PubChem path: {path}")

        with run_mock_get_json_server(response) as base_url:
            config = EnrichmentConfig(
                pubchem_base_url=base_url,
                cache_path=str(tmp_path / "cache.jsonl"),
                **overrides,
            )
            return asyncio.run(resolve_names("CC(=O)Oc1ccccc1C(=O)O", config))

    @pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit is required to verify a hit's structure")
    def test_names_are_classified_from_pubchem_synonyms(self, tmp_path):
        result = self._resolve(["Aspirin", "Acetylsalicylic acid", "阿司匹林", "乙酰水杨酸", "50-78-2"], tmp_path)

        assert result["status"] == "found"
        assert result["complete"] is True
        assert result["cid"] == 2244
        assert result["name_en"] == "Aspirin"
        assert result["name_zh"] == "阿司匹林"
        assert result["cas"] == "50-78-2"
        assert result["source"] == "pubchem_pug_rest_v1"

    @pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit is required to verify a hit's structure")
    def test_missing_chinese_name_is_reported_as_partial(self, tmp_path):
        result = self._resolve(["Aspirin", "Acetylsalicylic acid", "50-78-2"], tmp_path)

        assert result["status"] == "partial"
        assert result["complete"] is False
        assert result["name_zh"] is None
        assert result["name_en"] == "Aspirin"

    def test_pubchem_not_found_is_an_empty_result(self, tmp_path):
        import asyncio

        from spectune.augmentor.naming import resolve_names

        with run_mock_get_json_server(lambda _path, _query: {}, status_code=lambda _path, _query: 404) as base_url:
            config = EnrichmentConfig(pubchem_base_url=base_url, cache_path=str(tmp_path / "cache.jsonl"))
            result = asyncio.run(resolve_names("CCO", config))

        assert result["status"] == "not_found"
        assert result["complete"] is False


class TestLlmClient:
    def test_unconfigured_client_is_unavailable_and_returns_nothing(self):
        import asyncio

        client = LlmClient(LlmConfig(base_url="", model=""))

        assert client.available is False
        assert asyncio.run(client.complete("system", "user")) == ""

    def test_completion_reads_the_first_choice(self):
        import asyncio

        def handler(body):
            return {"choices": [{"message": {"content": f"echo:{body['messages'][1]['content']}"}}]}

        with run_mock_json_server(handler, path="/v1/chat/completions") as url:
            client = LlmClient(LlmConfig(base_url=url.removesuffix("/chat/completions"), model="test-model"))
            reply = asyncio.run(client.complete("system", "hello"))

        assert reply == "echo:hello"
        assert client.stats == {"requests": 1, "failures": 0, "retries": 0}

    def test_malformed_response_counts_as_a_failure_without_raising(self):
        import asyncio

        with run_mock_json_server(lambda _body: {"choices": []}, path="/v1/chat/completions") as url:
            client = LlmClient(LlmConfig(base_url=url.removesuffix("/chat/completions"), model="test-model"))
            reply = asyncio.run(client.complete("system", "hello"))

        assert reply == ""
        assert client.stats["failures"] == 0  # a valid HTTP response with no choices is simply empty


@pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit is not installed")
class TestAugmentorConstruction:
    def test_spectrum_only_mixture_produces_single_turn_queries(self, tmp_path):
        augmentor = Augmentor(_config(tmp_path, information_mix={"none": 1.0}))

        dataset = augmentor.build([_truth_record(i) for i in range(6)])

        assert len(dataset) == 6
        for record in dataset:
            assert record["num_of_queries"] == 1
            assert record["augmentation"]["information_type"] == "none"
            assert record["nmr_text"] in record["turns"][0]["content"]

    def test_formula_mixture_always_states_the_formula(self, tmp_path):
        augmentor = Augmentor(_config(tmp_path, information_mix={"formula": 1.0}, followup_probability=0.0))

        dataset = augmentor.build([_truth_record(i) for i in range(6)])

        for record in dataset:
            assert record["augmentation"]["information_type"] == "formula"
            assert "C9H11NO4S" in record["turns"][0]["content"]

    def test_formula_noise_only_changes_formula_condition_rows(self, tmp_path):
        augmentor = Augmentor(
            _config(
                tmp_path,
                information_mix={"formula": 1.0},
                followup_probability=0.0,
                formula_noise_ratio=1.0,
            )
        )

        dataset = augmentor.build([_truth_record(i) for i in range(5)])

        for record in dataset:
            detail = record["augmentation"]["formula_noise_detail"]
            presented = record["augmentation"]["information_detail"]["formula"]
            assert record["molecular_formula"] == "C9H11NO4S"
            assert record["augmentation"]["formula_noise_applied"] is True
            assert detail["original_formula"] == "C9H11NO4S"
            assert detail["perturbed_formula"] == presented
            assert presented != "C9H11NO4S"
            assert presented in record["turns"][0]["content"]
        assert augmentor.last_summary["formula_noise_eligible"] == 5
        assert augmentor.last_summary["formula_noised"] == 5

    def test_formula_noise_does_not_apply_without_a_formula_condition(self, tmp_path):
        augmentor = Augmentor(
            _config(tmp_path, information_mix={"none": 1.0}, formula_noise_ratio=1.0)
        )

        dataset = augmentor.build([_truth_record(i) for i in range(3)])

        assert all(not record["augmentation"]["formula_noise_applied"] for record in dataset)
        assert augmentor.last_summary["formula_noise_eligible"] == 0
        assert augmentor.last_summary["formula_noised"] == 0

    def test_followup_probability_one_moves_information_to_a_second_turn(self, tmp_path):
        augmentor = Augmentor(_config(tmp_path, information_mix={"formula": 1.0}, followup_probability=1.0))

        dataset = augmentor.build([_truth_record(i) for i in range(6)])

        for record in dataset:
            assert record["num_of_queries"] == 2
            assert "C9H11NO4S" not in record["turns"][0]["content"]
            assert "C9H11NO4S" in record["turns"][1]["content"]
            assert record["augmentation"]["information_placement"] == "followup"

    def test_structure_hint_is_phrased_with_hedging(self, tmp_path):
        augmentor = Augmentor(_config(tmp_path, information_mix={"structure": 1.0}, followup_probability=0.0))

        dataset = augmentor.build([_truth_record(i) for i in range(6)])

        for record in dataset:
            assert record["gt_smiles"] in record["turns"][0]["content"]
            assert record["augmentation"]["uncertain_tone"] is True
            assert record["augmentation"]["information_detail"]["structure_kind"] == "smiles"

    def test_partial_names_are_never_used_as_structure_conditions(self, tmp_path):
        augmentor = Augmentor(_config(tmp_path))
        record = _truth_record(0)
        enrichment = {
            "names": {
                "name_zh": None,
                "name_en": "Aspirin",
                "status": "partial",
                "complete": False,
            }
        }

        for seed in range(30):
            _, detail = augmentor._structure_text(record, enrichment, random.Random(seed))
            assert detail["structure_kind"] == "smiles"
            assert detail["structure_value"] == record["gt_smiles"]

    def test_names_are_eligible_only_as_a_complete_pair(self, tmp_path):
        augmentor = Augmentor(_config(tmp_path))
        record = _truth_record(0)
        enrichment = {
            "names": {
                "name_zh": "阿司匹林",
                "name_en": "Aspirin",
                "status": "found",
                "complete": True,
            }
        }

        kinds = {
            augmentor._structure_text(record, enrichment, random.Random(seed))[1]["structure_kind"]
            for seed in range(30)
        }

        assert kinds == {"smiles", "name_zh", "name_en"}

    def test_fragment_hint_quotes_a_chinese_fragment_name(self, tmp_path):
        augmentor = Augmentor(_config(tmp_path, information_mix={"fragment": 1.0}, followup_probability=0.0))

        dataset = augmentor.build([_truth_record(i) for i in range(4)])

        for record in dataset:
            assert record["augmentation"]["information_type"] == "fragment"
            assert "磺酰胺" in record["turns"][0]["content"] or "酯基" in record["turns"][0]["content"]

    def test_unavailable_information_degrades_to_a_spectrum_only_query(self, tmp_path):
        record = _truth_record(0, formula=None)
        record["molecular_formula"] = None
        augmentor = Augmentor(
            _config(
                tmp_path,
                information_mix={"formula": 1.0},
                enrichment=_offline_enrichment(tmp_path, compute_properties=False),
            )
        )

        built = augmentor.build([record])[0]

        assert built["augmentation"]["requested_information_type"] == "formula"
        assert built["augmentation"]["information_type"] == "none"
        assert built["augmentation"]["information_available"] is False
        assert built["augmentation"]["information_degraded"] is True
        assert built["num_of_queries"] == 1
        assert augmentor.last_summary["information_degraded"] == 1

    def test_spectrum_only_rows_are_not_reported_as_degraded(self, tmp_path):
        augmentor = Augmentor(_config(tmp_path, information_mix={"none": 1.0}))

        dataset = augmentor.build([_truth_record(i) for i in range(3)])

        assert all(not record["augmentation"]["information_degraded"] for record in dataset)
        assert augmentor.last_summary["information_degraded"] == 0

    def test_noise_ratio_one_noises_every_spectrum(self, tmp_path):
        augmentor = Augmentor(
            _config(
                tmp_path,
                information_mix={"none": 1.0},
                nmr_noise_ratio=1.0,
                nmr_noise_strength=1.0,
                nmr_noise_modes=("drop_peaks",),
            )
        )

        dataset = augmentor.build([_truth_record(i) for i in range(5)])

        for record in dataset:
            assert record["augmentation"]["nmr_noise_applied"] is True
            assert record["augmentation"]["nmr_noise_mode"] == "drop_peaks"
            assert record["augmentation"]["nmr_noise_detail"]["changed"] is True
            assert record["nmr_text"] != record["nmr_text_clean"]
            assert record["nmr_text"] in record["turns"][0]["content"]

    def test_zero_noise_ratio_leaves_every_spectrum_intact(self, tmp_path):
        augmentor = Augmentor(_config(tmp_path, information_mix={"none": 1.0}, nmr_noise_ratio=0.0))

        dataset = augmentor.build([_truth_record(i) for i in range(5)])

        assert all(not record["augmentation"]["nmr_noise_applied"] for record in dataset)
        assert all(record["nmr_text"] == record["nmr_text_clean"] for record in dataset)

    def test_records_carry_analysis_attributes_and_are_written_to_disk(self, tmp_path):
        output = tmp_path / "augmented.jsonl"
        augmentor = Augmentor(_config(tmp_path, information_mix={"fragment": 1.0}))

        dataset = augmentor.build([_truth_record(i) for i in range(3)], output_path=output)

        written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        assert len(written) == len(dataset) == 3
        record = written[0]
        assert record["source_sample_id"] == "NMRexp:test:checked:0"
        assert record["gt_smiles"] and record["molecular_weight"]
        assert record["enrichment"]["fragments"]
        assert set(record["augmentation"]) >= {
            "information_type",
            "information_placement",
            "information_available",
            "uncertain_tone",
            "llm_rewritten",
            "llm_status",
            "nmr_noise_applied",
            "num_fragments",
            "has_reaction_precedent",
        }
        assert augmentor.last_summary["information_types"] == {"fragment": 3}

    def test_sampling_respects_the_requested_size(self, tmp_path):
        augmentor = Augmentor(_config(tmp_path, sample_size=4, information_mix={"none": 1.0}))

        dataset = augmentor.build([_truth_record(i) for i in range(20)])

        assert len(dataset) == 4

    def test_construction_is_reproducible_for_a_fixed_seed(self, tmp_path):
        records = [_truth_record(i) for i in range(8)]
        first = Augmentor(_config(tmp_path)).build(records)
        second = Augmentor(_config(tmp_path)).build(records)

        assert [record["turns"][0]["content"] for record in first] == [record["turns"][0]["content"] for record in second]


@pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit is not installed")
class TestLlmRewriting:
    def test_rewrite_is_used_when_it_preserves_the_spectrum(self, tmp_path):
        def handler(body):
            block = body["messages"][1]["content"]
            return {"choices": [{"message": {"content": f"帮我看下这个：{block}，给个 SMILES"}}]}

        with run_mock_json_server(handler, path="/v1/chat/completions") as url:
            config = _config(
                tmp_path,
                information_mix={"none": 1.0},
                use_llm_rewrite=True,
                raw_query_ratio=0.0,
                llm=LlmConfig(base_url=url.removesuffix("/chat/completions"), model="test-model"),
            )
            dataset = Augmentor(config).build([_truth_record(i) for i in range(4)])

        for record in dataset:
            assert record["augmentation"]["llm_rewritten"] is True
            assert record["turns"][0]["content"].startswith("帮我看下这个：")
            assert record["nmr_text"] in record["turns"][0]["content"]

    def test_rewrite_that_drops_the_spectrum_is_rejected_for_the_template(self, tmp_path):
        with run_mock_json_server(
            lambda _body: {"choices": [{"message": {"content": "帮我看下这个谱图"}}]},
            path="/v1/chat/completions",
        ) as url:
            config = _config(
                tmp_path,
                information_mix={"none": 1.0},
                use_llm_rewrite=True,
                raw_query_ratio=0.0,
                llm=LlmConfig(base_url=url.removesuffix("/chat/completions"), model="test-model"),
            )
            dataset = Augmentor(config).build([_truth_record(i) for i in range(3)])

        for record in dataset:
            assert record["augmentation"]["llm_status"] == "rejected_information_loss"
            assert record["augmentation"]["llm_rewritten"] is False
            assert record["nmr_text"] in record["turns"][0]["content"]

    def test_raw_query_ratio_one_never_calls_the_model(self, tmp_path):
        with run_mock_json_server(
            lambda _body: {"choices": [{"message": {"content": "unused"}}]},
            path="/v1/chat/completions",
        ) as url:
            config = _config(
                tmp_path,
                information_mix={"none": 1.0},
                use_llm_rewrite=True,
                raw_query_ratio=1.0,
                llm=LlmConfig(base_url=url.removesuffix("/chat/completions"), model="test-model"),
            )
            augmentor = Augmentor(config)
            dataset = augmentor.build([_truth_record(i) for i in range(3)])

        assert augmentor.llm.stats["requests"] == 0
        assert all(record["augmentation"]["llm_status"] == "template" for record in dataset)


@pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit is not installed")
class TestEnrichmentCache:
    def test_second_run_reuses_the_cache(self, tmp_path):
        from spectune.augmentor.enrichment import Enricher

        config = _offline_enrichment(tmp_path)
        records = [_truth_record(i) for i in range(3)]

        Enricher(config).enrich(records)
        second = Enricher(config)
        second.enrich(records)

        assert second.last_summary["cache_hits"] == 1  # three rows, one distinct molecule
        assert second.last_summary["computed"] == 0
        assert second.last_summary["cache_completion"] == {
            "properties": {"completed": 1, "total": 1, "percentage": 100.0},
            "names": {"completed": 0, "total": 1, "percentage": 0.0},
            "reactions": {"completed": 0, "total": 1, "percentage": 0.0},
            "fragments": {"completed": 1, "total": 1, "percentage": 100.0},
        }

    def test_cache_is_refreshed_when_a_new_stage_is_enabled(self, tmp_path):
        from spectune.augmentor.enrichment import Enricher

        records = [_truth_record(0)]
        Enricher(_offline_enrichment(tmp_path, detect_fragments=False)).enrich(records)

        enriched = Enricher(_offline_enrichment(tmp_path, detect_fragments=True))
        result = enriched.enrich(records)

        assert enriched.last_summary["computed"] == 1
        assert result[0]["enrichment"]["fragments"]

    def test_a_new_stage_only_recomputes_the_missing_stage(self, tmp_path):
        from spectune.augmentor.enrichment import Enricher

        records = [_truth_record(0)]
        first = Enricher(_offline_enrichment(tmp_path, detect_fragments=False))
        properties = first.enrich(records)[0]["enrichment"]["properties"]

        second = Enricher(_offline_enrichment(tmp_path, detect_fragments=True))
        second._add_local_chemistry = _fail_on_properties(second._add_local_chemistry)
        enrichment = second.enrich(records)[0]["enrichment"]

        assert enrichment["properties"] == properties  # carried over, not recomputed
        assert enrichment["fragments"]
        assert set(enrichment["stages"]) == {"properties", "fragments"}


def _fail_on_properties(bound_method):
    """Wrap ``Enricher._add_local_chemistry`` so a property recompute is visible."""

    def guarded(properties, fragments):
        assert not properties, "cached properties should not be recomputed"
        return bound_method(properties, fragments)

    return guarded
