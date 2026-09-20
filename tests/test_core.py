from rdkit import Chem

from nmr_craft.chemistry import cut_non_ring, stitch_fragments
from nmr_craft.ranking import candidate_pool, rank_prior_support


def test_cut_and_stitch_round_trip() -> None:
    molecule = Chem.MolFromSmiles("CCOC")
    for left, cut_type in cut_non_ring(molecule):
        for right, other_type in cut_non_ring(molecule):
            if cut_type[0] == (other_type[0][1], other_type[0][0]):
                try:
                    child = stitch_fragments(left, right, cut_type[1])
                    Chem.SanitizeMol(child)
                except Exception:
                    continue
                assert child.GetNumAtoms() > 0
                return
    raise AssertionError("no stitchable fragments")


def test_sample_likelihood_changes_sample_rank_only() -> None:
    rows = {
        "db": {"candidates": [{"pred_smiles": "CC"}]},
        "beam": {"candidates": [{"pred_smiles": "CCC", "inverse_mean_logprob": -0.2}]},
        "sample": {"candidates": [
            {"pred_smiles": "CCCC", "inverse_mean_logprob": -2.0},
            {"pred_smiles": "CCCCC", "inverse_mean_logprob": -0.1},
        ]},
    }
    parents = candidate_pool(rows, {"db": 1, "beam": 1, "sample": 2})
    assert parents["CCC"]["prior"] == 1.0 / 5.0
    assert parents["CC"]["prior"] == 0.35 / 5.0
    assert parents["CCCCC"]["prior"] > parents["CCCC"]["prior"]


def test_prior_support_ranking() -> None:
    items = {
        "a": {"pair_prior": 0.2, "support": 1},
        "b": {"pair_prior": 0.1, "support": 5},
    }
    assert set(rank_prior_support(items)) == {"a", "b"}

