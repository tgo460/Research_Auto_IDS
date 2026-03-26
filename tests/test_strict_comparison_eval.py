from src_replica.strict_comparison_eval_replica import _build_comparison


def test_build_comparison_merges_ablation_and_unimodal_reports():
    ablation = {
        "split_manifest": "data/splits/split_v3_research_valid.json",
        "pairing_mode": "label_cartesian",
        "splits": {
            "val": {
                "modes": {
                    "fused": {"f1": 0.9},
                    "can_only": {"f1": 0.7},
                    "eth_only": {"f1": 0.6},
                }
            }
        },
    }
    baseline = {
        "split_manifest": "data/splits/split_v3_research_valid.json",
        "pairing_mode": "label_cartesian",
        "modes": {
            "can_only": {"eval": {"val": {"metrics": {"f1": 0.65}}}},
            "eth_only": {"eval": {"val": {"metrics": {"f1": 0.55}}}},
        },
    }

    merged = _build_comparison(ablation, baseline)
    row = merged["splits"]["val"]
    assert row["hybrid_fused"]["f1"] == 0.9
    assert row["hybrid_can_masked"]["f1"] == 0.7
    assert row["hybrid_eth_masked"]["f1"] == 0.6
    assert row["baseline_can_only"]["f1"] == 0.65
    assert row["baseline_eth_only"]["f1"] == 0.55
