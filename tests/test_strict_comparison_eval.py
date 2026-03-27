from src_replica.strict_comparison_eval_replica import _build_comparison


def test_build_comparison_merges_ablation_and_unimodal_reports():
    ablation = {
        "split_manifest": "data/splits/split_v3_research_valid.json",
        "pairing_mode": "label_cartesian",
        "splits": {
            "val": {
                "label_summary": {
                    "samples": 10,
                    "positives": 10,
                    "negatives": 0,
                    "positive_rate": 1.0,
                    "classes_present": [1],
                    "is_single_class": True,
                },
                "validity_warnings": ["evaluation labels contain only the positive class"],
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
    assert row["label_summary"]["is_single_class"] is True
    assert row["is_valid_for_detection_claims"] is False
    assert merged["validity_summary"]["invalid_splits"] == ["val"]
    assert merged["validity_summary"]["research_claim_supported"] is False


def test_build_comparison_marks_placeholder_eth_labels_invalid():
    ablation = {
        "split_manifest": "data/splits/split_v3_research_valid.json",
        "pairing_mode": "label_cartesian",
        "splits": {
            "test": {
                "label_summary": {
                    "samples": 12,
                    "positives": 6,
                    "negatives": 6,
                    "positive_rate": 0.5,
                    "classes_present": [0, 1],
                    "is_single_class": False,
                },
                "validity_warnings": ["evaluation uses scenario_placeholder ETH labels"],
                "is_valid_for_detection_claims": False,
                "modes": {
                    "fused": {"f1": 0.8},
                    "can_only": {"f1": 0.7},
                    "eth_only": {"f1": 0.6},
                },
            }
        },
    }
    baseline = {
        "split_manifest": "data/splits/split_v3_research_valid.json",
        "pairing_mode": "label_cartesian",
        "modes": {
            "can_only": {"eval": {"test": {"metrics": {"f1": 0.65}}}},
            "eth_only": {"eval": {"test": {"metrics": {"f1": 0.55}}}},
        },
    }

    merged = _build_comparison(ablation, baseline)
    assert merged["splits"]["test"]["is_valid_for_detection_claims"] is False
    assert merged["validity_summary"]["invalid_splits"] == ["test"]


def test_build_comparison_marks_other_untrusted_eth_labels_invalid():
    ablation = {
        "split_manifest": "data/splits/split_v3_research_valid.json",
        "pairing_mode": "label_cartesian",
        "splits": {
            "test": {
                "label_summary": {
                    "samples": 12,
                    "positives": 6,
                    "negatives": 6,
                    "positive_rate": 0.5,
                    "classes_present": [0, 1],
                    "is_single_class": False,
                },
                "validity_warnings": ["evaluation uses untrusted ETH labels: inferred_full_session"],
                "is_valid_for_detection_claims": False,
                "modes": {
                    "fused": {"f1": 0.8},
                    "can_only": {"f1": 0.7},
                    "eth_only": {"f1": 0.6},
                },
            }
        },
    }
    baseline = {
        "split_manifest": "data/splits/split_v3_research_valid.json",
        "pairing_mode": "label_cartesian",
        "modes": {
            "can_only": {"eval": {"test": {"metrics": {"f1": 0.65}}}},
            "eth_only": {"eval": {"test": {"metrics": {"f1": 0.55}}}},
        },
    }

    merged = _build_comparison(ablation, baseline)
    assert merged["splits"]["test"]["is_valid_for_detection_claims"] is False
    assert merged["validity_summary"]["research_claim_supported"] is False
