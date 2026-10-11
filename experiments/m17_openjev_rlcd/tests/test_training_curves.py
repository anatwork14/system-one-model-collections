import json

from write_summary import paired_bootstrap, training_curves


def test_training_curves_export_main_stage_csv_and_svg(tmp_path):
    row = {
        "step": 1,
        "step_seconds": 2.5,
        "loss": -0.2,
        "mean_correct_label_probability": 0.7,
        "mean_log_ratio": 0.03,
        "grad_norm": 0.4,
        "rationale_tokens": 12,
        "mean_rationale_tokens": 6.0,
    }
    for stage in ("stage1", "stage2"):
        directory = tmp_path / stage
        directory.mkdir()
        (directory / "train.jsonl").write_text(json.dumps(row) + "\n")

    training_curves(tmp_path)

    csv_text = (tmp_path / "training_curves.csv").read_text()
    svg_text = (tmp_path / "training_curves.svg").read_text()
    assert "stage1,1,2.500000" in csv_text
    assert "stage2,1,2.500000" in csv_text
    assert "Objective loss" in svg_text
    assert "Mean probability of observed label" in svg_text
    assert "Stage 2 sampled policy/reference log ratio" in svg_text
    assert "polyline" in svg_text


def test_paired_bootstrap_is_zero_for_identical_predictions():
    labels = [0, 1, 0, 1]
    probabilities = [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]]
    result = paired_bootstrap(labels, probabilities, probabilities, reps=20)
    assert all(row["delta_candidate_minus_base"] == 0 for row in result.values())
    assert all(row["ci95_low"] == row["ci95_high"] == 0 for row in result.values())
