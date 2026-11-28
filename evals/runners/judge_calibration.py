import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from agent.extensions.connects_roi_optimizer import ConnectsROIOptimizer, ROIConfig

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "judge_calibration.json"


def load_cases(path: Path) -> list[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])


def decisions_from_cases(
    cases: list[Mapping[str, object]], optimizer: ConnectsROIOptimizer
) -> list[str]:
    decisions = []
    for case in cases:
        base_prize = float(case.get("base_prize", optimizer.config.base_prize))
        has_connects = bool(case.get("has_connects", True))
        roi = optimizer.evaluate(base_prize=base_prize, connects=has_connects)
        decisions.append(roi["decision"])
    return decisions


def compute_kappa(ground: list[str], pred: list[str]) -> float:
    # Simple implementation of Cohen's kappa without external deps.
    # Compute confusion matrix.
    labels = ["accept", "reject"]
    total = len(ground)
    if total == 0:
        return 0.0
    # Count agreements.
    agree = sum(1 for g, p in zip(ground, pred, strict=False) if g == p)
    po = agree / total
    # Expected agreement.
    ground_counts = {label: ground.count(label) for label in labels}
    pred_counts = {label: pred.count(label) for label in labels}
    pe = sum((ground_counts[lbl] / total) * (pred_counts[lbl] / total) for lbl in labels)
    if pe == 1.0:
        return 1.0
    kappa = (po - pe) / (1 - pe)
    return kappa


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge calibration for Connects ROI Optimizer")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "dev_set.json",
        help="Path to the evaluation dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="File to write calibration JSON",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional JSON string to override ROIConfig parameters",
    )
    args = parser.parse_args()

    if args.config:
        cfg = ROIConfig(**json.loads(args.config))
    else:
        cfg = ROIConfig()
    optimizer = ConnectsROIOptimizer(config=cfg)

    cases = load_cases(args.dataset)
    ground = [case.get("ground_truth_decision", "reject") for case in cases]
    pred = decisions_from_cases(cases, optimizer)

    kappa = compute_kappa(ground, pred)
    result = {
        "kappa": kappa,
        "total": len(cases),
        "ground_truth_distribution": {
            "accept": ground.count("accept"),
            "reject": ground.count("reject"),
        },
        "prediction_distribution": {"accept": pred.count("accept"), "reject": pred.count("reject")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Calibration complete: {args.output}")


if __name__ == "__main__":
    main()
