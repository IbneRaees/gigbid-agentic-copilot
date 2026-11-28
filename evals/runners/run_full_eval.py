import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from agent.extensions.connects_roi_optimizer import ConnectsROIOptimizer, ROIConfig

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "full_eval_report.json"


def load_cases(path: Path) -> list[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])


def evaluate_cases(
    cases: list[Mapping[str, object]], optimizer: ConnectsROIOptimizer
) -> list[Mapping[str, object]]:
    results = []
    for case in cases:
        base_prize = float(case.get("base_prize", optimizer.config.base_prize))
        has_connects = bool(case.get("has_connects", True))
        roi = optimizer.evaluate(base_prize=base_prize, connects=has_connects)
        result = {
            "job_id": case.get("job_id"),
            "decision": roi["decision"],
            "ground_truth": case.get("ground_truth_decision"),
            "win_probability": roi["win_probability"],
            "expected_value": roi["expected_value"],
        }
        results.append(result)
    return results


def compute_metrics(evaluations: list[Mapping[str, object]]) -> Mapping[str, float]:
    total = len(evaluations)
    if total == 0:
        return {"accuracy": 0.0}
    correct = sum(1 for ev in evaluations if ev["decision"] == ev["ground_truth"])
    accuracy = correct / total
    return {"accuracy": accuracy, "total": total, "correct": correct}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full evaluation for Connects ROI Optimizer")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "dev_set.json",
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_PATH,
        help="File to write JSON results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional JSON string to override ROIConfig parameters",
    )
    args = parser.parse_args()

    if args.config:
        cfg_dict = json.loads(args.config)
        config = ROIConfig(**cfg_dict)
    else:
        config = ROIConfig()
    optimizer = ConnectsROIOptimizer(config=config)

    cases = load_cases(args.dataset)
    evaluations = evaluate_cases(cases, optimizer)
    metrics = compute_metrics(evaluations)

    output_data = {"metrics": metrics, "evaluations": evaluations}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Evaluation completed: {args.output}")


if __name__ == "__main__":
    main()
