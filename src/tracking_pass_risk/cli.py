from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .features import build_pass_features
from .model import train
from .visualise import risk_map


def run(events: Path, home: Path, away: Path, output: Path, epochs: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    table = build_pass_features(events, home, away)
    model, scaler, result, test_probability = train(table, epochs=epochs)
    test_start = int(len(table) * 0.85)
    table["split"] = "train"
    table.loc[int(len(table) * 0.7) : test_start - 1, "split"] = "validation"
    table.loc[test_start:, "split"] = "test"
    table["predicted_turnover_risk"] = float("nan")
    table.loc[test_start:, "predicted_turnover_risk"] = test_probability
    table.to_csv(output / "pass_features.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(result.as_dict(), indent=2) + "\n")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": __import__("tracking_pass_risk.features", fromlist=["FEATURES"]).FEATURES,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        output / "pass_risk_model.pt",
    )
    risk_map(table[table["split"] == "test"], output / "pass_risk_map.png")
    print(json.dumps(result.as_dict(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    parser.add_argument("home_tracking", type=Path)
    parser.add_argument("away_tracking", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=160)
    args = parser.parse_args()
    run(args.events, args.home_tracking, args.away_tracking, args.output, args.epochs)


if __name__ == "__main__":
    main()

