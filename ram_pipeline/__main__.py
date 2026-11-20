"""Single command-line entry point for batch preparation, training, and prediction."""

import argparse
import json
from pathlib import Path

from .features import INITIAL, prepare_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "predict"):
        sub = modes.add_parser(mode)
        sub.add_argument("--data", type=Path, required=True)
        sub.add_argument("--history", type=Path, action="append", default=[])
        sub.add_argument("--out", type=Path, required=True)
        if mode == "predict":
            sub.add_argument("--model", type=Path, default=Path("models/model_v1.joblib"))
    train = modes.add_parser("train")
    train.add_argument("--data", type=Path, default=INITIAL)
    train.add_argument("--out-dir", type=Path, default=Path("."))
    train.add_argument("--date", required=True, help="Дата регистрации запуска в формате ISO.")
    args = parser.parse_args()
    if args.mode == "train":
        from .experiments import train_models

        train_models(args.data, args.out_dir, args.date)
        return
    frame, protocol = prepare_batch(args.data, args.history)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "predict":
        import joblib

        from .features import HORIZON
        predictions = joblib.load(args.model).predict(frame)
        frame = frame[["model_id", "feature_date", "current_price_rub"]].copy()
        frame["target_date"] = frame["feature_date"] + HORIZON
        frame["forecast_price_rub"] = predictions
    frame.to_csv(args.out, index=False, date_format="%Y-%m-%d", float_format="%.12g", lineterminator="\n")
    print(json.dumps(protocol, ensure_ascii=False))


if __name__ == "__main__":
    main()
