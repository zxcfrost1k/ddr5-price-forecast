"""Reproduce the full-snapshot description and seven-day persistence baseline."""

import csv
import json
from collections import Counter
from datetime import date, timedelta
from statistics import mean, median

from prepare_data import RAW


def main():
    with RAW.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    prices = {(r["model_id"], date.fromisoformat(r["date"])): int(r["price_rub"]) for r in rows}
    model_counts = Counter(model for model, _ in prices)
    validation_start = date(2026, 5, 25)
    test_start = date(2026, 7, 25)

    def split(day):
        if day < validation_start:
            return "train"
        if day < test_start:
            return "validation"
        return "test"

    sizes, errors, labelled_models = Counter(), [], set()
    purged = 0
    for (model, day), price in prices.items():
        target_day = day + timedelta(days=7)
        target = prices.get((model, target_day))
        if target is None:
            continue
        part = split(day)
        if part != split(target_day):
            purged += 1
            continue
        sizes[part] += 1
        labelled_models.add(model)
        if part == "test":
            errors.append(abs(target - price))
    result = {
        "rows": len(rows),
        "models": len(model_counts),
        "period": [min(r["date"] for r in rows), max(r["date"] for r in rows)],
        "price_rub": [min(prices.values()), median(prices.values()), max(prices.values())],
        "observations_per_model": [min(model_counts.values()), median(model_counts.values()), max(model_counts.values())],
        "capacity_32gb": sum(r["total_capacity_gb"] == "32" for r in rows),
        "split_counts": dict(sizes),
        "purged": purged,
        "labelled_models": len(labelled_models),
        "test_baseline_mae_rub": mean(errors),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
