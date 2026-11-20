"""Chronological replay batches and EDA of the initial DDR5 observations."""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import correlation, mean, median, pstdev, quantiles

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw" / "observations.csv"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(path, rows):
    return {
        "file": str(path.relative_to(ROOT)),
        "rows": len(rows),
        "first_date": min(row["date"] for row in rows),
        "last_date": max(row["date"] for row in rows),
        "sha256": digest(path),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ranks(values):
    indices = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(indices):
        end = index + 1
        while end < len(indices) and values[indices[end]] == values[indices[index]]:
            end += 1
        for pos in indices[index:end]:
            result[pos] = ((index + 1) + end) / 2
        index = end
    return result


def spearman(xs, ys):
    return correlation(ranks(xs), ranks(ys))


def eda(rows):
    prices = [int(row["price_rub"]) for row in rows]
    by_model = defaultdict(list)
    cohorts = defaultdict(lambda: defaultdict(list))
    missing = {field: sum(not row[field] for row in rows) for field in rows[0]}
    impossible = Counter()
    for row in rows:
        by_model[row["model_id"]].append(row)
        capacity = int(row["total_capacity_gb"])
        modules = int(row["module_count"])
        module_capacity = int(row["module_capacity_gb"])
        impossible["nonpositive_price"] += int(row["price_rub"]) <= 0
        impossible["capacity_mismatch"] += capacity != modules * module_capacity
        impossible["nonpositive_spec"] += min(capacity, modules, module_capacity, int(row["speed_mt_s"])) <= 0 or (bool(row["cl"]) and int(row["cl"]) <= 0)
        impossible["incorrect_cl_flag"] += int(row["cl_missing"]) != int(not row["cl"])
        impossible["wrong_currency_or_retailer"] += row["currency"] != "RUB" or row["retailer"] != "DNS"
        if row["is_server"] == "0" and capacity in (16, 32):
            cohorts[row["brand"], row["date"][:7]][capacity].append(int(row["price_rub"]))
    comparable = [median(group[32]) / median(group[16]) for group in cohorts.values() if 16 in group and 32 in group]
    with_cl = [row for row in rows if row["cl"]]
    caps = [int(row["total_capacity_gb"]) for row in rows]
    counts = [int(row["module_count"]) for row in rows]
    rates = [int(row["speed_mt_s"]) for row in with_cl]
    cl = [int(row["cl"]) for row in with_cl]
    examples = []
    for model_id, series in by_model.items():
        future = {date.fromisoformat(row["date"]): int(row["price_rub"]) for row in series}
        for index, row in enumerate(series):
            day = date.fromisoformat(row["date"])
            target = future.get(day + timedelta(days=7))
            if target is None:
                continue
            price = int(row["price_rub"])
            previous = series[index - 1] if index else None
            gap = (day - date.fromisoformat(previous["date"])).days if previous else None
            window = [int(old["price_rub"]) for old in series[:index+1] if (day-date.fromisoformat(old["date"])).days <= 30]
            dynamic_std = pstdev([math.log(value) for value in window]) if len(window) >= 3 else None
            examples.append({"model_id": model_id, "feature_date": day.isoformat(),
                "target_date": (day+timedelta(days=7)).isoformat(), "current_price_rub": price,
                "target_price_rub": target, "capacity_gb": int(row["total_capacity_gb"]),
                "relative_error_naive": abs(target-price)/price, "gap_days": gap,
                "log_price_sd_30d": dynamic_std})
    targets = [row["target_price_rub"] for row in examples]
    q1, _, q3 = quantiles(prices, n=4, method="inclusive")
    upper = q3 + 1.5*(q3-q1)
    lower = q1 - 1.5*(q3-q1)
    target_quantiles = quantiles(targets, n=4, method="inclusive")
    target_bins = [0, 10000, 20000, 30000, 40000, 60000, 80000, 100000, 160000]
    histogram = [sum(low <= price < high for price in targets) for low, high in zip(target_bins,target_bins[1:])]
    assert sum(histogram)==len(targets)
    gap_groups = {
        "up_to_14": [row["relative_error_naive"] for row in examples if row["gap_days"] is not None and row["gap_days"] <= 14],
        "over_14": [row["relative_error_naive"] for row in examples if row["gap_days"] is not None and row["gap_days"] > 14],
    }
    volatile = [row for row in examples if row["log_price_sd_30d"] is not None]
    result = {
        "rows": len(rows), "models": len(by_model), "missing": missing,
        "exact_duplicate_rows": len(rows)-len({tuple(row.items()) for row in rows}),
        "duplicate_model_dates": len(rows)-len({(row["model_id"],row["date"]) for row in rows}),
        "impossible": dict(impossible),
        "prices": {"min":min(prices),"q1":q1,"median":median(prices),"q3":q3,"max":max(prices),
                   "iqr_lower":lower,"iqr_upper":upper,"iqr_outliers":sum(p<lower or p>upper for p in prices)},
        "capacity_counts": dict(sorted(Counter(caps).items())),
        "speed_counts": dict(sorted(Counter(int(r["speed_mt_s"]) for r in rows).items())),
        "top_brands": dict(Counter(row["brand"] for row in rows).most_common(5)),
        "models_with_single_observation":sum(len(series)==1 for series in by_model.values()),
        "observations_per_model_median":median(map(len,by_model.values())),
        "feature_correlations":{"capacity_vs_count":spearman(caps,counts),"speed_vs_cl":spearman(rates,cl), "speed_vs_cl_n":len(with_cl)},
        "target":{"rows":len(examples),"models":len({r["model_id"] for r in examples}),
                  "min":min(targets),"max":max(targets),"q1":target_quantiles[0],"median":target_quantiles[1],"q3":target_quantiles[2],
                  "edges":target_bins,"counts":histogram,
                  "up":sum(r["target_price_rub"]>r["current_price_rub"] for r in examples),
                  "down":sum(r["target_price_rub"]<r["current_price_rub"] for r in examples),
                  "same":sum(r["target_price_rub"]==r["current_price_rub"] for r in examples),
                  "current_spearman":spearman([r["current_price_rub"] for r in examples],targets),
                  "current_naive_mae":mean(abs(r["target_price_rub"]-r["current_price_rub"]) for r in examples),
                  "by_capacity":{str(cap):{"n":len(values),"median":median(values)} for cap in sorted(set(r["capacity_gb"] for r in examples)) if (values:=[r["target_price_rub"] for r in examples if r["capacity_gb"]==cap])}},
        "capacity_hypothesis":{"comparable_brand_months":len(comparable),"over_1":sum(value>1 for value in comparable),"median_price_ratio":median(comparable)},
        "gap_hypothesis":{key:{"n":len(values),"median_relative_error":median(values)} for key,values in gap_groups.items()},
        "volatility_hypothesis":{"n":len(volatile),"spearman":spearman([r["log_price_sd_30d"] for r in volatile],[r["relative_error_naive"] for r in volatile])},
    }
    return result, examples


def main():
    before = digest(RAW)
    with RAW.open(encoding="utf-8-sig",newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row:(row["date"],row["model_id"]))
    assert all(date.fromisoformat(row["date"]).isoformat()==row["date"] for row in rows)
    ends = [i for i in range(1,len(rows)) if rows[i-1]["date"]!=rows[i]["date"]]
    cuts = [min(ends,key=lambda end:(abs(end-len(rows)*fraction),end)) for fraction in (.6,.7,.8,.9)]
    assert len(set(cuts))==4
    splits = [rows[start:end] for start,end in zip([0]+cuts,cuts+[len(rows)])]
    files = [DATA/"initial"/"initial.csv"]+[DATA/"incoming"/f"batch_{i:02}.csv" for i in range(1,5)]
    for path,part in zip(files,splits):
        write_csv(path,part)
    assert [row for part in splits for row in part]==rows
    assert all(a[-1]["date"]<b[0]["date"] for a,b in zip(splits,splits[1:]))
    assert digest(RAW)==before
    manifest = {"source":"PCstonks / DNS, подготовленный срез DDR5 на 17.11.2026",
                "source_url":"https://pcstonks.com/catalog/", "raw":describe(RAW,rows),
                "partitions":[describe(path,part) for path,part in zip(files,splits)],
                "split_rule":"Nearest full day boundaries to 60%, 70%, 80%, 90% cumulative row counts"}
    write_json(DATA/"manifest.json",manifest)
    result,examples=eda(splits[0])
    report=ROOT/"reports"
    report.mkdir(exist_ok=True)
    write_json(report/"eda_initial.json", result)
    write_csv(report/"eda_target_7d.csv", examples)
    print(json.dumps({"partitions":manifest["partitions"],"EDA":result},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
