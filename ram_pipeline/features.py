"""Validate price snapshots and generate point-in-time features."""

from collections import defaultdict, deque
from hashlib import sha256
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INITIAL = ROOT / "data/initial/initial.csv"
METADATA = [
    "total_capacity_gb", "module_capacity_gb", "module_count", "speed_mt_s",
    "cl", "cl_missing", "is_server",
]
REQUIRED = ["model_id", "date", "price_rub", "brand", *METADATA, "retailer", "currency"]
BASE = [
    *METADATA, "current_price_rub", "lag1_price_rub", "lag2_price_rub", "gap_days",
    "price_change_fraction", "last3_mean_rub", "last3_std_rub", "observed_count",
    "year", "month", "weekday",
]
NEW = ["price_vs_segment", "cas_latency_ns", "log_price_sd_30d", "log_change_per_day"]
TARGET = "target_price_rub"
HORIZON = pd.Timedelta(days=7)
WINDOW = pd.Timedelta(days=30)


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read_observations(path):
    path = Path(path).resolve()
    data_hash = file_hash(path)
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    for entry in [manifest["raw"], *manifest["partitions"]]:
        if path == ROOT / entry["file"] and data_hash != entry["sha256"]:
            raise ValueError(f"Контрольная сумма не совпадает: {path}")
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    missing = set(REQUIRED) - set(raw.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные поля: {sorted(missing)}")

    def number(field):
        return pd.to_numeric(frame[field], errors="coerce")

    frame = raw[REQUIRED].copy()
    for name in ("model_id", "brand"):
        frame[name] = frame[name].str.strip()
    valid_dates = frame["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    frame["date"] = pd.to_datetime(frame["date"].where(valid_dates), format="%Y-%m-%d", errors="coerce")
    valid = frame["model_id"].ne("") & frame["date"].notna()
    valid &= frame["retailer"].eq("DNS") & frame["currency"].eq("RUB")
    for column in ["price_rub", *METADATA[:4]]:
        values = number(column)
        valid &= np.isfinite(values) & values.gt(0) & values.mod(1).eq(0)
        frame[column] = values
    valid &= frame["total_capacity_gb"].eq(frame["module_count"] * frame["module_capacity_gb"])
    cl = number("cl")
    cl_missing = number("cl_missing")
    valid &= (frame["cl"].eq("") | (np.isfinite(cl) & cl.gt(0) & cl.mod(1).eq(0)))
    valid &= cl_missing.eq(frame["cl"].eq("").astype(int))
    frame["cl"] = cl
    frame["cl_missing"] = cl_missing
    frame["is_server"] = number("is_server")
    valid &= frame["is_server"].isin([0, 1])
    frame.loc[frame["brand"].eq(""), "brand"] = np.nan

    keys = ["model_id", "date"]
    invalid_keys = pd.MultiIndex.from_frame(frame.loc[~valid, keys].dropna())
    frame = frame.loc[valid]
    ambiguous = pd.MultiIndex.from_frame(frame[keys]).isin(invalid_keys)
    frame = frame.loc[~ambiguous]
    distinct = frame.drop_duplicates()
    conflict = distinct.duplicated(keys, keep=False)
    clean = distinct.loc[~conflict].sort_values(["date", "model_id"]).reset_index(drop=True)
    protocol = {
        "sha256": data_hash, "read": len(raw), "accepted": len(clean),
        "invalid": int((~valid).sum()), "linked_to_invalid": int(ambiguous.sum()),
        "duplicates": len(frame) - len(distinct), "conflicting": int(conflict.sum()),
        "price_warnings": int(clean["price_rub"].gt(50849).sum()),
    }
    if clean.empty:
        raise ValueError(f"В файле нет допустимых наблюдений: {path}")
    return clean, protocol


def build_features(observations, *, start=None):
    model_tail = defaultdict(lambda: deque(maxlen=3))
    model_windows = defaultdict(deque)
    model_visits = defaultdict(int)
    segment_latest = defaultdict(dict)
    result = []
    for day, snapshot in observations.groupby("date", sort=True):
        records = list(snapshot.itertuples(index=False))
        for row in records:
            segment = segment_latest[row.total_capacity_gb]
            segment[row.model_id] = (day, row.price_rub)
        for row in records:
            model, price = row.model_id, row.price_rub
            window_start = day - WINDOW
            window = model_windows[model]
            while window and window[0][0] < window_start:
                window.popleft()
            window.append((day, price))
            tail = model_tail[model]
            last = tail[-1] if tail else None
            penult = tail[-2] if len(tail) >= 2 else None
            gap = (day - last[0]).days if last else np.nan
            tail.append((day, price))
            model_visits[model] += 1
            if start is not None and day < start:
                continue
            peers = [
                p for other, (observed, p) in segment_latest[row.total_capacity_gb].items()
                if other != model and observed >= window_start
            ]
            trailing = [p for _, p in tail]
            values = {field: getattr(row, field) for field in METADATA}
            values.update(
                model_id=model, feature_date=day, brand=row.brand, current_price_rub=price,
                lag1_price_rub=last[1] if last else np.nan,
                lag2_price_rub=penult[1] if penult else np.nan,
                gap_days=gap,
                price_change_fraction=price / last[1] - 1 if last else np.nan,
                last3_mean_rub=float(np.mean(trailing)), last3_std_rub=float(np.std(trailing)),
                observed_count=model_visits[model], year=day.year, month=day.month, weekday=day.weekday(),
                price_vs_segment=price / np.median(peers) if len(peers) >= 3 else np.nan,
                cas_latency_ns=2000 * row.cl / row.speed_mt_s,
                log_price_sd_30d=float(np.std(np.log([p for _, p in window]))) if len(window) >= 3 else np.nan,
                log_change_per_day=math.log(price / last[1]) / gap if last else np.nan,
            )
            result.append(values)
    return pd.DataFrame(result, columns=["model_id", "feature_date", "brand", *BASE, *NEW])


def prepare_batch(input_file, histories=()):
    incoming, protocol = read_observations(input_file)
    start = incoming["date"].min()
    background = [read_observations(path)[0] for path in histories]
    if not background:
        return build_features(incoming), protocol
    past = pd.concat(background, ignore_index=True)
    past = past.loc[past["date"] < start].drop_duplicates()
    if past.duplicated(["model_id", "date"]).any():
        raise ValueError("Исторические файлы содержат конфликтующие записи на одну дату.")
    return build_features(pd.concat([past, incoming], ignore_index=True), start=start), protocol


def label_rows(rows, features=None):
    keys = ["model_id", "date"]
    facts = rows[[*keys, "price_rub"]].rename(columns={"date": "target_date", "price_rub": TARGET})
    if features is None:
        features = rows[keys].rename(columns={"date": "feature_date"})
    forecast = features.assign(target_date=features["feature_date"] + HORIZON)
    return forecast.merge(facts, on=["model_id", "target_date"], how="inner", validate="one_to_one").sort_values(
        ["feature_date", "model_id"], ignore_index=True
    )
