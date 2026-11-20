"""Compare forecasts with calendar-based expanding-window validation."""

from datetime import date
from hashlib import sha256
from importlib.metadata import version
from subprocess import check_output, CalledProcessError, DEVNULL
from time import perf_counter
import json
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from .features import ROOT, TARGET, build_features, file_hash, label_rows, read_observations
from .model import make_pipeline

SEED = 42
SCALED = {"lasso", "ridge", "knn"}
ALPHAS_L1 = [0.0001, 0.001, 0.01, 0.05, 0.10, 0.30]
ALPHAS_L2 = [0.01, 0.1, 1, 10, 100, 1000]
DEPTHS = [1, 2, 3, 4, 6, 8, None]
DEFAULTS = {
    "lasso": {"alpha": 0.01},
    "ridge": {"alpha": 10},
    "knn": {"n_neighbors": 9, "weights": "distance"},
    "tree": {"max_depth": 3, "min_samples_leaf": 5},
    "forest": {"n_estimators": 160, "max_depth": 5, "min_samples_leaf": 4},
    "boost": {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05, "min_samples_leaf": 5},
}
GRIDS = {
    "lasso": {"alpha": ALPHAS_L1},
    "ridge": {"alpha": ALPHAS_L2},
    "knn": {"n_neighbors": [3, 5, 9, 15, 25], "weights": ["uniform", "distance"]},
    "tree": {"max_depth": [1, 2, 3, 4], "min_samples_leaf": [4, 8, 12]},
    "forest": {"n_estimators": [100, 200], "max_depth": [2, 4, 6], "min_samples_leaf": [4, 8]},
    "boost": {"n_estimators": [75, 150], "max_depth": [1, 2, 3], "learning_rate": [0.03, 0.08],
              "min_samples_leaf": [5]},
}
SIMPLICITY = ["lasso", "ridge", "tree", "knn", "forest", "boost"]


def estimator(key, params):
    if key == "baseline":
        return None
    classes = {"ridge": Ridge, "lasso": Lasso, "knn": KNeighborsRegressor, "tree": DecisionTreeRegressor,
               "forest": RandomForestRegressor, "boost": GradientBoostingRegressor}
    extra = {"random_state": SEED} if key in {"tree", "forest", "boost"} else {}
    if key == "lasso":
        extra["max_iter"] = 30000
    return classes[key](**params, **extra)


def boundary(rows, fraction):
    counts = rows["feature_date"].value_counts().sort_index().cumsum()
    options = counts.iloc[:-1]
    pos = int((options - fraction * len(rows)).abs().to_numpy().argmin())
    return counts.index[pos + 1]


def date_range(frame, column="feature_date"):
    return [frame[column].min().date().isoformat(), frame[column].max().date().isoformat()]


def training_split(observations):
    labelled = label_rows(observations)
    cutoff = boundary(labelled, .80)
    early = observations.loc[observations["date"] < cutoff]
    samples = label_rows(early, build_features(early))
    held = labelled.loc[labelled["feature_date"] >= cutoff]
    split = {
        "labelled_initial": len(labelled), "training": len(samples), "holdout": len(held),
        "holdout_start": cutoff.date().isoformat(), "training_dates": date_range(samples),
        "training_target_dates": date_range(samples, "target_date"), "holdout_dates": date_range(held),
        "purged": len(labelled) - len(samples) - len(held), "holdout_scores_computed": False,
    }
    starts = [boundary(samples, i / 6) for i in range(1, 6)]
    if len(set(starts)) != 5:
        raise ValueError("Недостаточно различных дат для пяти временных блоков.")
    folds, fold_info = [], []
    for i, start in enumerate(starts):
        earlier = samples["feature_date"].lt(start)
        train = samples.loc[earlier & samples["target_date"].lt(start)]
        later = samples["feature_date"].ge(start)
        if i + 1 < len(starts):
            later &= samples["feature_date"].lt(starts[i + 1])
        valid = samples.loc[later]
        if len(train) < 25 or len(valid) < 10:
            raise ValueError("Недостаточно размеченных строк для временной проверки.")
        assert train["target_date"].max() < valid["feature_date"].min()
        folds.append((train, valid))
        fold_info.append({
            "fold": i + 1, "validation_start": start.date().isoformat(), "train": len(train),
            "validation": len(valid), "validation_dates": date_range(valid),
            "purged_train": int(earlier.sum()) - len(train),
        })
    return samples, folds, {**split, "folds": fold_info}


def metric_values(truth, prediction):
    return {
        "mae": mean_absolute_error(truth, prediction),
        "rmse": root_mean_squared_error(truth, prediction),
        "mape": 100 * mean_absolute_percentage_error(truth, prediction),
    }


def json_file(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    pd.DataFrame(list(rows)).to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def git_version():
    try:
        return check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=DEVNULL).strip()
    except (OSError, CalledProcessError):
        return None


def train_models(data, output, run_date):
    if date.fromisoformat(run_date).isoformat() != run_date:
        raise ValueError("Дата запуска должна быть записана в формате ГГГГ-ММ-ДД.")
    report = output / "reports"
    models = output / "models"
    report.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    obs, protocol = read_observations(data)
    samples, folds, split = training_split(obs)
    json_file(report / "training_split.json", {**split, "source_quality": protocol})

    records, summaries, configs = [], {}, {}

    def evaluate(name, algorithm, params, group, extended=True):
        fold_rows = []
        for index, (train, valid) in enumerate(folds, 1):
            model = make_pipeline(estimator(algorithm, params), extended=extended, scale=algorithm in SCALED)
            started = perf_counter()
            model.fit(train, train[TARGET].to_numpy())
            seconds = perf_counter() - started
            weights = model.named_steps["price"].estimator_.coef_ if algorithm in ("lasso", "ridge") else []
            fold_rows.append({
                "experiment": name, "algorithm": algorithm, "group": group, "run_date": run_date,
                "parameters": json.dumps(params, sort_keys=True), "seed": SEED,
                "new_features": extended and algorithm != "baseline", "fold": index,
                **metric_values(valid[TARGET], model.predict(valid)),
                "train_mae": mean_absolute_error(train[TARGET], model.predict(train)),
                "fit_seconds": seconds,
                "nonzero_coefficients": np.count_nonzero(np.abs(weights) > 1e-8) if len(weights) else np.nan,
            })
        records.extend(fold_rows)
        summary = {"experiment": name, "algorithm": algorithm, "group": group, "parameters": json.dumps(params, sort_keys=True),
                   "new_features": extended and algorithm != "baseline"}
        for field in ("mae", "rmse", "mape", "train_mae", "fit_seconds"):
            summary[field] = float(np.mean([r[field] for r in fold_rows]))
        summary["mae_std"] = float(np.std([r["mae"] for r in fold_rows], ddof=1))
        summary["nonzero_coefficients"] = (
            float(np.mean([r["nonzero_coefficients"] for r in fold_rows])) if algorithm in ("lasso", "ridge") else np.nan
        )
        summaries[name] = summary
        configs[name] = (algorithm, params, extended)
        return name

    with threadpool_limits(limits=1):
        baseline = evaluate("baseline", "baseline", {}, "baseline")
        defaults = [evaluate(f"default_{key}", key, params, "default") for key, params in DEFAULTS.items()]
        for key, alphas in (("lasso", ALPHAS_L1), ("ridge", ALPHAS_L2)):
            for alpha in alphas:
                evaluate(f"regularization_{key}_{alpha}", key, {"alpha": alpha}, "regularization")
        for depth in DEPTHS:
            evaluate(f"depth_{depth}", "tree", {"max_depth": depth, "min_samples_leaf": 1}, "depth")
        for key in ("ridge", "boost"):
            evaluate(f"base_features_{key}", key, DEFAULTS[key], "ablation", extended=False)
        top = sorted(defaults, key=lambda item: summaries[item]["mae"])[:2]
        tuning = []
        for candidate in top:
            key = summaries[candidate]["algorithm"]
            experiments = [evaluate(f"grid_{key}_{i:02}", key, params, "grid")
                           for i, params in enumerate(ParameterGrid(GRIDS[key]), 1)]
            best = min(experiments, key=lambda item: (summaries[item]["mae"], item))
            tuning.append({"algorithm": key, "grid": GRIDS[key], "combinations": len(experiments),
                           "before": candidate, "best": best})
        finalists = defaults + [x["best"] for x in tuning]
        lowest = min(summaries[c]["mae"] for c in finalists)
        eligible = [c for c in finalists if summaries[c]["mae"] <= lowest * 1.01]
        selected = min(eligible, key=lambda c: (SIMPLICITY.index(summaries[c]["algorithm"]), summaries[c]["mae"]))
        algorithm, params, extended = configs[selected]
        winner_without_new = evaluate("base_features_selected", algorithm, params, "ablation", extended=False)
        final_model = make_pipeline(estimator(algorithm, params), extended=extended, scale=algorithm in SCALED)
        final_model.fit(samples, samples[TARGET].to_numpy())
        path = models / "model_v1.joblib"
        joblib.dump(final_model, path, compress=3)
        reloaded = joblib.load(path)
        assert np.array_equal(final_model.predict(samples.iloc[:10]), reloaded.predict(samples.iloc[:10]))

        for key in ("ridge", "lasso"):
            fit = min((s for s in summaries.values() if s["group"] == "regularization" and s["algorithm"] == key),
                      key=lambda s: s["mae"])
            _, config, extended_linear = configs[fit["experiment"]]
            linear = make_pipeline(estimator(key, config), extended=extended_linear, scale=True)
            linear.fit(samples, samples[TARGET].to_numpy())
            names = linear.named_steps["columns"].get_feature_names_out()[:-1]
            coefs = linear.named_steps["price"].estimator_.coef_
            write_csv(report / f"coefficients_{key}.csv", [
                {"feature": name, "coefficient": coefficient, "alpha": config["alpha"]}
                for name, coefficient in sorted(zip(names, coefs), key=lambda x: -abs(x[1]))
            ])

    write_csv(report / "experiments.csv", records)
    write_csv(report / "experiment_summary.csv", summaries.values())
    json_file(report / "model_selection.json", {
        "default_models": defaults, "tuning": tuning, "finalists": finalists,
        "selected": selected, "baseline": baseline, "selected_without_new_features": winner_without_new,
        "relative_improvement": 1 - summaries[selected]["mae"] / summaries[baseline]["mae"],
        "lowest_mae": lowest, "tie_candidates": eligible, "fit_rows": len(samples),
        "evaluation_rows_per_candidate": sum(len(valid) for _, valid in folds),
    })
    source_hash = sha256(b"".join((ROOT / "ram_pipeline" / f).read_bytes() for f in
                                 ("features.py", "model.py", "experiments.py", "__main__.py"))).hexdigest()
    json_file(models / "model_v1.json", {
        "version": "v1", "trained_on": run_date, "target": TARGET, "horizon_days": 7,
        "strategy": "current_price * exp(predicted_log_change)", "algorithm": algorithm,
        "parameters": params, "new_features": extended, "random_state": SEED,
        "snapshot_sha256": protocol["sha256"], "training_rows": len(samples), "split": split,
        "selection": {k: v for k, v in summaries[selected].items() if k != "nonzero_coefficients"},
        "release_status": "candidate_pending_holdout", "artifact_sha256": file_hash(path),
        "training_source_sha256": source_hash, "git_commit_at_run": git_version(),
        "python": platform.python_version(), "platform": f"{platform.system()} / {platform.machine()}",
        "dependencies": {p: version(p) for p in ("scikit-learn", "numpy", "pandas", "scipy", "joblib", "threadpoolctl")},
    })
    print(json.dumps({"selected": selected, "training": split, "tuning": tuning,
                      "mae": summaries[selected]["mae"], "baseline_mae": summaries[baseline]["mae"]},
                     ensure_ascii=False, indent=2))
