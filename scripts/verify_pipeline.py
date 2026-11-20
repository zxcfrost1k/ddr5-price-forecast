"""Verify repeatability and point-in-time parity for all supplied CSV batches."""

from hashlib import sha256
from pathlib import Path
from subprocess import check_output
from tempfile import TemporaryDirectory
import json
import sys

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INITIAL = DATA / "initial/initial.csv"
BATCHES = sorted((DATA / "incoming").glob("batch_*.csv"))
RAW = DATA / "raw/observations.csv"


def run(path, histories, output, mode="prepare"):
    cmd = [sys.executable, "-m", "ram_pipeline", mode, "--data", str(path), "--out", str(output)]
    for history in histories:
        cmd.extend(["--history", str(history)])
    return json.loads(check_output(cmd, cwd=ROOT, text=True))


def report(tmp):
    reference = tmp / "reference.csv"
    run(RAW, [], reference)
    expected = pd.read_csv(reference).set_index(["model_id", "feature_date"])
    result = []
    histories = []
    for path in [INITIAL, *BATCHES]:
        a, b = tmp / "a.csv", tmp / "b.csv"
        protocol_a = run(path, histories, a)
        protocol_b = run(path, histories, b)
        assert protocol_a == protocol_b
        contents = a.read_bytes()
        assert contents == b.read_bytes()
        prepared = pd.read_csv(a).set_index(["model_id", "feature_date"])
        assert_frame_equal(prepared, expected.loc[prepared.index], check_dtype=False)
        result.append({
            "source": str(path.relative_to(ROOT)), "rows": len(prepared),
            "matrix_sha256": sha256(contents).hexdigest(), "matches_full_archive_prefix": True,
            "quality": protocol_a,
        })
        histories.append(path)

    a, b = tmp / "forecast_a.csv", tmp / "forecast_b.csv"
    run(BATCHES[0], [INITIAL], a, mode="predict")
    run(BATCHES[0], [INITIAL], b, mode="predict")
    assert a.read_bytes() == b.read_bytes()
    forecast = pd.read_csv(a)
    assert np.isfinite(forecast["forecast_price_rub"]).all() and (forecast["forecast_price_rub"] > 0).all()
    assert (pd.to_datetime(forecast["target_date"]) - pd.to_datetime(forecast["feature_date"])).eq(
        pd.Timedelta(days=7)
    ).all()
    return {
        "matrix_checks": result, "repeat_predictions": len(forecast),
        "predictions_sha256": sha256(a.read_bytes()).hexdigest(), "repeat_prediction_match": True,
    }


if __name__ == "__main__":
    with TemporaryDirectory() as temp:
        result = report(Path(temp))
    out = ROOT / "reports/pipeline_verification.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)
