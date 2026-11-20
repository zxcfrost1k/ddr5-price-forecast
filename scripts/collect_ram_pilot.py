"""Collect ten public PCstonks card responses; preserve bytes and provenance.

The public catalogue exposes data-json="/catalog/<id>.json" for each card.
That response contains the same name and HTML history shown in its modal.

Run from any directory with Python 3. No third-party dependencies.
Existing responses are retained: delete/move the raw folder explicitly to refresh.
"""
import hashlib
import json
import time
import urllib.request
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ram_pilot"
RAW = DATA / "raw"
UA = "RAM-Coursework-Pilot/1.0 (public price-history research; 10 pages)"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=40) as response:
        payload = response.read()
        info = {
            "requested_url": url,
            "final_url": response.url,
            "http_status": response.status,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "content_type": response.headers.get("Content-Type"),
            "server_date": response.headers.get("Date"),
            "last_modified": response.headers.get("Last-Modified"),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    return payload, info


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    sources = json.loads((DATA / "sources.json").read_text())
    robots_url = "https://pcstonks.com/robots.txt"
    robots_bytes, robots_info = fetch(robots_url)
    (RAW / "robots.txt").write_bytes(robots_bytes)
    (RAW / "robots.meta.json").write_text(json.dumps(robots_info, indent=2))
    robots = urllib.robotparser.RobotFileParser(robots_url)
    robots.parse(robots_bytes.decode("utf-8").splitlines())
    for source in sources:
        destination = RAW / (source["id"] + ".json")
        if destination.exists():
            print(source["id"], "cached", flush=True)
            continue
        source_url = "https://pcstonks.com/catalog/" + source["id"] + ".json"
        if not robots.can_fetch(UA, source_url):
            raise RuntimeError("robots.txt disallows " + source_url)
        time.sleep(1.2)
        payload, info = fetch(source_url)
        parsed = json.loads(payload)
        if not isinstance(parsed.get("html"), str) or "История цен и наличия" not in parsed["html"]:
            raise RuntimeError("Missing history in response for " + source["id"])
        destination.write_bytes(payload)
        (RAW / (source["id"] + ".json.meta.json")).write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(source["id"], info["http_status"], info["bytes"], parsed["name"], flush=True)


if __name__ == "__main__":
    main()
