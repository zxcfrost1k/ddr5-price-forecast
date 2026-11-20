"""Parse saved PCstonks responses and assess the ten-model pilot without imputation.

Standard-library only. The raw response and source order remain intact.
Repeated date/price/stock tuples are flagged, not deleted from the raw export.
"""
import csv
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ram_pilot"
RAW = DATA / "raw"
OUTPUT = ROOT / "outputs" / "01a1eab1"
AS_OF = date(2026, 11, 17)
EVENT_RE = re.compile(r"^(\d{2}\.\d{2}\.\d{4})\s*-\s*(Цена не указана|[\d\s]+\s*₽)\s*-\s*(В наличии|Нет в наличии)$")


class Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        text = " ".join(data.split())
        if text:
            self.parts.append(text)


def plain(s):
    parser = Text()
    parser.feed(s)
    return " ".join(parser.parts)


def iso(s):
    return datetime.strptime(s, "%d.%m.%Y").date().isoformat()


def write_csv(name, rows):
    if not rows:
        return
    with (DATA / name).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse(source):
    product_id = source["id"]
    payload = (RAW / f"{product_id}.json").read_bytes()
    meta = json.loads((RAW / f"{product_id}.json.meta.json").read_text())
    assert hashlib.sha256(payload).hexdigest() == meta["sha256"]
    item = json.loads(payload)
    fragment = item["html"]
    p = Text()
    p.feed(fragment)

    def following(label):
        return p.parts[p.parts.index(label) + 1]

    title = item["name"]
    specs = following("Характеристики")
    mpn = specs.split(",")[0].strip().strip("[]")
    ddr = re.search(r"DDR[345]", title + " " + specs).group()
    assert ddr == source["expected_ddr"], (product_id, title, specs)
    module = re.search(r"(\d+)\s*ГБ\s*[xх×]\s*(\d+)\s*шт", specs)
    assert module, (product_id, specs)
    size, count = map(int, module.groups())
    total = int(re.search(r"(\d+)\s*ГБ\s*$", title)[1])
    assert size * count == total
    speed = int(re.search(r"(\d+)\s*МГц", specs)[1])
    cl_match = re.search(r"CL\s*(\d+)|(\d+)\s*\(CL\)", specs)
    cl = int(next(g for g in cl_match.groups() if g)) if cl_match else None
    brands = ["Corsair", "Patriot", "ADATA", "Kingston", "G.Skill", "Apacer"]
    brand = next(b for b in brands if b in title)
    page_link = re.search(r'href="(/catalog/\d+)"', fragment)
    page_url = "https://pcstonks.com" + page_link[1] if page_link else None
    current_price = int(re.sub(r"\D", "", following("Цена")))
    product = {
        "product_id": product_id, "mpn": mpn, "name": title,
        "brand": brand, "ddr": ddr, "total_capacity_gb": total,
        "module_capacity_gb": size, "module_count": count,
        "speed_mt_s": speed, "speed_source_unit": "МГц (обозначение магазина)",
        "cl": cl, "form_factor": "UDIMM" if "UDIMM" in title else None,
        "source_added_date": iso(following("Товар добавлен")),
        "displayed_current_price_rub": current_price,
        "retailer": "DNS", "region": None, "currency": "RUB",
        "specifications_raw": specs, "page_url": page_url,
        "data_url": meta["requested_url"], "retrieved_at_utc": meta["retrieved_at_utc"],
        "raw_sha256": meta["sha256"],
    }
    events, seen = [], set()
    paragraphs = re.findall(r"<p\b[^>]*>(.*?)</p>", fragment, re.S)
    candidate_count = 0
    for paragraph in paragraphs:
        text = plain(paragraph)
        if not re.match(r"\d{2}\.\d{2}\.\d{4}", text):
            continue
        candidate_count += 1
        match = EVENT_RE.fullmatch(text)
        assert match, (product_id, text)
        day, price_string, stock = match.groups()
        price = None if price_string == "Цена не указана" else int(re.sub(r"\D", "", price_string))
        day = iso(day)
        assert date.fromisoformat(day) <= AS_OF
        assert price is None or price > 0
        state = stock == "В наличии"
        key = (day, price, state)
        repeat = key in seen
        seen.add(key)
        events.append({
            "product_id": product_id, "mpn": mpn,
            "source_order_desc": len(events) + 1,
            "date": day, "price_rub": price, "in_stock": state,
            "same_tuple_repeat": repeat,
            "original_text": text, "data_url": product["data_url"],
            "retrieved_at_utc": product["retrieved_at_utc"],
        })
    assert candidate_count == len(events) and events
    assert all(events[i]["date"] >= events[i+1]["date"] for i in range(len(events)-1))
    assert events[0]["price_rub"] == current_price
    return product, events


def assess(product, events):
    unique = [e for e in events if not e["same_tuple_repeat"]]
    by_date = defaultdict(list)
    for event in unique:
        by_date[event["date"]].append(event)
    daily = []
    for day, records in sorted(by_date.items()):
        all_prices = sorted({e["price_rub"] for e in records if e["price_rub"] is not None})
        available_prices = sorted({e["price_rub"] for e in records if e["in_stock"] and e["price_rub"] is not None})
        states = {e["in_stock"] for e in records}
        # Strict point: one price, all known states in stock, no missing price.
        strict = len(all_prices) == 1 and states == {True} and all(e["price_rub"] is not None for e in records)
        daily.append({
            "product_id": product["product_id"], "mpn": product["mpn"], "date": day,
            "distinct_records": len(records), "price_variants": len(all_prices),
            "in_stock_price_variants": len(available_prices),
            "mixed_availability": len(states) > 1,
            "has_missing_price": any(e["price_rub"] is None for e in records),
            "observed_in_stock_min_rub": min(available_prices) if available_prices else None,
            "observed_in_stock_median_rub": median(available_prices) if available_prices else None,
            "observed_in_stock_max_rub": max(available_prices) if available_prices else None,
            "strict_observed_price_rub": all_prices[0] if strict else None,
            "is_strict_point": strict,
        })
    dates = [date.fromisoformat(d) for d in sorted(by_date)]
    span = (dates[-1] - dates[0]).days + 1
    gaps = [(b-a).days for a, b in zip(dates, dates[1:])]
    strict_days = [d for d in daily if d["is_strict_point"]]
    strict_dates = [date.fromisoformat(d["date"]) for d in strict_days]
    strict_gaps = [(b-a).days for a, b in zip(strict_dates, strict_dates[1:])]
    in_stock = [e for e in unique if e["in_stock"] and e["price_rub"] is not None]
    monthly = Counter(e["date"][:7] for e in strict_days)
    out = {
        "product_id": product["product_id"], "mpn": product["mpn"],
        "name": product["name"], "brand": product["brand"], "ddr": product["ddr"],
        "total_capacity_gb": product["total_capacity_gb"],
        "module_count": product["module_count"], "speed_mt_s": product["speed_mt_s"],
        "first_event_date": dates[0].isoformat(), "last_event_date": dates[-1].isoformat(),
        "span_days": span, "raw_records": len(events), "unique_tuples": len(unique),
        "repeated_tuples": len(events)-len(unique), "repeat_fraction": (len(events)-len(unique))/len(events),
        "distinct_event_days": len(dates), "event_day_fraction": len(dates)/span,
        "days_without_event": span-len(dates),
        "median_event_gap_days": median(gaps) if gaps else None,
        "max_event_gap_days": max(gaps) if gaps else None,
        "gaps_over_30_days": sum(g > 30 for g in gaps),
        "in_stock_unique_records": len(in_stock),
        "out_of_stock_unique_records": sum(not e["in_stock"] for e in unique),
        "out_of_stock_fraction": sum(not e["in_stock"] for e in unique)/len(unique),
        "missing_price_unique_records": sum(e["price_rub"] is None for e in unique),
        "distinct_in_stock_prices": len({e["price_rub"] for e in in_stock}),
        "dates_with_multiple_prices": sum(d["price_variants"] > 1 for d in daily),
        "dates_with_mixed_availability": sum(d["mixed_availability"] for d in daily),
        "strict_observed_days": len(strict_days),
        "strict_price_changes_between_observed_days": sum(a["strict_observed_price_rub"] != b["strict_observed_price_rub"] for a,b in zip(strict_days,strict_days[1:])),
        "max_strict_gap_days": max(strict_gaps) if strict_gaps else None,
        "strict_points_last_365_days": sum(0 <= (AS_OF-date.fromisoformat(d["date"])).days < 365 for d in strict_days),
        "strict_points_last_90_days": sum(0 <= (AS_OF-date.fromisoformat(d["date"])).days < 90 for d in strict_days),
        "months_with_strict_points": len(monthly),
        "days_since_last_event": (AS_OF-dates[-1]).days,
        "min_in_stock_price_rub": min(e["price_rub"] for e in in_stock) if in_stock else None,
        "max_in_stock_price_rub": max(e["price_rub"] for e in in_stock) if in_stock else None,
        "data_url": product["data_url"], "page_url": product["page_url"],
    }
    return out, daily, unique


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    products, all_events, summaries, all_daily, all_unique = [], [], [], [], []
    for source in json.loads((DATA / "sources.json").read_text()):
        product, events = parse(source)
        summary, daily, unique = assess(product, events)
        products.append(product)
        all_events.extend(events)
        summaries.append(summary)
        all_daily.extend(daily)
        all_unique.extend(unique)
    assert len(products) == len({p["mpn"] for p in products}) == 10
    total = {
        "as_of_date": AS_OF.isoformat(), "models": len(products),
        "ddr_counts": dict(Counter(p["ddr"] for p in products)),
        "brands": len({p["brand"] for p in products}),
        "first_date": min(s["first_event_date"] for s in summaries),
        "last_date": max(s["last_event_date"] for s in summaries),
        "raw_records": len(all_events), "unique_tuples": len(all_unique),
        "repeated_tuples": len(all_events)-len(all_unique),
        "out_of_stock_unique_records": sum(not e["in_stock"] for e in all_unique),
        "missing_price_unique_records": sum(e["price_rub"] is None for e in all_unique),
        "model_date_pairs": len(all_daily),
        "strict_observed_days": sum(d["is_strict_point"] for d in all_daily),
        "dates_with_multiple_prices": sum(d["price_variants"] > 1 for d in all_daily),
        "dates_with_mixed_availability": sum(d["mixed_availability"] for d in all_daily),
        "date_parse_failures": 0, "nonpositive_prices": 0, "future_dates": 0,
        "metadata_required_complete": all(all(p[k] is not None for k in ["mpn","brand","ddr","total_capacity_gb","module_count","speed_mt_s","cl"]) for p in products),
        "region_known_models": sum(p["region"] is not None for p in products),
        "form_factor_explicit_models": sum(p["form_factor"] is not None for p in products),
    }
    write_csv("products.csv", products)
    write_csv("history_raw.csv", all_events)
    write_csv("history_distinct_tuples.csv", all_unique)
    write_csv("daily_observed.csv", all_daily)
    write_csv("quality_by_model.csv", summaries)
    result = {"summary":total,"products":products,"quality":summaries,"raw_events":all_events,"daily":all_daily}
    (DATA / "analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(total,ensure_ascii=False,indent=2))
    for s in summaries:
        fields = ["product_id","mpn","raw_records","repeated_tuples","distinct_event_days","strict_observed_days","out_of_stock_unique_records","max_event_gap_days","max_strict_gap_days","days_since_last_event"]
        print(json.dumps({k:s[k] for k in fields},ensure_ascii=False))


if __name__ == "__main__":
    main()
