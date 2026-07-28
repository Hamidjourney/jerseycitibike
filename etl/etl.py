# etl/etl.py
import os, io, json
import requests, zipfile
import pandas as pd
from datetime import datetime, timezone

# ---------- Config ----------
SYSTEM_PREFIX = "JC"  # Jersey City
START_YEAR = 2025
START_MONTH = 1
S3_BASE = "https://s3.amazonaws.com/tripdata"
OUT_DIR = os.path.join("docs", "data")
MONTHLY_TOTALS_JSON = os.path.join(OUT_DIR, "monthly_totals.json")
TOP_STATIONS_JSON = os.path.join(OUT_DIR, "top_stations_latest.json")

# ---------- Helpers ----------
def month_url(year: int, month: int) -> list[str]:
    ym = f"{year}{month:02d}"
    return [
        f"{S3_BASE}/{SYSTEM_PREFIX}-{ym}-citibike-tripdata.csv.zip",
        f"{S3_BASE}/{SYSTEM_PREFIX}-{ym}-citibike-tripdata.zip",
    ]

def try_fetch_zip(url: str) -> bytes | None:
    r = requests.get(url, timeout=60, stream=True)
    print(f"    status={r.status_code}, content-type={r.headers.get('Content-Type','')}, url={url}")
    ctype = r.headers.get("Content-Type", "").lower()
    if r.status_code == 200 and ("zip" in ctype or "octet-stream" in ctype):
        return r.content
    return None

def read_trips_from_zip(zbytes: bytes, year: int, month: int) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv") and "__macosx" not in n.lower()]
        if not csv_names:
            raise ValueError("No CSV found in zip")
        with zf.open(csv_names[0]) as f:
            usecols = [
                "started_at", "ended_at",
                "start_station_id", "start_station_name",
                "end_station_id", "end_station_name",
                "member_casual",
            ]
            df = pd.read_csv(f, low_memory=False, dtype=str, usecols=lambda c: c in usecols)
    df.columns = [c.strip().lower() for c in df.columns]
    if "started_at" in df.columns:
        df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce", utc=True)
    if "ended_at" in df.columns:
        df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce", utc=True)
    if "member_casual" in df.columns:
        df["member_casual"] = df["member_casual"].str.lower()
    elif "usertype" in df.columns:
        df["member_casual"] = df["usertype"].str.lower().map({"subscriber": "member", "customer": "casual"})
    else:
        df["member_casual"] = "unknown"
    # Safety filter: keep only rows actually in the target month
    if "started_at" in df.columns:
        df = df[(df["started_at"].dt.year == year) & (df["started_at"].dt.month == month)]
    return df

def safe_name(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return "Unknown"
    s = str(s)
    return "Unknown" if s.lower() == "nan" else s

def top5(grouped: pd.DataFrame) -> list[dict]:
    out = []
    for _, row in grouped.nlargest(5, "trips").iterrows():
        out.append({
            "station_id": safe_name(row.get("station_id")),
            "station_name": safe_name(row.get("station_name")),
            "trips": int(row.get("trips", 0)),
        })
    return out

def net_flow(starts: pd.DataFrame, ends: pd.DataFrame) -> list[dict]:
    # starts/ends: columns ['station_id','station_name','trips']
    merged = pd.merge(
        starts, ends, on=["station_id", "station_name"], how="outer", suffixes=("_start", "_end")
    ).fillna(0)
    merged["net"] = merged["trips_start"] - merged["trips_end"]
    merged = merged.sort_values("net", ascending=False)
    out = []
    for _, row in merged.iterrows():
        out.append({
            "station_id": safe_name(row.get("station_id")),
            "station_name": safe_name(row.get("station_name")),
            "net": int(row["net"]),
        })
    return out

def month_range_forward(start_year: int, start_month: int):
    """Yield (year, month) tuples from start up to the current UTC month, inclusive."""
    now = datetime.now(timezone.utc)
    y, m = start_year, start_month
    while (y, m) <= (now.year, now.month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1

def load_existing_totals() -> dict:
    """Return {'2025-01': rides_count_int_or_dict, ...} from prior run, if present."""
    if os.path.exists(MONTHLY_TOTALS_JSON):
        with open(MONTHLY_TOTALS_JSON, "r", encoding="utf-8") as f:
            rows = json.load(f)
        return {r["month"]: r for r in rows}
    return {}

# ---------- Main ETL ----------
def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    existing = load_existing_totals()
    months_to_check = list(month_range_forward(START_YEAR, START_MONTH))
    latest_key = f"{months_to_check[-1][0]}-{months_to_check[-1][1]:02d}"

    monthly_rows = []
    latest_df = None
    latest_month_key = None

    for (year, month) in months_to_check:
        key = f"{year}-{month:02d}"
        is_latest = (key == latest_key)

        # Skip refetching months we already have, UNLESS it's the current/latest
        # month (Citi Bike sometimes republishes the current month with corrections),
        # or we also need the raw month for top-station / net-flow computation.
        if key in existing and not is_latest:
            monthly_rows.append(existing[key])
            continue

        z = None
        for url in month_url(year, month):
            print("Checking:", url)
            z = try_fetch_zip(url)
            if z is not None:
                print(f"  -> found ({len(z)} bytes)")
                break
        if z is None:
            print("  -> not found (not yet published)")
            continue

        try:
            df = read_trips_from_zip(z, year, month)
        except Exception as e:
            print("  -> error reading zip:", e)
            continue

        member_trips = int((df["member_casual"] == "member").sum())
        casual_trips = int((df["member_casual"] == "casual").sum())

        monthly_rows.append({
            "month": key,
            "rides": int(len(df)),
            "member": member_trips,
            "casual": casual_trips,
        })

        if is_latest:
            latest_df = df
            latest_month_key = key

    # Preserve chronological order, de-dup by month key (latest write wins)
    dedup = {r["month"]: r for r in monthly_rows}
    monthly_rows = [dedup[k] for k in sorted(dedup.keys())]

    with open(MONTHLY_TOTALS_JSON, "w", encoding="utf-8") as f:
        json.dump(monthly_rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {MONTHLY_TOTALS_JSON} with {len(monthly_rows)} rows")

    if latest_df is None:
        print("No new latest-month data fetched this run; leaving top_stations file untouched.")
        return

    df_latest = latest_df.copy()
    df_latest["member_casual"] = df_latest["member_casual"].fillna("unknown").str.lower()

    def grouped_counts(df, col_id, col_name):
        return (
            df.groupby([col_id, col_name], dropna=False)
            .size().reset_index(name="trips")
            .rename(columns={col_id: "station_id", col_name: "station_name"})
        )

    result = {"latest_month": latest_month_key, "top5": {}, "net_flow": {}}

    for rider_type in ("casual", "member"):
        sub = df_latest[df_latest["member_casual"] == rider_type]
        starts = grouped_counts(sub, "start_station_id", "start_station_name")
        ends = grouped_counts(sub, "end_station_id", "end_station_name")

        result["top5"].setdefault("starts", {})[rider_type] = top5(starts)
        result["top5"].setdefault("ends", {})[rider_type] = top5(ends)
        result["net_flow"][rider_type] = net_flow(starts, ends)[:10]  # top 10 by net magnitude

    with open(TOP_STATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote {TOP_STATIONS_JSON}")

if __name__ == "__main__":
    run()
