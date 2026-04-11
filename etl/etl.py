# etl/etl.py
import os, io, json, calendar
import requests, zipfile
import pandas as pd
from datetime import datetime

# ---------- Config ----------
SYSTEM_PREFIX = "JC"  # (Jersey City) given your filenames
YEAR = 2025
S3_BASE = "https://s3.amazonaws.com/tripdata"
OUT_DIR = os.path.join("docs", "data")
MONTHLY_TOTALS_JSON = os.path.join(OUT_DIR, "monthly_totals_2025.json")
TOP_STATIONS_JSON = os.path.join(OUT_DIR, "top_stations_latest.json")

# ---------- Helpers ----------
def month_url(year: int, month: int) -> str:
    ym = f"{year}{month:02d}"
    fname = f"{SYSTEM_PREFIX}-{ym}-citibike-tripdata.csv.zip"
    return f"{S3_BASE}/{fname}"

def try_fetch_zip(url: str) -> bytes | None:
    r = requests.get(url, timeout=60, stream=True)
    if r.status_code == 200 and r.headers.get("Content-Type", "").lower().find("zip") != -1:
        return r.content
    # Some S3 versions return octet-stream; still accept if 200
    if r.status_code == 200:
        return r.content
    return None

def read_trips_from_zip(zbytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        # find the first CSV inside
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV found in zip")
        with zf.open(csv_names[0]) as f:
            # Robust dtypes; we only need certain columns
            usecols = [
                "started_at","ended_at",
                "start_station_id","start_station_name",
                "end_station_id","end_station_name",
                "member_casual"
            ]
            # Some schemas may differ slightly; let pandas select what exists
            df = pd.read_csv(
                f, low_memory=False, dtype=str, usecols=lambda c: c in usecols
            )
    # Standardize column names (lowercase)
    df.columns = [c.strip().lower() for c in df.columns]
    # Parse dates
    if "started_at" in df.columns:
        df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce", utc=True)
    if "ended_at" in df.columns:
        df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce", utc=True)
    # Clean rider type
    if "member_casual" in df.columns:
        df["member_casual"] = df["member_casual"].str.lower()
    else:
        # fallback if schema differs (older years might use 'usertype')
        if "usertype" in df.columns:
            df["member_casual"] = df["usertype"].str.lower().map(
                {"subscriber":"member","customer":"casual"}
            )
        else:
            df["member_casual"] = "unknown"
    # Keep only 2025 rows (safety)
    if "started_at" in df.columns:
        df = df[df["started_at"].dt.year == YEAR]
    return df

def month_label(dtobj: pd.Timestamp) -> str:
    return dtobj.strftime("%Y-%m")

def safe_name(s: str | float | None) -> str:
    if pd.isna(s) or s is None:
        return "Unknown"
    return str(s)

def top5(grouped_counts: pd.DataFrame) -> list[dict]:
    # grouped_counts: columns ['station_id','station_name','trips']
    out = []
    for _, row in grouped_counts.nlargest(5, "trips").iterrows():
        out.append({
            "station_id": safe_name(row.get("station_id")),
            "station_name": safe_name(row.get("station_name")),
            "trips": int(row.get("trips", 0))
        })
    return out

# ---------- Main ETL ----------
def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Discover which months exist in 2025 (Jan..Dec, but only those actually uploaded)
    month_data = {}  # { "2025-01": DataFrame, ... }
    existing_months = []
    for m in range(1, 13):
        url = month_url(YEAR, m)
        print("Checking:", url)
        z = try_fetch_zip(url)
        if z is None:
            print(f"  -> not found")
            continue
        print(f"  -> found ({len(z)} bytes)")
        try:
            df = read_trips_from_zip(z)
        except Exception as e:
            print("  -> error reading zip:", e)
            continue
        key = f"{YEAR}-{m:02d}"
        month_data[key] = df
        existing_months.append(key)

    if not existing_months:
        raise RuntimeError("No 2025 monthly files found.")

    # 2) Monthly totals for 2025
    monthly_rows = []
    for key in sorted(existing_months):
        df = month_data[key]
        trips = len(df)
        # Optionally, you can compute by started_at month grouping; but files are per-month already
        monthly_rows.append({"month": key, "rides": int(trips)})
    with open(MONTHLY_TOTALS_JSON, "w", encoding="utf-8") as f:
        json.dump(monthly_rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {MONTHLY_TOTALS_JSON} with {len(monthly_rows)} rows")

    # 3) Latest available month (for Top 5s)
    latest_month = sorted(existing_months)[-1]
    df_latest = month_data[latest_month].copy()

    # Filter rider types explicitly
    df_latest["member_casual"] = df_latest["member_casual"].fillna("unknown").str.lower()
    df_casual = df_latest[df_latest["member_casual"] == "casual"]
    df_member = df_latest[df_latest["member_casual"] == "member"]

    # Starts — casual
    starts_casual = (
        df_casual.groupby(["start_station_id","start_station_name"], dropna=False)
        .size().reset_index(name="trips")
        .rename(columns={"start_station_id":"station_id","start_station_name":"station_name"})
    )
    # Ends — casual
    ends_casual = (
        df_casual.groupby(["end_station_id","end_station_name"], dropna=False)
        .size().reset_index(name="trips")
        .rename(columns={"end_station_id":"station_id","end_station_name":"station_name"})
    )
    # Starts — member
    starts_member = (
        df_member.groupby(["start_station_id","start_station_name"], dropna=False)
        .size().reset_index(name="trips")
        .rename(columns={"start_station_id":"station_id","start_station_name":"station_name"})
    )
    # Ends — member
    ends_member = (
        df_member.groupby(["end_station_id","end_station_name"], dropna=False)
        .size().reset_index(name="trips")
        .rename(columns={"end_station_id":"station_id","end_station_name":"station_name"})
    )

    result = {
        "latest_month": latest_month,  # e.g., "2025-07"
        "top5": {
            "starts": {
                "casual": top5(starts_casual),
                "member": top5(starts_member),
            },
            "ends": {
                "casual": top5(ends_casual),
                "member": top5(ends_member),
            }
        }
    }

    with open(TOP_STATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote {TOP_STATIONS_JSON}")

if __name__ == "__main__":
    run()
