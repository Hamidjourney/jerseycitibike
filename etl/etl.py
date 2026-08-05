# etl/etl.py
import os, io, json
import requests, zipfile
import pandas as pd
from datetime import datetime, timezone

# ---------- Config ----------
SYSTEM_PREFIX = "JC"  # Jersey City (used only for the S3 filename)
START_YEAR = 2025
START_MONTH = 1
S3_BASE = "https://s3.amazonaws.com/tripdata"
OUT_DIR = os.path.join("docs", "data")
MONTHLY_TOTALS_JSON = os.path.join(OUT_DIR, "monthly_totals.json")
TOP_STATIONS_JSON = os.path.join(OUT_DIR, "top_stations_latest.json")
INVENTORY_CSV = os.path.join(OUT_DIR, "inventory_latest.csv")

# Station id prefixes that belong to the actual JC/Hoboken system. Trips can
# start or end at out-of-system stations (e.g. NYC stations, which use plain
# numeric ids like "5216.04") when a bike is picked up in one system and
# dropped in another. Those are excluded from station-level rankings, but
# still counted in the plain monthly totals above.
NJ_ID_PATTERN = r"^(JC|HB)"


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


# ---------- Station identity resolution ----------
# A trip row can have a station name with a blank id, or (less commonly) an
# id with a blank name. Grouping directly on the raw columns fragments those
# rows into their own "partial" groups instead of folding into the real
# station's counts. To fix this we build one canonical id<->name reference
# from the month's data (station appears with complete data on at least one
# side, most of the time), then fill in whichever side is missing before any
# counting happens.
def build_station_reference(df: pd.DataFrame):
    starts = df[["start_station_id", "start_station_name"]].rename(
        columns={"start_station_id": "station_id", "start_station_name": "station_name"}
    )
    ends = df[["end_station_id", "end_station_name"]].rename(
        columns={"end_station_id": "station_id", "end_station_name": "station_name"}
    )
    combined = pd.concat([starts, ends], ignore_index=True).dropna().drop_duplicates()

    name_to_id = combined.drop_duplicates("station_name").set_index("station_name")["station_id"]
    id_to_name = combined.drop_duplicates("station_id").set_index("station_id")["station_name"]
    return name_to_id, id_to_name


def resolve_station_ids(df: pd.DataFrame, name_to_id: pd.Series):
    start_id = df["start_station_id"].copy()
    mask = start_id.isna() & df["start_station_name"].notna()
    start_id[mask] = df.loc[mask, "start_station_name"].map(name_to_id)

    end_id = df["end_station_id"].copy()
    mask = end_id.isna() & df["end_station_name"].notna()
    end_id[mask] = df.loc[mask, "end_station_name"].map(name_to_id)

    return start_id, end_id


def grouped_counts(resolved_ids: pd.Series, id_to_name: pd.Series) -> pd.DataFrame:
    """Count trips per resolved station_id. Rows where the id could not be
    resolved at all (id and name both blank on that side) are dropped —
    there is nothing to attribute them to."""
    counts = resolved_ids.dropna().value_counts().rename("trips").reset_index()
    counts.columns = ["station_id", "trips"]
    counts["station_name"] = counts["station_id"].map(id_to_name)
    return counts


def top5(grouped: pd.DataFrame) -> list[dict]:
    out = []
    for _, row in grouped.nlargest(5, "trips").iterrows():
        out.append({
            "station_id": safe_name(row.get("station_id")),
            "station_name": safe_name(row.get("station_name")),
            "trips": int(row.get("trips", 0)),
        })
    return out


def net_flow(starts: pd.DataFrame, ends: pd.DataFrame, top_n: int = 5) -> list[dict]:
    # starts/ends: columns ['station_id','station_name','trips']
    # Join on station_id only — joining on station_id + station_name (as
    # before) can silently fail to match a station's starts to its ends if
    # the name differs even slightly between the two sides.
    merged = pd.merge(
        starts, ends, on="station_id", how="outer", suffixes=("_start", "_end")
    )
    merged["trips_start"] = merged["trips_start"].fillna(0)
    merged["trips_end"] = merged["trips_end"].fillna(0)
    merged["station_name"] = merged["station_name_start"].combine_first(merged["station_name_end"])

    # Sign convention unchanged: positive = net exporter ("bleeder", needs
    # restocking), negative = net importer ("sink", fills up).
    merged["net"] = merged["trips_start"] - merged["trips_end"]
    merged = merged.sort_values("net", ascending=False)

    bleeders = merged.head(top_n)   # most positive: net exporters, need restocking
    sinks = merged.tail(top_n)      # most negative: net importers, fill up
    # Guard against overlap if there are fewer than 2*top_n stations total
    combined = pd.concat([bleeders, sinks]).drop_duplicates(subset=["station_id"])
    combined = combined.sort_values("net", ascending=False)

    out = []
    for _, row in combined.iterrows():
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

    # Force-refetch the last 2 months in the checked range (cheap), since we
    # don't know in advance which of them is actually published yet, and we
    # always want a real DataFrame in hand for whichever one turns out latest.
    force_refetch_keys = {f"{y}-{m:02d}" for (y, m) in months_to_check[-2:]}

    monthly_rows = []
    latest_df = None
    latest_month_key = None

    for (year, month) in months_to_check:
        key = f"{year}-{month:02d}"

        # Skip refetching months we already have, UNLESS it's one of the
        # last 2 in range (Citi Bike sometimes republishes recent months with
        # corrections, and we need the raw month in memory to compute
        # top-station / net-flow stats for whichever one is truly latest).
        if key in existing and key not in force_refetch_keys:
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
            if key in existing:
                monthly_rows.append(existing[key])
            continue

        try:
            df = read_trips_from_zip(z, year, month)
        except Exception as e:
            print("  -> error reading zip:", e)
            if key in existing:
                monthly_rows.append(existing[key])
            continue

        member_trips = int((df["member_casual"] == "member").sum())
        casual_trips = int((df["member_casual"] == "casual").sum())
        monthly_rows.append({
            "month": key,
            "rides": int(len(df)),
            "member": member_trips,
            "casual": casual_trips,
        })

        # Months are processed in ascending chronological order, so the last
        # one that successfully fetches is the latest actually-published month.
        latest_df = df
        latest_month_key = key

    # Preserve chronological order, de-dup by month key (latest write wins)
    dedup = {r["month"]: r for r in monthly_rows}
    monthly_rows = [dedup[k] for k in sorted(dedup.keys())]

    with open(MONTHLY_TOTALS_JSON, "w", encoding="utf-8") as f:
        json.dump(monthly_rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {MONTHLY_TOTALS_JSON} with {len(monthly_rows)} rows")

    # If nothing new was published this run, still re-fetch the most recent
    # known month so top5/net_flow/inventory can be (re)computed — raw trip
    # data isn't cached between runs, only the aggregated monthly totals are.
    if latest_df is None and monthly_rows:
        fallback_key = max(r["month"] for r in monthly_rows)
        fb_year, fb_month = (int(p) for p in fallback_key.split("-"))
        print(f"No new month published this run; re-fetching {fallback_key} to compute station stats.")
        z = None
        for url in month_url(fb_year, fb_month):
            print("Checking:", url)
            z = try_fetch_zip(url)
            if z is not None:
                print(f"  -> found ({len(z)} bytes)")
                break
        if z is not None:
            try:
                latest_df = read_trips_from_zip(z, fb_year, fb_month)
                latest_month_key = fallback_key
            except Exception as e:
                print("  -> error reading zip:", e)

    if latest_df is None:
        print("No latest-month data available this run; leaving top_stations/inventory files untouched.")
        return

    df_latest = latest_df.copy()
    df_latest["member_casual"] = df_latest["member_casual"].fillna("unknown").str.lower()

    # Build one canonical station id<->name reference from this month's data,
    # then resolve start/end station ids for every row (filling in a blank
    # id from the name when possible). This is done once, up front, for the
    # whole month — not separately per rider type.
    name_to_id, id_to_name = build_station_reference(df_latest)
    start_id_resolved, end_id_resolved = resolve_station_ids(df_latest, name_to_id)
    df_latest = df_latest.assign(
        _start_id_resolved=start_id_resolved,
        _end_id_resolved=end_id_resolved,
    )

    result = {"latest_month": latest_month_key, "top5": {}, "net_flow": {}}

    for rider_type in ("casual", "member"):
        sub = df_latest[df_latest["member_casual"] == rider_type]

        starts = grouped_counts(sub["_start_id_resolved"], id_to_name)
        ends = grouped_counts(sub["_end_id_resolved"], id_to_name)

        # Restrict station-level rankings to in-system (JC/HB) stations only.
        starts = starts[starts["station_id"].astype(str).str.match(NJ_ID_PATTERN)]
        ends = ends[ends["station_id"].astype(str).str.match(NJ_ID_PATTERN)]

        result["top5"].setdefault("starts", {})[rider_type] = top5(starts)
        result["top5"].setdefault("ends", {})[rider_type] = top5(ends)
        result["net_flow"][rider_type] = net_flow(starts, ends)  # top 5 bleeders + bottom 5 sinks

    with open(TOP_STATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote {TOP_STATIONS_JSON}")

    # ---- Inventory trajectory table (per selected station, 15-min buckets) ----
    # Reuses name_to_id / id_to_name / the resolved id columns already built
    # above for top5/net_flow — nothing read from disk, since GitHub Actions
    # only ever has whatever this run fetched, never a local CSV.

    # Next phase: a pipeline will supply this station config instead of it
    # being hardcoded here. Capacity is from the GBFS station_information
    # feed; baseline is an assumed bike count at 05:00 (data-checked for
    # McGinley Square; a test assumption for Clinton St & 7 St, to revisit).
    selected_stations = {
        "JC055": {"capacity": 22, "baseline": 15},  # McGinley Square
        "HB303": {"capacity": 18, "baseline": 12},  # Clinton St & 7 St
    }
    selected_station_ids = list(selected_stations.keys())

    interval = "15min"
    interval_frames = []

    for sid in selected_station_ids:
        pickup_ts = (
            df_latest[df_latest["_start_id_resolved"] == sid]
            .groupby(pd.Grouper(key="started_at", freq=interval))
            .size()
            .rename("pickups")
        )
        return_ts = (
            df_latest[df_latest["_end_id_resolved"] == sid]
            .groupby(pd.Grouper(key="ended_at", freq=interval))
            .size()
            .rename("returns")
        )
        pickup_ts.index.name = "datetime"
        return_ts.index.name = "datetime"

        ts = pd.concat([pickup_ts, return_ts], axis=1).fillna(0)
        ts["net_flow"] = ts["returns"] - ts["pickups"]
        ts["station_id"] = sid
        ts["station_name"] = id_to_name.get(sid, sid)
        ts["capacity"] = selected_stations[sid]["capacity"]
        ts["baseline"] = selected_stations[sid]["baseline"]
        ts["date"] = ts.index.date
        ts["hour"] = ts.index.hour
        ts["minute"] = ts.index.minute
        ts["op_date"] = (ts.index - pd.Timedelta(hours=5)).date
        ts["daily_cumulative"] = ts.groupby("op_date")["net_flow"].cumsum()

        # Precompute the bounds status per bucket, so the dashboard doesn't
        # need to re-derive the out-of-bounds definition itself. Absolute
        # estimated bike count = baseline + daily_cumulative.
        absolute = ts["baseline"] + ts["daily_cumulative"]
        ts["status"] = "in_bounds"
        ts.loc[absolute > ts["capacity"], "status"] = "over_capacity"
        ts.loc[absolute < 0, "status"] = "below_zero"

        interval_frames.append(ts.reset_index())

    interval_flow = pd.concat(interval_frames, ignore_index=True)

    # Drop the partial first operations day. Its 00:00-04:59 timestamps
    # belong, under the 5am operations-day convention, to the last calendar
    # day of the PREVIOUS month — but this run only has the current month's
    # data, so that day would only ever show a near-empty ~5-hour stub.
    latest_year, latest_month = (int(p) for p in latest_month_key.split("-"))
    first_day_of_month = pd.Timestamp(latest_year, latest_month, 1).date()
    interval_flow = interval_flow[interval_flow["op_date"] >= first_day_of_month]

    # ---- Save interval_flow for the dashboard ----
    # Flat CSV, latest month only — this table is large and purely tabular
    # (no real nesting need), so it's far more compact than the equivalent
    # JSON, and this file is fully overwritten each run (like TOP_STATIONS_JSON,
    # not accumulated like MONTHLY_TOTALS_JSON).
    interval_flow.to_csv(INVENTORY_CSV, index=False)
    print(f"Wrote {INVENTORY_CSV} with {len(interval_flow)} rows")


if __name__ == "__main__":
    run()
