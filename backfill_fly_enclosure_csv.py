#!/usr/bin/env python3
"""
Backfill enclosure environment data for all fly sessions.

Scans all session directories under /securedstorage/DATAsec/cole/Data-secured/,
reads session_metadata.txt for born date, starved date/time, and first training time,
then queries InfluxDB for mean enclosure values during TWO periods:
  1. Pre-starved: born date -> starved date/time
  2. Starved: starved date/time -> first training start

Produces ONE ROW PER FLY (detected from fly*_global_distance_stats_class_0.json files).
Cross-references flagged-flys-truth.csv to add a flagged_state column.

Usage:
    python backfill_fly_enclosure_csv.py
    python backfill_fly_enclosure_csv.py --dry-run          # preview without writing
    python backfill_fly_enclosure_csv.py --master-only      # only write master CSV
"""

import argparse
import csv
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

DATA_ROOT = Path("/securedstorage/DATAsec/cole/Data-secured")
FLAGGED_CSV = Path("/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/flagged-flys-truth.csv")


def _load_dotenv():
    """Load .env file from script directory into os.environ (no external deps)."""
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

INFLUX_HOST = os.environ.get("INFLUXDB_HOST", "localhost")
INFLUX_PORT = int(os.environ.get("INFLUXDB_PORT", "8086"))
INFLUX_DB = os.environ.get("INFLUXDB_DB", "homeassistant")
INFLUX_USER = os.environ.get("INFLUXDB_USER", "")
INFLUX_PASS = os.environ.get("INFLUXDB_PASS", "")

# InfluxDB data starts from Jan 30, 2026
INFLUX_DATA_START = datetime(2026, 1, 30)

SENSORS = {
    "temp_inside_C":      ("°C",  "flynursery2_temperature_fly_2"),
    "temp_outside_C":     ("°C",  "flynursery2_temperature_room"),
    "humidity_inside_pct": ("%",  "flynursery2_humidity_fly_2"),
    "humidity_outside_pct":("%",  "flynursery2_humidity_room"),
    "pressure_inside_hPa": ("hPa","flynursery2_pressure_fly_2"),
    "pressure_outside_hPa":("hPa","flynursery2_pressure_room"),
    "light_brightness_pct":("%",  "flynursery2_fly_sun_brightness"),
    "heat_pad_watts":      ("W",  "flynursery2_heat_pad_watts"),
}

SENSOR_KEYS = list(SENSORS.keys())

# Build columns: pre_starved_mean_* and starved_mean_* for each sensor
CSV_COLUMNS = [
    "experiment_group", "session_dir", "fly_number", "fly_id", "fly_type", "born_date",
    "starved_date", "starved_time_local", "starved_datetime",
    "first_training_start_local", "fly_age_days", "starvation_hours",
    "flagged_state", "flagged_comment",
    "pre_starved_influxdb_available", "starved_influxdb_available",
] + [f"pre_starved_mean_{k}" for k in SENSOR_KEYS] + [f"starved_mean_{k}" for k in SENSOR_KEYS]


def load_flagged_flies(csv_path):
    """Load flagged-flys-truth.csv into a lookup dict.
    Key: (dataset, session_dir, fly_number) -> {"state": int, "comment": str}
    State meanings: 1 = good/alive, 0 = questionable, -1 = dead/bad
    """
    flagged = {}
    if not csv_path.exists():
        print(f"WARNING: Flagged flies CSV not found: {csv_path}")
        return flagged
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row.get("dataset", "").strip()
            fly = row.get("fly", "").strip()
            try:
                fly_num = int(row.get("fly_number", "").strip())
            except (ValueError, AttributeError):
                continue
            state = row.get("FLY-State(1, 0, -1)", "").strip()
            comment = row.get("comment", "").strip()
            base_dataset = re.sub(r"-flagged$", "", dataset)
            flagged[(base_dataset, fly, fly_num)] = {
                "state": state,
                "comment": comment,
            }
    return flagged


def influx_query(q):
    """Execute InfluxQL query, return parsed JSON or None."""
    params = {
        "db": INFLUX_DB, "epoch": "s", "q": q,
        "u": INFLUX_USER, "p": INFLUX_PASS,
    }
    url = f"http://{INFLUX_HOST}:{INFLUX_PORT}/query?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def query_sensor_means(start_utc, end_utc):
    """Query mean value for each sensor in time range. Returns dict of sensor_key -> mean."""
    results = {}
    for key, (measurement, entity_id) in SENSORS.items():
        q = (
            f'SELECT mean("value") FROM "{measurement}" '
            f'WHERE "entity_id" = \'{entity_id}\' '
            f'AND time >= \'{start_utc}\' AND time <= \'{end_utc}\''
        )
        data = influx_query(q)
        if data is None:
            continue
        try:
            series = data["results"][0].get("series", [])
            if series and series[0]["values"][0][1] is not None:
                results[key] = round(series[0]["values"][0][1], 2)
        except (KeyError, IndexError):
            pass
    return results


def parse_date(s):
    """Parse YYYY-MM-DD date string."""
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def parse_time_to_pm(time_str):
    """Parse time string. Per user: all times are PM.
    '1:27' -> 13:27, '12:26' -> 12:26, '11:40' -> 23:40 (11:40 PM)."""
    time_str = time_str.strip()
    m = re.match(r"(\d{1,2}):(\d{2})", time_str)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2))
    if hour < 12:
        hour += 12
    return f"{hour:02d}:{minute:02d}"


def parse_session_metadata(meta_path):
    """Parse session_metadata.txt, return dict with key fields."""
    info = {}
    text = meta_path.read_text().replace("\r", "")

    m = re.search(r"Fly ID:\s*(.+)", text)
    if m:
        info["fly_id"] = m.group(1).strip()

    m = re.search(r"Fly Type:\s*(.+)", text)
    if m:
        info["fly_type"] = m.group(1).strip()

    m = re.search(r"Born \(approx\):\s*(.+)", text)
    if m:
        info["born_date"] = m.group(1).strip()

    m = re.search(r"Starved Date:\s*(.+)", text)
    if m:
        info["starved_date"] = m.group(1).strip()

    m = re.search(r"Starved Time \(local\):\s*(.+)", text)
    if m:
        info["starved_time_raw"] = m.group(1).strip()

    m = re.search(r"First training trial start \(local\):\s*(.+)", text)
    if m:
        info["first_training_start_local"] = m.group(1).strip()

    return info


def find_first_training_time_from_files(session_dir):
    """Fallback: extract first training start time from CSV filenames."""
    training_files = []
    for f in session_dir.iterdir():
        m = re.search(r"training_1_(\d{8})_(\d{6})\.", f.name)
        if m:
            try:
                dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
                training_files.append(dt)
            except ValueError:
                pass
    if training_files:
        return min(training_files)
    return None


def detect_fly_numbers(session_dir):
    """Detect fly numbers from fly*_global_distance_stats_class_0.json files."""
    fly_nums = []
    for f in session_dir.iterdir():
        m = re.match(r"fly(\d+)_global_distance_stats_class_0\.json", f.name)
        if m:
            fly_nums.append(int(m.group(1)))
    return sorted(fly_nums)


def process_session(session_dir, experiment_group, flagged_lookup):
    """Process a single session directory. Returns list of row dicts (one per fly)."""
    meta_path = session_dir / "session_metadata.txt"
    if not meta_path.exists():
        return None

    info = parse_session_metadata(meta_path)
    if not info.get("born_date") or not info.get("starved_date"):
        return None

    born_dt = parse_date(info["born_date"])
    starved_dt = parse_date(info["starved_date"])
    if not born_dt or not starved_dt:
        return None

    starved_time_24h = parse_time_to_pm(info.get("starved_time_raw", "12:00"))
    if not starved_time_24h:
        starved_time_24h = "12:00"

    starved_datetime = datetime.strptime(
        f"{info['starved_date']} {starved_time_24h}", "%Y-%m-%d %H:%M"
    )

    first_training_dt = None
    if info.get("first_training_start_local"):
        try:
            first_training_dt = datetime.strptime(
                info["first_training_start_local"], "%Y-%m-%d %H:%M:%S"
            )
        except ValueError:
            pass

    if first_training_dt is None:
        first_training_dt = find_first_training_time_from_files(session_dir)

    if first_training_dt is None:
        return None

    fly_age_days = (first_training_dt - born_dt).days
    starvation_delta = first_training_dt - starved_datetime
    starvation_hours = round(starvation_delta.total_seconds() / 3600, 2)

    # --- Query InfluxDB for TWO periods ---
    # Period 1: Pre-starved (born -> starved)
    pre_starved_data = {}
    pre_starved_available = "no_pre_influx"
    if born_dt >= INFLUX_DATA_START:
        born_utc = born_dt.strftime("%Y-%m-%dT00:00:00Z")
        starved_utc = starved_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
        pre_starved_data = query_sensor_means(born_utc, starved_utc)
        pre_starved_available = "yes" if pre_starved_data else "no"

    # Period 2: Starved (starved -> first training start)
    starved_data = {}
    starved_available = "no_pre_influx"
    if starved_datetime >= INFLUX_DATA_START:
        starved_utc = starved_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
        training_utc = first_training_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        starved_data = query_sensor_means(starved_utc, training_utc)
        starved_available = "yes" if starved_data else "no"

    # Detect flies
    fly_numbers = detect_fly_numbers(session_dir)
    if not fly_numbers:
        try:
            fly_numbers = [int(info.get("fly_id", "1"))]
        except ValueError:
            fly_numbers = [1]

    # Build one row per fly
    rows = []
    for fly_num in fly_numbers:
        flag_info = flagged_lookup.get((experiment_group, session_dir.name, fly_num), {})
        flagged_state = flag_info.get("state", "")
        flagged_comment = flag_info.get("comment", "")

        row = {
            "experiment_group": experiment_group,
            "session_dir": session_dir.name,
            "fly_number": fly_num,
            "fly_id": info.get("fly_id", ""),
            "fly_type": info.get("fly_type", ""),
            "born_date": info["born_date"],
            "starved_date": info["starved_date"],
            "starved_time_local": starved_time_24h,
            "starved_datetime": starved_datetime.strftime("%Y-%m-%dT%H:%M"),
            "first_training_start_local": first_training_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "fly_age_days": fly_age_days,
            "starvation_hours": starvation_hours,
            "flagged_state": flagged_state,
            "flagged_comment": flagged_comment,
            "pre_starved_influxdb_available": pre_starved_available,
            "starved_influxdb_available": starved_available,
        }
        for key in SENSOR_KEYS:
            row[f"pre_starved_mean_{key}"] = pre_starved_data.get(key, "")
            row[f"starved_mean_{key}"] = starved_data.get(key, "")

        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Backfill fly enclosure environment CSVs")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    parser.add_argument("--master-only", action="store_true", help="Only write master CSV")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Root data directory")
    parser.add_argument("--flagged-csv", default=str(FLAGGED_CSV), help="Path to flagged-flys-truth.csv")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    flagged_lookup = load_flagged_flies(Path(args.flagged_csv))
    print(f"Loaded {len(flagged_lookup)} flagged fly entries\n")

    all_rows = []
    errors = []

    experiment_groups = sorted([
        d.name for d in data_root.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "CSVs" and d.name != "flagged"
    ])

    print(f"Scanning {len(experiment_groups)} experiment groups in {data_root}\n")

    for group in experiment_groups:
        group_dir = data_root / group
        sessions = sorted([
            d for d in group_dir.iterdir()
            if d.is_dir() and (d / "session_metadata.txt").exists()
        ])

        if not sessions:
            continue

        print(f"  {group}: {len(sessions)} sessions")

        for session_dir in sessions:
            try:
                rows = process_session(session_dir, group, flagged_lookup)
                if rows is None:
                    errors.append(f"  SKIP {session_dir.name}: missing metadata or training files")
                    continue

                all_rows.extend(rows)

                fly_nums = [r["fly_number"] for r in rows]
                flagged_flies = [r["fly_number"] for r in rows if r["flagged_state"]]
                pre_status = rows[0]["pre_starved_influxdb_available"]
                starved_status = rows[0]["starved_influxdb_available"]
                flag_str = f", flagged={flagged_flies}" if flagged_flies else ""
                print(f"    {session_dir.name}: {len(rows)} flies {fly_nums}, "
                      f"age={rows[0]['fly_age_days']}d, "
                      f"starved={rows[0]['starvation_hours']}h, "
                      f"pre_starved={pre_status}, starved={starved_status}{flag_str}")

                if not args.dry_run and not args.master_only:
                    csv_path = session_dir / "enclosure_environment.csv"
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
                        writer.writeheader()
                        for row in rows:
                            writer.writerow(row)

            except Exception as e:
                errors.append(f"  ERROR {session_dir.name}: {e}")

    if all_rows and not args.dry_run:
        master_path = data_root / "all_flies_enclosure_summary.csv"
        with open(master_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)
        print(f"\nMaster CSV: {master_path}")

    total_sessions = len(set((r["experiment_group"], r["session_dir"]) for r in all_rows))
    print(f"\nProcessed: {len(all_rows)} fly rows across {total_sessions} sessions")
    if errors:
        print(f"Skipped/errors: {len(errors)}")
        for e in errors:
            print(e)


if __name__ == "__main__":
    main()
