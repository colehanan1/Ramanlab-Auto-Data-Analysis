#!/usr/bin/env python3
"""
Standalone test script for InfluxDB enclosure environment queries.

Run from ANY machine with network access to the InfluxDB host (10.229.137.171:8086).

Usage:
    # Quick connectivity test (last 1 hour of data):
    python test_influxdb_enclosure.py

    # Test with specific time range:
    python test_influxdb_enclosure.py --start "2026-03-10T08:00:00Z" --end "2026-03-10T20:00:00Z"

    # Test pre-starved + starved period queries (simulates pipeline):
    python test_influxdb_enclosure.py --born-date 2026-03-05 --starved-date 2026-03-09 --starved-time 14:00

    # Discover all flynursery2 entity IDs:
    python test_influxdb_enclosure.py --discover
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.parse
import statistics
from datetime import datetime, timedelta
from pathlib import Path


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


SENSORS = {
    "temp_inside":      ("°C",  "flynursery2_temperature_fly_2"),
    "temp_outside":     ("°C",  "flynursery2_temperature_room"),
    "humidity_inside":  ("%",   "flynursery2_humidity_fly_2"),
    "humidity_outside": ("%",   "flynursery2_humidity_room"),
    "pressure_inside":  ("hPa", "flynursery2_pressure_fly_2"),
    "pressure_outside": ("hPa", "flynursery2_pressure_room"),
    "light_brightness": ("%",   "flynursery2_fly_sun_brightness"),
}

UNITS = {
    "temp_inside": "C", "temp_outside": "C",
    "humidity_inside": "%", "humidity_outside": "%",
    "pressure_inside": "hPa", "pressure_outside": "hPa",
    "light_brightness": "%",
}

LABELS = {
    "temp_inside": "Temp (enclosure)", "temp_outside": "Temp (room)",
    "humidity_inside": "Humidity (enclosure)", "humidity_outside": "Humidity (room)",
    "pressure_inside": "Pressure (enclosure)", "pressure_outside": "Pressure (room)",
    "light_brightness": "Light brightness",
}


def influx_query(host, port, db, q, username=None, password=None):
    """Execute an InfluxQL query and return parsed JSON."""
    params = {"db": db, "epoch": "s", "q": q}
    if username:
        params["u"] = username
    if password:
        params["p"] = password
    url = f"http://{host}:{port}/query?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def test_connectivity(host, port, db, username, password):
    """Test basic InfluxDB connectivity."""
    print(f"=== Connectivity Test: http://{host}:{port} db={db} user={username} ===\n")
    data = influx_query(host, port, db, "SHOW DATABASES", username, password)
    if data is None:
        print("FAIL: Cannot connect to InfluxDB.")
        return False
    dbs = []
    for r in data.get("results", []):
        for s in r.get("series", []):
            for v in s.get("values", []):
                dbs.append(v[0])
    if db in dbs:
        print(f"OK: Connected. Database '{db}' exists.")
        print(f"    All databases: {dbs}\n")
        return True
    else:
        print(f"FAIL: Connected but database '{db}' not found.")
        print(f"    Available databases: {dbs}\n")
        return False


def discover_entities(host, port, db, username, password):
    """Discover all flynursery2 entity IDs across all measurements."""
    print("=== Discovering flynursery2 Entity IDs ===\n")
    measurements = ["°C", "%", "hPa", "RPM", "W", "kWh"]
    for meas in measurements:
        q = f'SHOW TAG VALUES FROM "{meas}" WITH KEY = "entity_id" WHERE "entity_id" =~ /flynursery2/'
        data = influx_query(host, port, db, q, username, password)
        if data is None:
            continue
        for r in data.get("results", []):
            for s in r.get("series", []):
                vals = [v[1] for v in s.get("values", [])]
                if vals:
                    print(f"  Measurement '{meas}':")
                    for v in vals:
                        print(f"    - {v}")
    print()


def query_sensor_range(host, port, db, start_utc, end_utc, username=None, password=None):
    """Query all sensors for a time range and print results. Returns summary dict."""
    results = {}
    for sensor_label, (measurement, entity_id) in SENSORS.items():
        q = (
            f'SELECT mean("value") AS "v" FROM "{measurement}" '
            f'WHERE "entity_id" = \'{entity_id}\' '
            f'AND time >= \'{start_utc}\' AND time <= \'{end_utc}\' '
            f'GROUP BY time(1m) fill(none)'
        )
        data = influx_query(host, port, db, q, username, password)
        if data is None:
            print(f"  {sensor_label}: QUERY FAILED")
            continue
        series = data.get("results", [{}])[0].get("series", [])
        if not series:
            print(f"  {sensor_label} ({entity_id}): NO DATA in range")
            continue
        values = [pt[1] for pt in series[0].get("values", []) if pt[1] is not None]
        if not values:
            print(f"  {sensor_label} ({entity_id}): Empty values")
            continue

        mean_v = statistics.mean(values)
        min_v = min(values)
        max_v = max(values)
        std_v = statistics.pstdev(values)
        unit = UNITS.get(sensor_label, "")
        results[sensor_label] = {
            "mean": round(mean_v, 2),
            "min": round(min_v, 2),
            "max": round(max_v, 2),
            "std": round(std_v, 2),
            "count": len(values),
        }
        print(f"  {sensor_label}: mean={mean_v:.2f}{unit}, min={min_v:.2f}, max={max_v:.2f}, std={std_v:.2f}, samples={len(values)}")

    # Light exposure
    if "light_brightness" in results and results["light_brightness"]["count"] > 0:
        q_light = (
            f'SELECT mean("value") AS "v" FROM "%" '
            f'WHERE "entity_id" = \'flynursery2_fly_sun_brightness_pct\' '
            f'AND time >= \'{start_utc}\' AND time <= \'{end_utc}\' '
            f'GROUP BY time(1m) fill(none)'
        )
        data = influx_query(host, port, db, q_light, username, password)
        if data:
            series = data.get("results", [{}])[0].get("series", [])
            if series:
                vals = [pt[1] for pt in series[0].get("values", []) if pt[1] is not None]
                if vals:
                    light_on = sum(1 for v in vals if v > 5)
                    total = len(vals)
                    pct_on = round(100.0 * light_on / total, 1)
                    print(f"  Light exposure: {pct_on}% light, {round(100.0 - pct_on, 1)}% dark")
                    results["light_on_pct"] = pct_on
                    results["light_off_pct"] = round(100.0 - pct_on, 1)

    return results


def test_basic_query(host, port, db, start_utc, end_utc, username, password):
    """Test querying all sensors for a time range."""
    print(f"=== Sensor Query Test ===")
    print(f"    Range: {start_utc} -> {end_utc}\n")
    results = query_sensor_range(host, port, db, start_utc, end_utc, username, password)
    if not results:
        print("\nFAIL: No sensor data returned. Possible causes:")
        print("  - Entity IDs may be different (run with --discover)")
        print("  - No data in this time range")
        print("  - InfluxDB measurement names differ")
    else:
        found = list(results.keys())
        missing = [k for k in SENSORS if k not in results and k != "light_brightness"]
        print(f"\nOK: Got data for {len(found)} sensors.")
        if missing:
            print(f"  Missing: {missing} (run --discover to check entity IDs)")
    print()
    return results


def test_period_queries(host, port, db, born_date, starved_date, starved_time, session_start_utc, username, password):
    """Test the pre-starved and starved period queries (simulates what the pipeline does)."""
    print("=== Period Query Test (simulates pipeline) ===\n")

    born_utc = f"{born_date}T00:00:00Z"
    starved_utc = f"{starved_date}T{starved_time}:00Z"

    print(f"--- Pre-starved period (birth -> starvation) ---")
    print(f"    {born_utc} -> {starved_utc}")
    pre_results = query_sensor_range(host, port, db, born_utc, starved_utc, username, password)

    print(f"\n--- Starved period (starvation -> session start) ---")
    print(f"    {starved_utc} -> {session_start_utc}")
    starved_results = query_sensor_range(host, port, db, starved_utc, session_start_utc, username, password)

    print()
    if pre_results:
        print(f"OK: Pre-starved data found for {len(pre_results)} sensors.")
    else:
        print("WARN: No pre-starved data. Is InfluxDB retention long enough to cover birth date?")

    if starved_results:
        print(f"OK: Starved data found for {len(starved_results)} sensors.")
    else:
        print("WARN: No starved data. Check time range and entity IDs.")

    # Print formatted summary
    print("\n--- Formatted summary (as it would appear in session_metadata.txt) ---\n")
    for period_name, results in [("Pre-starved", pre_results), ("Starved", starved_results)]:
        if not results:
            print(f"Enclosure Environment: {period_name} Period -- NO DATA\n")
            continue
        print(f"Enclosure Environment: {period_name} Period")
        print("-" * 55)
        for key in ["temp_inside", "temp_outside", "humidity_inside", "humidity_outside",
                     "pressure_inside", "pressure_outside", "light_brightness"]:
            if key not in results:
                continue
            s = results[key]
            unit = UNITS.get(key, "")
            lbl = LABELS.get(key, key)
            print(f"  {lbl}: mean={s['mean']}{unit}, min={s['min']}{unit}, max={s['max']}{unit}, std={s['std']}, samples={s['count']}")
        if "light_on_pct" in results:
            print(f"  Light exposure: {results['light_on_pct']}% light, {results['light_off_pct']}% dark")
        print()


def main():
    parser = argparse.ArgumentParser(description="Test InfluxDB enclosure environment queries")
    parser.add_argument("--host", default=os.environ.get("INFLUXDB_HOST", "localhost"), help="InfluxDB host")
    parser.add_argument("--port", type=int, default=int(os.environ.get("INFLUXDB_PORT", "8086")), help="InfluxDB port")
    parser.add_argument("--db", default=os.environ.get("INFLUXDB_DB", "homeassistant"), help="InfluxDB database")
    parser.add_argument("--user", default=os.environ.get("INFLUXDB_USER", ""), help="InfluxDB username")
    parser.add_argument("--password", default=os.environ.get("INFLUXDB_PASS", ""), help="InfluxDB password")
    parser.add_argument("--start", default=None, help="Query start time (UTC ISO)")
    parser.add_argument("--end", default=None, help="Query end time (UTC ISO)")
    parser.add_argument("--discover", action="store_true", help="Discover all flynursery2 entity IDs")
    parser.add_argument("--born-date", default=None, help="Born date YYYY-MM-DD (for period test)")
    parser.add_argument("--starved-date", default=None, help="Starved date YYYY-MM-DD (for period test)")
    parser.add_argument("--starved-time", default=None, help="Starved time HH:MM 24h (for period test)")
    parser.add_argument("--session-start", default=None, help="Session start UTC ISO (for period test)")
    args = parser.parse_args()

    if not test_connectivity(args.host, args.port, args.db, args.user, args.password):
        sys.exit(1)

    if args.discover:
        discover_entities(args.host, args.port, args.db, args.user, args.password)
        return

    if args.born_date and args.starved_date:
        starved_time = args.starved_time or "14:00"
        session_start = args.session_start or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        test_period_queries(args.host, args.port, args.db,
                            args.born_date, args.starved_date, starved_time, session_start,
                            args.user, args.password)
        return

    if args.start and args.end:
        start_utc = args.start
        end_utc = args.end
    else:
        now = datetime.utcnow()
        start_utc = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"(No --start/--end provided, using last 1 hour)\n")

    test_basic_query(args.host, args.port, args.db, start_utc, end_utc, args.user, args.password)


if __name__ == "__main__":
    main()
