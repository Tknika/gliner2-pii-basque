#!/usr/bin/env python3
"""Streets, towns and postcodes of the Basque Country from OpenStreetMap.

Reads Geofabrik extracts (País Vasco, Navarra, Aquitaine filtered to
postcodes 64xxx). Street names and postcodes come from address points
(`addr:street` + `addr:postcode` on nodes and buildings), so every
(street, town, postcode) triple exists in reality. Each address is
assigned to its `addr:city` tag or to the nearest place node (≤ 10 km).

    python scripts/fetch_sources/fetch_osm.py --download --extract

Writes streets.csv and towns.csv into --out. Data © OpenStreetMap
contributors, ODbL 1.0.
"""

import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path

import osmium
import requests

EXTRACTS = {
    "pais-vasco": "https://download.geofabrik.de/europe/spain/pais-vasco-latest.osm.pbf",
    "navarra": "https://download.geofabrik.de/europe/spain/navarra-latest.osm.pbf",
    "aquitaine": "https://download.geofabrik.de/europe/france/aquitaine-latest.osm.pbf",
}
POSTCODE_PREFIX = {"aquitaine": "64"}        # keep only department 64
TERRITORY = {"01": "Araba", "20": "Gipuzkoa", "48": "Bizkaia",
             "31": "Nafarroa", "32": "Nafarroa", "64": "Iparralde"}
PLACE_TYPES = {"city", "town", "village"}
MAX_KM_TO_TOWN = 10.0


def download(out):
    for name, url in EXTRACTS.items():
        path = out / f"{name}-latest.osm.pbf"
        if path.exists() and path.stat().st_size > 1_000_000:
            continue
        print(f"downloading {path.name}...")
        with requests.get(url, timeout=900, stream=True) as r:
            r.raise_for_status()
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(2 ** 20):
                    f.write(chunk)
        tmp.rename(path)


class Collector(osmium.SimpleHandler):
    """Collects place nodes and address points of one extract."""

    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix
        self.towns = []          # (name, lat, lon)
        self.addresses = []      # (street, postcode, city tag, lat, lon)

    def _postcode(self, tags):
        if "addr:street" not in tags or "addr:postcode" not in tags:
            return None
        postcode = tags["addr:postcode"].strip()
        if len(postcode) != 5 or not postcode.isdigit():
            return None
        if self.prefix and not postcode.startswith(self.prefix):
            return None
        return postcode

    def _add(self, tags, postcode, lat, lon):
        city = tags.get("addr:city") or tags.get("addr:town") or ""
        self.addresses.append((tags["addr:street"], postcode, city, lat, lon))

    def node(self, n):
        tags = n.tags
        try:
            lat, lon = n.location.lat, n.location.lon
        except osmium.InvalidLocation:
            return
        if tags.get("place") in PLACE_TYPES and "name" in tags:
            self.towns.append((tags["name"], lat, lon))
        elif postcode := self._postcode(tags):
            self._add(tags, postcode, lat, lon)

    def way(self, w):
        postcode = self._postcode(w.tags)
        if not postcode:
            return
        points = []
        for node in w.nodes:
            try:
                points.append((node.location.lat, node.location.lon))
            except osmium.InvalidLocation:
                pass
        if points:
            self._add(w.tags, postcode, sum(p[0] for p in points) / len(points),
                      sum(p[1] for p in points) / len(points))


def haversine_km(lat1, lon1, lat2, lon2):
    d_lat, d_lon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def extract(out):
    towns, addresses = [], []
    for name in EXTRACTS:
        path = out / f"{name}-latest.osm.pbf"
        if not path.exists():
            print(f"missing {path.name}; run --download")
            continue
        start = time.time()
        collector = Collector(POSTCODE_PREFIX.get(name))
        collector.apply_file(str(path), locations=True, idx="flex_mem")
        print(f"{path.name}: {len(collector.towns)} towns, "
              f"{len(collector.addresses)} addresses ({time.time() - start:.0f} s)")
        towns += collector.towns
        addresses += collector.addresses

    # 0.1° grid index of place nodes for nearest-town lookup
    def cell(lat, lon):
        return round(lat / 0.1), round(lon / 0.1)

    grid = defaultdict(list)
    town_location = {}
    for name, lat, lon in towns:
        grid[cell(lat, lon)].append((name, lat, lon))
        town_location.setdefault(name, (lat, lon))

    def nearest_town(lat, lon):
        cx, cy = cell(lat, lon)
        best, best_km = None, MAX_KM_TO_TOWN
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for name, t_lat, t_lon in grid.get((cx + dx, cy + dy), []):
                    km = haversine_km(lat, lon, t_lat, t_lon)
                    if km < best_km:
                        best, best_km = name, km
        return best

    groups = defaultdict(lambda: [0, 0.0, 0.0])   # count, Σlat, Σlon
    for street, postcode, city, lat, lon in addresses:
        town = city if city in town_location else nearest_town(lat, lon)
        if town:
            g = groups[(town, street, postcode)]
            g[0] += 1
            g[1] += lat
            g[2] += lon

    streets = []
    town_postcodes = defaultdict(lambda: defaultdict(int))
    town_territories = defaultdict(set)
    for (town, street, postcode), (count, lat, lon) in groups.items():
        territory = TERRITORY.get(postcode[:2])
        if territory is None:
            continue
        streets.append({"street": street, "town": town, "postcode": postcode,
                        "territory": territory, "addresses": count,
                        "lat": round(lat / count, 5), "lon": round(lon / count, 5)})
        town_postcodes[town][postcode] += count
        town_territories[town].add(territory)
    streets.sort(key=lambda s: -s["addresses"])

    town_rows = []
    for town, postcodes in town_postcodes.items():
        if town in town_location:
            lat, lon = town_location[town]
            town_rows.append({"town": town,
                              "territory": "/".join(sorted(town_territories[town])),
                              "postcode": max(postcodes, key=postcodes.get),
                              "lat": round(lat, 5), "lon": round(lon, 5)})
    town_rows.sort(key=lambda t: t["town"])

    for filename, rows in (("streets.csv", streets), ("towns.csv", town_rows)):
        with open(out / filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ {len(rows)} rows → {out / filename}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("data/sources"))
    args = ap.parse_args()
    if not (args.download or args.extract):
        ap.error("pass --download and/or --extract")
    if args.download:
        download(args.out)
    if args.extract:
        extract(args.out)
