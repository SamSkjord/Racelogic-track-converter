#!/usr/bin/env python3
"""
Tracks database module.
SQLite-based storage with spatial indexing for efficient location queries.
Supports both custom KMZ tracks and RaceLogic XML database imports.
"""

import sqlite3
import os
import math
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple


# =============================================================================
# Geohash Implementation
# =============================================================================

GEOHASH_BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'

def encode_geohash(lat: float, lon: float, precision: int = 6) -> str:
    """
    Encode latitude/longitude to geohash string.

    Precision guide (approximate cell size):
        4 chars: ~40km x 20km  - good for regional filtering
        5 chars: ~5km x 5km    - good for city-level
        6 chars: ~1.2km x 0.6km - good for neighborhood
        7 chars: ~150m x 150m
    """
    lat_range = (-90.0, 90.0)
    lon_range = (-180.0, 180.0)

    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    is_lon = True

    while len(geohash) < precision:
        if is_lon:
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon >= mid:
                ch |= bits[bit]
                lon_range = (mid, lon_range[1])
            else:
                lon_range = (lon_range[0], mid)
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat >= mid:
                ch |= bits[bit]
                lat_range = (mid, lat_range[1])
            else:
                lat_range = (lat_range[0], mid)

        is_lon = not is_lon

        if bit < 4:
            bit += 1
        else:
            geohash.append(GEOHASH_BASE32[ch])
            bit = 0
            ch = 0

    return ''.join(geohash)


def decode_geohash_bounds(geohash: str) -> Tuple[float, float, float, float]:
    """Decode geohash to bounding box (min_lat, max_lat, min_lon, max_lon)."""
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]

    is_lon = True

    for char in geohash.lower():
        idx = GEOHASH_BASE32.index(char)
        for bit in [16, 8, 4, 2, 1]:
            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2
                if idx & bit:
                    lon_range[0] = mid
                else:
                    lon_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if idx & bit:
                    lat_range[0] = mid
                else:
                    lat_range[1] = mid
            is_lon = not is_lon

    return (lat_range[0], lat_range[1], lon_range[0], lon_range[1])


def geohash_neighbors(geohash: str) -> List[str]:
    """Get the 8 neighboring geohash cells (plus self = 9 cells)."""
    min_lat, max_lat, min_lon, max_lon = decode_geohash_bounds(geohash)

    lat_delta = (max_lat - min_lat) / 2
    lon_delta = (max_lon - min_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    precision = len(geohash)
    neighbors = []

    for dlat in [-1, 0, 1]:
        for dlon in [-1, 0, 1]:
            nlat = center_lat + dlat * lat_delta * 2
            nlon = center_lon + dlon * lon_delta * 2
            # Handle wraparound
            if -90 <= nlat <= 90 and -180 <= nlon <= 180:
                neighbors.append(encode_geohash(nlat, nlon, precision))

    return list(set(neighbors))


# =============================================================================
# Distance Calculation
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in meters between two points."""
    R = 6371000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))


def racelogic_to_decimal(minutes_val: float) -> float:
    """Convert RaceLogic minutes format to decimal degrees."""
    return minutes_val / 60.0


# =============================================================================
# Database Schema & Management
# =============================================================================

SCHEMA = """
-- Main tracks table
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    track_type TEXT NOT NULL CHECK(track_type IN ('loop', 'point_to_point')),

    -- Start line (decimal degrees)
    start_lat REAL NOT NULL,
    start_lon REAL NOT NULL,

    -- Finish line (same as start for loops)
    finish_lat REAL,
    finish_lon REAL,

    -- Track center point (for distance calculations)
    center_lat REAL NOT NULL,
    center_lon REAL NOT NULL,

    -- Bounding box (for R-tree queries)
    min_lat REAL NOT NULL,
    max_lat REAL NOT NULL,
    min_lon REAL NOT NULL,
    max_lon REAL NOT NULL,

    -- Geohash of center point (for prefix filtering)
    -- 4 chars = ~40km cell, 5 chars = ~5km cell
    geohash_4 TEXT NOT NULL,
    geohash_5 TEXT NOT NULL,
    geohash_6 TEXT NOT NULL,

    -- Metadata
    length_meters INTEGER,
    country TEXT,
    region TEXT,
    source_file TEXT,
    gate_width REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- R-tree spatial index for bounding box queries
CREATE VIRTUAL TABLE IF NOT EXISTS tracks_rtree USING rtree(
    id,
    min_lon, max_lon,
    min_lat, max_lat
);

-- Index on geohash for prefix queries
CREATE INDEX IF NOT EXISTS idx_geohash_4 ON tracks(geohash_4);
CREATE INDEX IF NOT EXISTS idx_geohash_5 ON tracks(geohash_5);
CREATE INDEX IF NOT EXISTS idx_geohash_6 ON tracks(geohash_6);

-- Index on country/region for filtering
CREATE INDEX IF NOT EXISTS idx_country ON tracks(country);
CREATE INDEX IF NOT EXISTS idx_region ON tracks(region);
"""


class TracksDB:
    """Tracks database with spatial querying."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Connect to database and initialize schema."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def add_track(self,
                  name: str,
                  track_type: str,
                  start_lat: float, start_lon: float,
                  finish_lat: float = None, finish_lon: float = None,
                  min_lat: float = None, max_lat: float = None,
                  min_lon: float = None, max_lon: float = None,
                  length_meters: int = None,
                  country: str = None,
                  region: str = None,
                  source_file: str = None,
                  gate_width: float = None,
                  replace: bool = False) -> int:
        """
        Add a track to the database.

        Args:
            name: Track name
            track_type: 'loop' or 'point_to_point'
            start_lat, start_lon: Start line coordinates (decimal degrees)
            finish_lat, finish_lon: Finish line (optional, defaults to start for loops)
            min/max_lat/lon: Bounding box (optional, calculated from start/finish if not provided)
            length_meters: Track length in meters
            country, region: Location metadata
            source_file: Original file path
            gate_width: Gate width in meters
            replace: If True, replace existing track with same name

        Returns:
            Track ID
        """
        # If replace=True, delete existing track first
        if replace:
            self.delete_track(name)
        # Default finish to start for loops
        if finish_lat is None:
            finish_lat = start_lat
        if finish_lon is None:
            finish_lon = start_lon

        # Calculate bounding box from start/finish if not provided
        if min_lat is None:
            min_lat = min(start_lat, finish_lat)
        if max_lat is None:
            max_lat = max(start_lat, finish_lat)
        if min_lon is None:
            min_lon = min(start_lon, finish_lon)
        if max_lon is None:
            max_lon = max(start_lon, finish_lon)

        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

        # Generate geohashes at multiple precisions
        geohash_4 = encode_geohash(center_lat, center_lon, 4)
        geohash_5 = encode_geohash(center_lat, center_lon, 5)
        geohash_6 = encode_geohash(center_lat, center_lon, 6)

        cursor = self.conn.cursor()

        # Insert into main table
        cursor.execute("""
            INSERT INTO tracks (
                name, track_type,
                start_lat, start_lon, finish_lat, finish_lon,
                center_lat, center_lon,
                min_lat, max_lat, min_lon, max_lon,
                geohash_4, geohash_5, geohash_6,
                length_meters, country, region, source_file, gate_width
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, track_type,
            start_lat, start_lon, finish_lat, finish_lon,
            center_lat, center_lon,
            min_lat, max_lat, min_lon, max_lon,
            geohash_4, geohash_5, geohash_6,
            length_meters, country, region, source_file, gate_width
        ))

        track_id = cursor.lastrowid

        # Insert into R-tree
        cursor.execute("""
            INSERT INTO tracks_rtree (id, min_lon, max_lon, min_lat, max_lat)
            VALUES (?, ?, ?, ?, ?)
        """, (track_id, min_lon, max_lon, min_lat, max_lat))

        self.conn.commit()
        return track_id

    def find_nearby(self,
                    lat: float, lon: float,
                    radius_km: float = 50,
                    limit: int = 20) -> List[Dict]:
        """
        Find tracks within radius_km of a point, sorted by distance.

        Uses a two-stage filter:
        1. Geohash prefix for coarse filtering (fast)
        2. Haversine distance for accurate filtering (slower)

        Args:
            lat, lon: Search center (decimal degrees)
            radius_km: Search radius in kilometers
            limit: Maximum results to return

        Returns:
            List of tracks with distance_km field, sorted by distance
        """
        radius_m = radius_km * 1000

        # Choose geohash precision based on search radius
        if radius_km >= 30:
            precision = 4
            geohash_col = 'geohash_4'
        elif radius_km >= 5:
            precision = 5
            geohash_col = 'geohash_5'
        else:
            precision = 6
            geohash_col = 'geohash_6'

        # Get geohash of search center and neighbors
        center_geohash = encode_geohash(lat, lon, precision)
        search_hashes = geohash_neighbors(center_geohash)

        # Query with geohash filter
        placeholders = ','.join(['?' for _ in search_hashes])
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT * FROM tracks
            WHERE {geohash_col} IN ({placeholders})
        """, search_hashes)

        # Calculate actual distances and filter
        results = []
        for row in cursor.fetchall():
            dist = haversine_distance(lat, lon, row['center_lat'], row['center_lon'])
            if dist <= radius_m:
                track = dict(row)
                track['distance_km'] = dist / 1000
                results.append(track)

        # Sort by distance and limit
        results.sort(key=lambda x: x['distance_km'])
        return results[:limit]

    def find_in_bbox(self,
                     min_lat: float, max_lat: float,
                     min_lon: float, max_lon: float) -> List[Dict]:
        """
        Find tracks whose bounding box intersects the given box.
        Uses R-tree for efficient spatial query.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT t.* FROM tracks t
            JOIN tracks_rtree r ON t.id = r.id
            WHERE r.max_lon >= ? AND r.min_lon <= ?
              AND r.max_lat >= ? AND r.min_lat <= ?
        """, (min_lon, max_lon, min_lat, max_lat))

        return [dict(row) for row in cursor.fetchall()]

    def get_track(self, name: str) -> Optional[Dict]:
        """Get a track by name."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tracks WHERE name = ?", (name,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_tracks(self, country: str = None, region: str = None) -> List[Dict]:
        """List all tracks, optionally filtered by country/region."""
        cursor = self.conn.cursor()

        if country and region:
            cursor.execute("SELECT * FROM tracks WHERE country = ? AND region = ?",
                          (country, region))
        elif country:
            cursor.execute("SELECT * FROM tracks WHERE country = ?", (country,))
        elif region:
            cursor.execute("SELECT * FROM tracks WHERE region = ?", (region,))
        else:
            cursor.execute("SELECT * FROM tracks")

        return [dict(row) for row in cursor.fetchall()]

    def count_tracks(self, country: str = None) -> int:
        """Count tracks, optionally by country."""
        cursor = self.conn.cursor()
        if country:
            cursor.execute("SELECT COUNT(*) FROM tracks WHERE country = ?", (country,))
        else:
            cursor.execute("SELECT COUNT(*) FROM tracks")
        return cursor.fetchone()[0]

    def list_countries(self) -> List[str]:
        """List all unique countries."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT country FROM tracks WHERE country IS NOT NULL ORDER BY country")
        return [row[0] for row in cursor.fetchall()]

    def delete_track(self, name: str) -> bool:
        """Delete a track by name."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM tracks WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            return False

        track_id = row['id']
        cursor.execute("DELETE FROM tracks_rtree WHERE id = ?", (track_id,))
        cursor.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
        self.conn.commit()
        return True


# =============================================================================
# RaceLogic XML Import
# =============================================================================

def import_racelogic_xml(db: TracksDB, xml_path: str, skip_combo: bool = True,
                         replace: bool = False) -> Dict:
    """
    Import tracks from RaceLogic StartFinishDataBase.xml.

    Args:
        db: TracksDB instance
        xml_path: Path to StartFinishDataBase.xml
        skip_combo: Skip combo tracks (length=0, combo=true)
        replace: If True, replace existing tracks with same name

    Returns:
        Dict with import statistics
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    stats = {
        'imported': 0,
        'replaced': 0,
        'skipped_combo': 0,
        'errors': 0,
        'by_country': {}
    }

    for country_elem in root.findall('.//country'):
        country_name = country_elem.get('name')

        for circuit in country_elem.findall('.//circuit'):
            name = circuit.get('name')

            # Skip combo tracks
            is_combo = (
                circuit.get('combo') == 'true' or
                circuit.get('length') == '0' or
                'Combo' in name
            )
            if skip_combo and is_combo:
                stats['skipped_combo'] += 1
                continue

            try:
                # Parse bounding box (in minutes format)
                min_coords = circuit.get('min', '0,0').split(',')
                max_coords = circuit.get('max', '0,0').split(',')

                min_lon = racelogic_to_decimal(float(min_coords[0]))
                min_lat = racelogic_to_decimal(float(min_coords[1]))
                max_lon = racelogic_to_decimal(float(max_coords[0]))
                max_lat = racelogic_to_decimal(float(max_coords[1]))

                # Parse S/F coordinates
                start_finish = circuit.find('.//startFinish')
                finish_elem = circuit.find('.//Finish')

                if start_finish is None:
                    stats['errors'] += 1
                    continue

                start_lon = racelogic_to_decimal(float(start_finish.get('long', 0)))
                start_lat = racelogic_to_decimal(float(start_finish.get('lat', 0)))

                # Determine track type
                if finish_elem is not None:
                    track_type = 'point_to_point'
                    finish_lon = racelogic_to_decimal(float(finish_elem.get('long', 0)))
                    finish_lat = racelogic_to_decimal(float(finish_elem.get('lat', 0)))
                else:
                    track_type = 'loop'
                    finish_lon = start_lon
                    finish_lat = start_lat

                # Parse other attributes
                length = circuit.get('length')
                length_meters = int(length) if length and length != '0' else None

                gate_width = circuit.get('gatewidth')
                gate_width = float(gate_width) if gate_width else None

                # Check if exists (for stats tracking)
                existing = db.get_track(name)

                # Add to database (will replace if exists and replace=True)
                db.add_track(
                    name=name,
                    track_type=track_type,
                    start_lat=start_lat,
                    start_lon=start_lon,
                    finish_lat=finish_lat,
                    finish_lon=finish_lon,
                    min_lat=min_lat,
                    max_lat=max_lat,
                    min_lon=min_lon,
                    max_lon=max_lon,
                    length_meters=length_meters,
                    country=country_name,
                    gate_width=gate_width,
                    source_file=os.path.basename(xml_path),
                    replace=replace
                )

                if existing and replace:
                    stats['replaced'] += 1
                else:
                    stats['imported'] += 1
                stats['by_country'][country_name] = stats['by_country'].get(country_name, 0) + 1

            except sqlite3.IntegrityError:
                # Only happens if replace=False and track exists
                stats['errors'] += 1
            except Exception as e:
                print(f"  Error importing {name}: {e}")
                stats['errors'] += 1

    return stats


# =============================================================================
# KMZ Import
# =============================================================================

def parse_kmz_track(kmz_path: str) -> Dict:
    """Extract track data from a KMZ file."""
    with zipfile.ZipFile(kmz_path, 'r') as z:
        kml_content = z.read('doc.kml').decode('utf-8')

    root = ET.fromstring(kml_content)
    ns = {
        'kml': 'http://www.opengis.net/kml/2.2',
        'gx': 'http://www.google.com/kml/ext/2.2'
    }

    track_data = {
        'name': None,
        'path': [],
        'start_lat': None, 'start_lon': None,
        'finish_lat': None, 'finish_lon': None,
        'track_type': 'loop'
    }

    # Get track name
    doc_name = root.find('.//kml:Document/kml:name', ns)
    if doc_name is not None:
        track_data['name'] = doc_name.text.replace('.kmz', '').strip()

    # Find Placemarks
    for placemark in root.findall('.//kml:Placemark', ns):
        name_elem = placemark.find('kml:name', ns)
        name = name_elem.text if name_elem is not None else ''

        # Point markers
        point = placemark.find('.//kml:Point/kml:coordinates', ns)
        if point is not None:
            coords = point.text.strip().split(',')
            lon, lat = float(coords[0]), float(coords[1])

            if 'Start / Finish' in name:
                track_data['start_lat'] = lat
                track_data['start_lon'] = lon
                track_data['finish_lat'] = lat
                track_data['finish_lon'] = lon
            elif 'Start' in name:
                track_data['start_lat'] = lat
                track_data['start_lon'] = lon
            elif 'Finish' in name:
                track_data['finish_lat'] = lat
                track_data['finish_lon'] = lon
                track_data['track_type'] = 'point_to_point'

        # LineString (track path)
        linestring = placemark.find('.//kml:LineString/kml:coordinates', ns)
        if linestring is not None:
            for coord in linestring.text.strip().split():
                parts = coord.strip().split(',')
                if len(parts) >= 2:
                    lon, lat = float(parts[0]), float(parts[1])
                    track_data['path'].append({'lat': lat, 'long': lon})

    return track_data


def import_kmz_to_db(db: TracksDB, kmz_path: str,
                     country: str = None, region: str = None) -> int:
    """Import a KMZ file into the database."""
    track_data = parse_kmz_track(kmz_path)

    # Calculate bounding box from path
    if track_data['path']:
        lats = [p['lat'] for p in track_data['path']]
        lons = [p['long'] for p in track_data['path']]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
    else:
        min_lat = max_lat = track_data['start_lat']
        min_lon = max_lon = track_data['start_lon']

    return db.add_track(
        name=track_data['name'],
        track_type=track_data['track_type'],
        start_lat=track_data['start_lat'],
        start_lon=track_data['start_lon'],
        finish_lat=track_data['finish_lat'],
        finish_lon=track_data['finish_lon'],
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        country=country,
        region=region,
        source_file=os.path.basename(kmz_path)
    )


# =============================================================================
# CLI
# =============================================================================

def print_usage():
    print("""
Tracks Database CLI

Usage:
  python tracks_db.py <database> <command> [args...]

Commands:
  import-racelogic <xml_file>     Import RaceLogic StartFinishDataBase.xml
  import-kmz <kmz_file> [country] Import a KMZ file
  nearby <lat> <lon> [radius_km]  Find tracks near a location
  list [country]                  List tracks (optionally by country)
  countries                       List all countries
  info <track_name>               Show track details
  stats                           Show database statistics

Examples:
  python tracks_db.py tracks/racelogic.db import-racelogic ../Racelogic-tracks/Start\\ Finish\\ Database/StartFinishDataBase.xml
  python tracks_db.py tracks/racelogic.db nearby 51.5 -1.5 50
  python tracks_db.py tracks/tracks.db import-kmz track.kmz UK
""")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)

    db_path = sys.argv[1]
    cmd = sys.argv[2]

    with TracksDB(db_path) as db:
        if cmd == "import-racelogic" and len(sys.argv) > 3:
            xml_path = sys.argv[3]
            print(f"Importing from: {xml_path}")
            stats = import_racelogic_xml(db, xml_path)
            print(f"\n=== Import Complete ===")
            print(f"Imported: {stats['imported']}")
            print(f"Skipped (combo): {stats['skipped_combo']}")
            print(f"Skipped (duplicate): {stats['skipped_duplicate']}")
            print(f"Errors: {stats['errors']}")
            print(f"\nBy country (top 10):")
            sorted_countries = sorted(stats['by_country'].items(), key=lambda x: -x[1])[:10]
            for country, count in sorted_countries:
                print(f"  {country}: {count}")

        elif cmd == "import-kmz" and len(sys.argv) > 3:
            kmz_path = sys.argv[3]
            country = sys.argv[4] if len(sys.argv) > 4 else None
            region = sys.argv[5] if len(sys.argv) > 5 else None
            track_id = import_kmz_to_db(db, kmz_path, country, region)
            print(f"Imported track ID: {track_id}")

        elif cmd == "nearby" and len(sys.argv) >= 5:
            lat = float(sys.argv[3])
            lon = float(sys.argv[4])
            radius = float(sys.argv[5]) if len(sys.argv) > 5 else 50

            tracks = db.find_nearby(lat, lon, radius)
            print(f"Found {len(tracks)} tracks within {radius}km of ({lat}, {lon}):")
            for t in tracks:
                length_str = f", {t['length_meters']}m" if t['length_meters'] else ""
                print(f"  {t['distance_km']:5.1f}km - {t['name']} ({t['country']}{length_str})")

        elif cmd == "list":
            country = sys.argv[3] if len(sys.argv) > 3 else None
            tracks = db.list_tracks(country=country)
            print(f"Total tracks: {len(tracks)}")
            for t in tracks[:50]:  # Limit output
                print(f"  {t['name']} - {t['track_type']} @ ({t['center_lat']:.4f}, {t['center_lon']:.4f})")
            if len(tracks) > 50:
                print(f"  ... and {len(tracks) - 50} more")

        elif cmd == "countries":
            countries = db.list_countries()
            print(f"Countries ({len(countries)}):")
            for country in countries:
                count = db.count_tracks(country)
                print(f"  {country}: {count} tracks")

        elif cmd == "info" and len(sys.argv) > 3:
            name = ' '.join(sys.argv[3:])
            track = db.get_track(name)
            if track:
                print(f"Track: {track['name']}")
                print(f"  Type: {track['track_type']}")
                print(f"  Country: {track['country']}")
                print(f"  Length: {track['length_meters']}m" if track['length_meters'] else "  Length: unknown")
                print(f"  Start: ({track['start_lat']:.6f}, {track['start_lon']:.6f})")
                if track['track_type'] == 'point_to_point':
                    print(f"  Finish: ({track['finish_lat']:.6f}, {track['finish_lon']:.6f})")
                print(f"  Center: ({track['center_lat']:.6f}, {track['center_lon']:.6f})")
                print(f"  Geohash: {track['geohash_4']} / {track['geohash_5']} / {track['geohash_6']}")
            else:
                print(f"Track not found: {name}")

        elif cmd == "stats":
            total = db.count_tracks()
            countries = db.list_countries()
            print(f"Database: {db_path}")
            print(f"Total tracks: {total}")
            print(f"Countries: {len(countries)}")

        else:
            print_usage()
