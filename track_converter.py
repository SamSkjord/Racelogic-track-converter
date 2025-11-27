#!/usr/bin/env python3
"""
Custom Track Converter

Imports KMZ track files from import/ folder:
- Adds to SQLite database (replaces if exists)
- Copies KMZ files to tracks/maps/
- Removes source files after successful import

Input:  import/*.kmz
Output: tracks/tracks.db + tracks/maps/*.kmz
"""

import os
import sys
import shutil

from tracks_db import TracksDB, parse_kmz_track

# Fixed paths relative to script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMPORT_DIR = os.path.join(SCRIPT_DIR, "import")
TRACKS_DIR = os.path.join(SCRIPT_DIR, "tracks")
MAPS_DIR = os.path.join(TRACKS_DIR, "maps")
DB_PATH = os.path.join(TRACKS_DIR, "tracks.db")


# =============================================================================
# Import Functions
# =============================================================================

def import_kmz(kmz_path: str, db: TracksDB) -> bool:
    """
    Import a single KMZ file.
    Returns True if successful.
    """
    filename = os.path.basename(kmz_path)
    print(f"  {filename}")

    try:
        track_data = parse_kmz_track(kmz_path)
        name = track_data['name']

        # Calculate bounding box
        if track_data['path']:
            lats = [p['lat'] for p in track_data['path']]
            lons = [p['long'] for p in track_data['path']]
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
        else:
            min_lat = max_lat = track_data['start_lat']
            min_lon = max_lon = track_data['start_lon']

        # Check if replacing
        existing = db.get_track(name)
        action = "replaced" if existing else "added"

        # Add to database (replace if exists)
        db.add_track(
            name=name,
            track_type=track_data['track_type'],
            start_lat=track_data['start_lat'],
            start_lon=track_data['start_lon'],
            finish_lat=track_data['finish_lat'],
            finish_lon=track_data['finish_lon'],
            min_lat=min_lat, max_lat=max_lat,
            min_lon=min_lon, max_lon=max_lon,
            source_file=filename,
            replace=True
        )

        # Copy KMZ to maps/ folder (overwrite if exists)
        output_kmz = os.path.join(MAPS_DIR, filename)
        shutil.copy2(kmz_path, output_kmz)

        # Remove source file
        os.remove(kmz_path)

        print(f"    -> {action}: {name} ({track_data['track_type']})")
        return True

    except Exception as e:
        print(f"    -> error: {e}")
        return False


def import_all():
    """Import all KMZ files from import/ folder."""
    os.makedirs(MAPS_DIR, exist_ok=True)

    # Find KMZ files
    if not os.path.exists(IMPORT_DIR):
        print("No import/ folder found.")
        return

    kmz_files = [f for f in os.listdir(IMPORT_DIR) if f.lower().endswith('.kmz')]

    if not kmz_files:
        print("No KMZ files in import/")
        return

    print(f"Importing {len(kmz_files)} custom tracks:")

    with TracksDB(DB_PATH) as db:
        success = 0
        for f in kmz_files:
            if import_kmz(os.path.join(IMPORT_DIR, f), db):
                success += 1

        print(f"\nImported: {success}/{len(kmz_files)}")
        print(f"Database: {DB_PATH}")


# =============================================================================
# CLI
# =============================================================================

def show_status():
    """Show current tracks in database."""
    if not os.path.exists(DB_PATH):
        print("No custom tracks database yet.")
        return

    with TracksDB(DB_PATH) as db:
        tracks = db.list_tracks()
        print(f"Custom tracks: {len(tracks)}")
        for t in tracks:
            print(f"  {t['name']} ({t['track_type']})")


def find_nearby(lat: float, lon: float, radius: float = 50):
    """Find tracks near a location."""
    if not os.path.exists(DB_PATH):
        print("No custom tracks database yet.")
        return

    with TracksDB(DB_PATH) as db:
        tracks = db.find_nearby(lat, lon, radius)
        print(f"Tracks within {radius}km of ({lat}, {lon}):")
        for t in tracks:
            print(f"  {t['distance_km']:5.1f}km - {t['name']}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default: import all
        import_all()
    elif sys.argv[1] == "--list":
        show_status()
    elif sys.argv[1] == "--nearby" and len(sys.argv) >= 4:
        lat, lon = float(sys.argv[2]), float(sys.argv[3])
        radius = float(sys.argv[4]) if len(sys.argv) > 4 else 50
        find_nearby(lat, lon, radius)
    else:
        print("Usage:")
        print("  python track_converter.py           Import all KMZ from import/")
        print("  python track_converter.py --list    List tracks in database")
        print("  python track_converter.py --nearby LAT LON [RADIUS]")
