# Track Converter - AI Context Document

## Project Overview

Two converters for importing track data into a SQLite database with spatial indexing:

1. **`track_converter.py`** - Imports custom KMZ track files
2. **`racelogic_converter.py`** - Converts RaceLogic .CIR boundary files to KMZ

Both converters:
- Add track metadata to SQLite databases with geohash + R-tree spatial indexing
- Support upsert (replace existing tracks with same name)
- Remove source files after successful import

## Folder Structure

```
import/                     # Drop files here for import
├── *.kmz                   # Custom tracks (track_converter.py)
└── Racelogic/              # RaceLogic folder (racelogic_converter.py)
    ├── CIR Files/[country]/[track].cir
    ├── Start Finish Database/StartFinishDataBase.xml
    └── DataBaseTrackmapFiles/[country]/TRACKMAPS/[track].CFG

tracks/                     # Output directory
├── tracks.db               # Custom tracks database
├── racelogic.db            # RaceLogic tracks database
├── maps/                   # Custom track KMZ files
│   └── *.kmz
└── racelogic/              # RaceLogic KMZ files by country
    └── [country]/*.kmz
```

## Running

```bash
# Import custom KMZ tracks from import/
python track_converter.py

# Import RaceLogic tracks from import/Racelogic/
python racelogic_converter.py

# List custom tracks in database
python track_converter.py --list

# Find tracks within 50km of a location
python track_converter.py --nearby 52.0 -1.5 50
```

## Database Schema (tracks_db.py)

SQLite with spatial indexing for fast location queries:

```sql
tracks (
    id, name, track_type,           -- 'loop' or 'point_to_point'
    start_lat, start_lon,           -- Start/finish line
    finish_lat, finish_lon,         -- For point-to-point only
    min_lat, max_lat, min_lon, max_lon,  -- Bounding box
    geohash,                        -- 6-char geohash for prefix filtering
    source_file, created_at
)

tracks_rtree (...)                  -- R-tree spatial index on bounding box
```

### Spatial Query Strategy

Two-stage filtering for `find_nearby(lat, lon, radius_km)`:
1. **Geohash prefix** - Fast elimination of distant tracks
2. **Haversine distance** - Accurate filtering of remaining candidates

## Custom Track KMZ Format

Custom tracks are Google Earth KMZ files containing:
- **Path**: LineString with track coordinates
- **Start marker**: Placemark named "Start" or "Start / Finish"
- **Finish marker**: Placemark named "Finish" (point-to-point only)

Track type auto-detected:
- `loop` - Single Start/Finish marker
- `point_to_point` - Separate Start and Finish markers

## RaceLogic Converter Details

### Problem Solved

RaceLogic .CIR files contain GPS boundary data in a concatenated format:
- Two track boundaries (inner and outer) stored as a single continuous path
- Path traces one boundary, crosses to the other side, traces back
- Creates visual artifacts when rendered directly

### Boundary Processing Pipeline

```
Raw CIR data (concatenated boundaries)
    ↓
detect_and_split_boundaries()  → Split at point returning to start
    ↓
remove_crossing_artifacts()    → Trim sharp turns (>50°) at split region
    ↓
close_boundary_loop()          → Connect end to start
    ↓
generate_kml()                 → Output with inner/outer boundaries + S/F markers
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `parse_track_data_file()` | Parse .CIR files, handle hemisphere detection |
| `convert_to_decimal_degrees()` | Convert RaceLogic minutes ÷ 60 |
| `detect_and_split_boundaries()` | Find where path returns to start |
| `remove_crossing_artifacts()` | Detect sharp turns (>50°), trim crossing points |
| `haversine_distance()` | Great-circle distance between GPS points |

### Track Filtering

Combo tracks (multiple layouts overlaid) are skipped:
- `combo="true"` attribute in database
- `length="0"` in database
- "Combo" in filename

### Output KML Structure

- **Outer Boundary** (red) - First traced path
- **Inner Boundary** (blue) - Second traced path
- **Start/Finish Line & Marker** (green)

## Test Tracks

| Track | Country | Points | Notes |
|-------|---------|--------|-------|
| Croft | UK | 631 | Clean dual boundary |
| Castle Combe | UK | 3,942 | Shallow crossing angle |
| Bathurst | Australia | 13,950 | Opposite direction tracing |
| Goodwood FoS | UK | 245 | Hillclimb (single boundary) |

## Debug

RaceLogic operations logged to `racelogic_debug.log`
