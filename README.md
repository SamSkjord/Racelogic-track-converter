# Track Converter

Import racing tracks into a SQLite database with spatial indexing for location-based queries.

Supports two track sources:
- **Custom KMZ tracks** - Google Earth files with track path and start/finish markers
- **RaceLogic tracks** - Official `.CIR` boundary files converted to KMZ

## Requirements

- Python 3.6+
- No external dependencies

## Quick Start

```bash
# Import custom KMZ tracks
cp MyTrack.kmz import/
python track_converter.py

# Import RaceLogic tracks (see setup below)
python racelogic_converter.py
```

## Folder Structure

```
import/                     # Drop files here for import
├── *.kmz                   # Custom tracks
└── Racelogic/              # RaceLogic folder structure

tracks/                     # Output (auto-created)
├── tracks.db               # Custom tracks database
├── racelogic.db            # RaceLogic tracks database
├── maps/*.kmz              # Custom track files
└── racelogic/[country]/    # RaceLogic KMZ by country
```

## Usage

```bash
# Import all KMZ files from import/
python track_converter.py

# List tracks in database
python track_converter.py --list

# Find tracks within 50km of a location
python track_converter.py --nearby 52.0 -1.5 50
```

## RaceLogic Setup

1. Download the **Video Setup & Circuit Tools** installer from:
   https://www.vboxmotorsport.co.uk/index.php/en/customer-ct-track-database

2. Copy the installed `Racelogic` folder to `import/Racelogic/`

3. Run the converter:
   ```bash
   python racelogic_converter.py
   ```

The converter processes 900+ tracks and removes the import folder when complete.

## Database Features

- **Spatial indexing** - Geohash + R-tree for fast location queries
- **Upsert support** - Re-importing a track replaces the existing entry
- **Bounding box** - Each track stores min/max lat/lon for efficient filtering

## Custom KMZ Format

Custom tracks should contain:
- A `LineString` with the track path coordinates
- A `Placemark` named "Start" or "Start / Finish"
- Optionally a `Placemark` named "Finish" (for point-to-point tracks)

Track type is auto-detected based on markers present.
