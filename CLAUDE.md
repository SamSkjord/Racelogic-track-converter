# RaceLogic Track Converter - AI Context Document

## Project Overview

Converts RaceLogic `.CIR` track boundary files into:
1. **Google Earth `.KML/.KMZ` files** - with boundaries and S/F markers
2. **Track map PNG images** - transparent background, aqua track fill, S/F line

## Problem Solved

RaceLogic track files contain GPS boundary data in a concatenated format:
- Two track boundaries (inner and outer) stored as a single continuous path
- Path traces one boundary, crosses to the other side, traces back
- Creates visual artifacts when rendered directly

The converter:
1. Detects and splits concatenated boundaries
2. Removes crossing artifacts (sharp turns where path jumps between boundaries)
3. Closes boundary loops after cleanup
4. Generates centerline by averaging corresponding points from both boundaries
5. Auto-detects and corrects boundaries traced in opposite directions

## Source Data Location

```
../Racelogic-tracks/
├── CIR Files/[country]/[track].cir    # GPS boundary data
├── Start Finish Database/
│   └── StartFinishDataBase.xml        # S/F coordinates
└── DataBaseTrackmapFiles/[country]/TRACKMAPS/
    └── [track].CFG                    # Hemisphere info
```

## Key Functions in `racelogic_converter.py`

### Data Parsing
- `parse_track_data_file()` - Parses .CIR files, handles hemisphere detection
- `convert_to_decimal_degrees()` - Converts RaceLogic minutes ÷ 60

### Boundary Detection & Cleanup
- `detect_and_split_boundaries()` - Finds where path returns to start point
- `remove_crossing_artifacts()` - Detects sharp turns (>50°) and trims crossing points
- `close_boundary_loop()` - Connects boundary end to start after trimming
- `haversine_distance()` - Great-circle distance between GPS points

### Centerline Generation
- `resample_boundary(boundary, num_points)` - Equal-distance point interpolation
- `generate_centerline(boundary1, boundary2)` - Averages two boundaries
  - Auto-detects opposite direction tracing and reverses if needed
  - Interpolates across points where boundaries converge (crossing artifacts)

### KML Output
- `generate_kml()` - Creates KML with boundaries, centerline, S/F markers
- `create_kmz()` - Packages KML into compressed KMZ

## Boundary Processing Pipeline

```
Raw CIR data (concatenated boundaries)
    ↓
detect_and_split_boundaries()  → Split at point returning to start
    ↓
remove_crossing_artifacts()    → Trim sharp turns (>50°) at split region
    ↓
close_boundary_loop()          → Connect end to start
    ↓
generate_centerline()          → Resample, align direction, average points
```

## Test Tracks

| Track | Country | Points | Notes |
|-------|---------|--------|-------|
| Croft | UK | 631 | Clean dual boundary |
| Castle Combe | UK | 3,942 | Shallow crossing angle (59.8°) |
| Bathurst | Australia | 13,950 | Opposite direction tracing |
| Oran Park Raceway | Australia | 771 | Over-under section |
| Goodwood FoS | UK | 245 | Hillclimb (single boundary) |

## Output KML Structure

- **Outer Boundary** (red) - First traced path
- **Inner Boundary** (blue) - Second traced path
- **Centerline** (cyan, 300 points) - For lap timing/delta
- **Start/Finish Line & Marker** (green)

## Running

```bash
# Full batch
python racelogic_converter.py /path/to/Racelogic /path/to/output

# Single track test
python3 -c "
from racelogic_converter import *
root_dir = '../Racelogic-tracks'
xml_db = find_database_file(root_dir)
track_info = get_track_info_from_database('Croft', xml_db)
parsed_data = parse_track_data_file('path/to/Croft.cir', xml_db, 'Croft', None)
kml = generate_kml(parsed_data, track_info)
"
```

## Track Filtering

### Combo Tracks (Skipped)
Tracks with multiple layouts overlaid are skipped. Detected via:
- `combo="true"` attribute in database
- `length="0"` in database
- "Combo" in filename

Use individual layout files instead (e.g., "Oulton Park International" not "Oulton Park Combo").

## Track Map PNG Generation (`generate_trackmap.py`)

Generates transparent PNG images of track outlines.

### Approach
- Draw thick centerline first (30px) - fills gaps between boundaries
- Draw thinner boundary lines on top (8px) - provides definition
- Centerline calculated by finding closest inner point for each outer point

### Usage
```python
from generate_trackmap import generate_track_image
generate_track_image("Croft", "/path/to/Racelogic-tracks", "output.png")
```

### Output
- 800x800 transparent PNG
- Aqua fill (RGBA: 0, 255, 255, 180)
- White S/F line
- Handles over-under sections (e.g., Oran Park)
- Single boundary tracks (hillclimbs) drawn as simple line

## Debug

All operations logged to `racelogic_debug.log`
