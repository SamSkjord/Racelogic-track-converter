"""
RaceLogic CIR file parsing and coordinate handling.

Handles:
- CIR file format parsing
- RaceLogic minutes to decimal degrees conversion
- Hemisphere detection from CFG files and database
- Track info extraction from XML database
"""

import os
import re
import xml.etree.ElementTree as ET


def log_message(message):
    """Logging stub - replaced at runtime by main module."""
    pass


def set_logger(logger_func):
    """Set the logging function to use."""
    global log_message
    log_message = logger_func


# =============================================================================
# Coordinate Conversion
# =============================================================================

def convert_to_decimal_degrees(racelogic_minutes):
    """Convert RaceLogic minutes to decimal degrees."""
    return racelogic_minutes / 60.0


def parse_config_file(cfg_path):
    """Parse a .CFG file to get base coordinates and other parameters."""
    config = {}
    try:
        with open(cfg_path, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()
        log_message(f"Parsed config file: {cfg_path}, keys: {list(config.keys())}")
    except Exception as e:
        log_message(f"Warning: Error parsing config file {cfg_path}: {e}")
    return config


def get_cfg_hemisphere(cfg_file):
    """Get hemisphere from CFG file based on base_longitude sign."""
    if not cfg_file:
        return None

    cfg_data = parse_config_file(cfg_file)
    if "base_longitude" in cfg_data:
        base_long_cfg = cfg_data["base_longitude"]
        is_eastern = base_long_cfg.startswith("+")
        log_message(f"  CFG hemisphere: {'Eastern' if is_eastern else 'Western'}, base_long={base_long_cfg}")
        return is_eastern  # True for Eastern, False for Western

    return None


def determine_hemisphere(track_name, xml_db, base_lat, base_long, cfg_file=None):
    """Determine if a location is in Western or Eastern hemisphere."""
    log_message(f"Determining hemisphere for track: {track_name}")

    # First priority: Check the CFG file if available - most reliable source
    if cfg_file:
        cfg_data = parse_config_file(cfg_file)
        if "base_longitude" in cfg_data:
            base_long_cfg = cfg_data["base_longitude"]
            is_eastern = base_long_cfg.startswith("+")
            log_message(f"  Determined from CFG file: {'Eastern' if is_eastern else 'Western'}, base_long={base_long_cfg}")
            return not is_eastern  # Return True for Western, False for Eastern

    # Second priority: Check database
    if xml_db is not None:
        for country in xml_db.findall(".//country"):
            for circuit in country.findall(".//circuit"):
                if circuit.get("name").strip().lower() == track_name.strip().lower():
                    if "min" in circuit.attrib:
                        min_coords = circuit.get("min").split(",")
                        if len(min_coords) == 2:
                            min_long = float(min_coords[0])
                            result = min_long < 0
                            log_message(f"  Determined from DB: {result}, min_long={min_long}")
                            return result

    # Third priority: Use common geographic knowledge
    if base_lat > 0 and base_long > 1000:  # North America
        log_message(f"  Determined as North America (West): base_lat={base_lat}, base_long={base_long}")
        return True

    if base_long > 300 and base_long < 1000:  # Asia/Australia
        log_message(f"  Determined as Asia/Australia (East): base_long={base_long}")
        return False

    if base_lat > 2800 and base_lat < 3800:  # Europe/UK
        log_message(f"  Determined as Europe (West): base_lat={base_lat}")
        return True

    result = base_long < 0
    log_message(f"  Using default sign: {result}, base_long={base_long}")
    return result


# =============================================================================
# Track Data File Discovery
# =============================================================================

def find_track_data_file(track_name, root_dir):
    """Find the corresponding data file for a track."""
    clean_track_name = re.sub(r"[^a-zA-Z0-9]", " ", track_name).strip()
    clean_track_name = re.sub(r"\s+", " ", clean_track_name)

    log_message(f"Looking for track data file for: {track_name} (cleaned: {clean_track_name})")

    track_variations = [
        track_name,
        clean_track_name,
        clean_track_name.replace(" ", ""),
        re.sub(r"[^a-zA-Z0-9]", "", track_name),
    ]

    if "International" in track_name:
        track_variations.append(track_name.replace("International", "Int"))
    if "Circuit" in track_name:
        track_variations.append(track_name.replace("Circuit", ""))
    if "Raceway" in track_name:
        track_variations.append(track_name.replace("Raceway", ""))

    for track_var in set(track_variations):
        for ext in [".txt", ".csv", ".vbo"]:
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.lower() == f"{track_var.lower()}{ext}":
                        log_message(f"Found exact match: {os.path.join(root, file)}")
                        return os.path.join(root, file)

    # Look for .CIR files
    cir_files_dir = os.path.join(root_dir, "CIR Files")
    if os.path.exists(cir_files_dir):
        log_message(f"Checking CIR files directory: {cir_files_dir}")
        for root, dirs, files in os.walk(cir_files_dir):
            for file in files:
                if file.lower() == f"{track_name.lower()}.cir":
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", errors="ignore") as f:
                            content = f.read()
                            if "[data]" in content:
                                log_message(f"Found CIR file with data: {file_path}")
                                return file_path
                    except Exception as e:
                        log_message(f"Error reading CIR file: {e}")

    log_message(f"No track data file found for: {track_name}")
    return None


# =============================================================================
# CIR File Parsing
# =============================================================================

def parse_track_data_file(data_path, xml_db, track_name, cfg_file=None):
    """
    Parse a track data file in RaceLogic CIR format.

    CIR files contain GPS boundary points in RaceLogic's "minutes" format
    (degrees * 60). The file has sections like [column names] and [data].

    Args:
        data_path: Path to .CIR file
        xml_db: Parsed XML database root (for hemisphere detection fallback)
        track_name: Track name (for hemisphere detection fallback)
        cfg_file: Optional .CFG file path (primary hemisphere source)

    Returns:
        List of dicts with keys: lat, long, velocity, heading, height
        All coordinates are decimal degrees. Empty list on error.
    """
    log_message(f"Parsing track data file: {data_path}")

    is_eastern = None
    if cfg_file:
        is_eastern = get_cfg_hemisphere(cfg_file)
        log_message(f"  Using CFG file hemisphere: {'Eastern' if is_eastern else 'Western'}")

    try:
        with open(data_path, "r", errors="ignore") as f:
            content = f.read()

        if "[data]" not in content:
            log_message(f"File does not contain [data] section: {data_path}")
            return []

        log_message("Found RaceLogic format with [data] section")
        lines = content.strip().split("\n")

        # Find data section
        data_start_index = -1
        for i, line in enumerate(lines):
            if line.strip() == "[data]":
                data_start_index = i + 1
                break

        if data_start_index == -1:
            log_message(f"Warning: Data section not found in {data_path}")
            return []

        # Parse column names
        column_indices = {}
        for i in range(data_start_index - 10, data_start_index):
            if 0 <= i < len(lines) and lines[i].strip().startswith("[column names]"):
                if i + 1 < len(lines):
                    col_names = lines[i + 1].split()
                    for j, name in enumerate(col_names):
                        column_indices[name.lower()] = j
                    log_message(f"Found column names: {col_names}")
                break

        # Auto-detect format if no column names
        if not column_indices and data_start_index < len(lines):
            first_data = lines[data_start_index].strip()
            parts = [part for part in first_data.split(" ") if part.strip()]
            log_message(f"First data line: {first_data}")

            if len(parts) >= 2:
                column_indices = {"lat": 0, "long": 1}
                log_message("Detected format: lat long")

        if not column_indices:
            log_message(f"Warning: Could not determine data format in {data_path}")
            return []

        # Determine hemisphere from sample data if not from CFG
        if is_eastern is None:
            lat_samples, long_samples = [], []
            for i in range(data_start_index, min(data_start_index + 10, len(lines))):
                line = lines[i].strip()
                if not line:
                    continue
                parts = [part for part in line.split(" ") if part.strip()]
                if len(parts) < 2:
                    continue
                lat_idx = column_indices.get("lat", 0)
                long_idx = column_indices.get("long", 1)
                if lat_idx < len(parts) and long_idx < len(parts):
                    lat_samples.append(parts[lat_idx])
                    long_samples.append(parts[long_idx])

            negative_count = sum(1 for sample in long_samples if sample.startswith("-"))
            if negative_count > len(long_samples) / 2:
                is_eastern = False
                log_message("Determined Western hemisphere from data samples")
            elif lat_samples and long_samples:
                try:
                    sample_lat = abs(float(lat_samples[0].replace("+", "").replace("-", "").replace(",", ".")))
                    sample_long = abs(float(long_samples[0].replace("+", "").replace("-", "").replace(",", ".")))
                    west_hemisphere = determine_hemisphere(track_name, xml_db, sample_lat, sample_long, cfg_file)
                    is_eastern = not west_hemisphere
                except Exception as e:
                    log_message(f"Error determining hemisphere from sample: {e}")
                    is_eastern = False

        # Parse data rows
        parsed_data = []
        for i in range(data_start_index, len(lines)):
            line = lines[i].strip()
            if not line:
                continue

            parts = [part for part in line.split(" ") if part.strip()]
            if len(parts) < 2:
                continue

            try:
                lat_idx = column_indices.get("lat", 0)
                long_idx = column_indices.get("long", 1)

                if lat_idx >= len(parts) or long_idx >= len(parts):
                    continue

                lat = float(parts[lat_idx].replace("+", "").replace(",", "."))
                long = float(parts[long_idx].replace("+", "").replace(",", "."))

                lat_degrees = convert_to_decimal_degrees(lat)

                if is_eastern:
                    if parts[long_idx].startswith("-"):
                        long_degrees = convert_to_decimal_degrees(abs(long))
                    else:
                        long_degrees = convert_to_decimal_degrees(long)
                else:
                    if parts[long_idx].startswith("-"):
                        long_degrees = -convert_to_decimal_degrees(abs(long))
                    else:
                        long_degrees = -convert_to_decimal_degrees(long)

                velocity = heading = height = 0
                if "velocity" in column_indices and column_indices["velocity"] < len(parts):
                    velocity = float(parts[column_indices["velocity"]].replace(",", "."))
                if "heading" in column_indices and column_indices["heading"] < len(parts):
                    heading = float(parts[column_indices["heading"]].replace(",", "."))
                if "height" in column_indices and column_indices["height"] < len(parts):
                    height = float(parts[column_indices["height"]].replace(",", ".").replace("+", ""))

                parsed_data.append({
                    "lat": lat_degrees,
                    "long": long_degrees,
                    "velocity": velocity,
                    "heading": heading,
                    "height": height,
                })
            except (ValueError, IndexError) as e:
                log_message(f"Error parsing line {i} in {data_path}: {e}")
                continue

        log_message(f"Successfully parsed {len(parsed_data)} points from {data_path}")
        return parsed_data

    except Exception as e:
        log_message(f"Error parsing file {data_path}: {e}")

    return []


# =============================================================================
# XML Database Functions
# =============================================================================

def get_track_info_from_database(track_name, xml_db):
    """Extract track information from XML database."""
    track_info = {"name": track_name, "splitinfo": []}

    if xml_db is None:
        log_message(f"No XML database provided for track: {track_name}")
        return track_info

    try:
        for country in xml_db.findall(".//country"):
            for circuit in country.findall(".//circuit"):
                circuit_name = circuit.get("name").strip().lower()
                if circuit_name == track_name.strip().lower():
                    for attr in ["name", "min", "max", "length", "gatewidth"]:
                        if attr in circuit.attrib:
                            track_info[attr] = circuit.get(attr)

                    splitinfo_elem = circuit.find("splitinfo")
                    if splitinfo_elem is not None:
                        for split in splitinfo_elem:
                            split_data = {attr: split.get(attr) for attr in split.attrib}
                            track_info["splitinfo"].append(split_data)

                    log_message(f"Found track info for {track_name}: attributes={list(track_info.keys())}")
                    return track_info

        log_message(f"No track info found in database for: {track_name}")
    except Exception as e:
        log_message(f"Warning: Error extracting track info for {track_name}: {e}")

    return track_info


def find_database_file(root_dir):
    """Find the XML database file that contains track information."""
    log_message(f"Searching for XML database file in {root_dir}")

    known_path = os.path.join(root_dir, "Start Finish Database", "StartFinishDataBase.xml")
    if os.path.exists(known_path):
        log_message(f"Checking known database location: {known_path}")
        try:
            tree = ET.parse(known_path)
            root = tree.getroot()
            log_message(f"Found track database at known location: {known_path}")
            return root
        except Exception as e:
            log_message(f"Error parsing known XML file {known_path}: {e}")

    log_message("Searching recursively for database file")
    for root_path, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".xml") and ("database" in file.lower() or "db" in file.lower()):
                xml_file = os.path.join(root_path, file)
                log_message(f"Found potential database file: {xml_file}")
                try:
                    tree = ET.parse(xml_file)
                    xml_root = tree.getroot()
                    if xml_root.tag.endswith("Database") or xml_root.find(".//circuit") is not None:
                        log_message(f"Found track database through search: {xml_file}")
                        return xml_root
                except Exception as e:
                    log_message(f"Error parsing XML file {xml_file}: {e}")

    log_message("No database file found!")
    return None


def is_combo_track(track_name, xml_db):
    """Check if a track is a combo track (multiple layouts overlaid)."""
    if "combo" in track_name.lower():
        return True

    if xml_db is None:
        return False

    try:
        for country in xml_db.findall(".//country"):
            for circuit in country.findall(".//circuit"):
                if circuit.get("name", "").strip().lower() == track_name.strip().lower():
                    if circuit.get("combo", "").lower() == "true":
                        return True
                    if circuit.get("length", "1") == "0":
                        return True
                    return False
    except Exception as e:
        log_message(f"Error checking combo status for {track_name}: {e}")

    return False
