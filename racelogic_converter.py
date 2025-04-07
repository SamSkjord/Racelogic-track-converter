"""

This script converts RaceLogic track files to KML/KMZ format based on the
specific RaceLogic folder structure. It processes all track configurations
and creates corresponding KMZ files with accurate track paths.

Usage:
    python racelogic_converter.py C:\\ProgramData\\Racelogic output_directory
"""

# List of .cir files to skip
SKIP_TRACKS = {
    "Falkenberg.CIR", #broken version in the china directory
    "test track.CIR", #Autobahn Country Club without the start\finish line
}


import sys
import os
import zipfile
import io
import math
import glob
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import multiprocessing
from datetime import datetime
import traceback

# Set up basic logging to file
log_file_path = "racelogic_debug.log"
def log_message(message):
    with open(log_file_path, "a") as log:
        log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

# Log startup information
log_message("Script started")
log_message(f"Python version: {sys.version}")
log_message(f"Arguments: {sys.argv}")

def convert_to_decimal_degrees(racelogic_minutes):
    """Convert RaceLogic minutes to decimal degrees"""
    return racelogic_minutes / 60.0


def determine_hemisphere(track_name, xml_db, base_lat, base_long, cfg_file=None):
    """Determine if a location is in Western or Eastern hemisphere"""
    log_message(f"Determining hemisphere for track: {track_name}")
    
    # First priority: Check the CFG file if available - most reliable source
    if cfg_file:
        cfg_data = parse_config_file(cfg_file)
        if "base_longitude" in cfg_data:
            # In RaceLogic format, the sign of base_longitude in CFG indicates hemisphere
            # + means Eastern hemisphere, - means Western hemisphere
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
                        # Use min coordinates to determine hemisphere
                        min_coords = circuit.get("min").split(",")
                        if len(min_coords) == 2:
                            min_long = float(min_coords[0])
                            # If min_long is negative in RaceLogic minutes, it's Western hemisphere
                            result = min_long < 0
                            log_message(f"  Determined from DB: {result}, min_long={min_long}")
                            return result
    
    # Third priority: Use common geographic knowledge
    # These lat/long are in RaceLogic minutes format
    
    # Check if it's clearly in the Americas (Western hemisphere)
    if base_lat > 0 and base_long > 1000:  # North America
        log_message(f"  Determined as North America (West): base_lat={base_lat}, base_long={base_long}")
        return True
    
    # Check if it's clearly in Asia/Australia/Eastern regions (Eastern hemisphere)
    if base_long > 300 and base_long < 1000:
        log_message(f"  Determined as Asia/Australia (East): base_long={base_long}")
        return False
    
    # Check if it's clearly in Europe/Africa
    if base_lat > 2800 and base_lat < 3800:  # Europe/UK latitude range
        log_message(f"  Determined as Europe (West): base_lat={base_lat}")
        return True  # Most European tracks are in Western hemisphere
    
    # Default to using the sign from the base_long value
    result = base_long < 0
    log_message(f"  Using default sign: {result}, base_long={base_long}")
    return result


def parse_config_file(cfg_path):
    """Parse the .CFG file to get base coordinates and other parameters"""
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
    """Get hemisphere from CFG file based on base_longitude sign"""
    if not cfg_file:
        return None
    
    cfg_data = parse_config_file(cfg_file)
    if "base_longitude" in cfg_data:
        base_long_cfg = cfg_data["base_longitude"]
        is_eastern = base_long_cfg.startswith("+")
        log_message(f"  CFG hemisphere: {'Eastern' if is_eastern else 'Western'}, base_long={base_long_cfg}")
        return is_eastern  # True for Eastern, False for Western
    
    return None


def get_track_info_from_database(track_name, xml_db):
    """Extract track information from XML database"""
    track_info = {"name": track_name, "splitinfo": []}

    if xml_db is None:
        log_message(f"No XML database provided for track: {track_name}")
        return track_info

    try:
        # Find the circuit in the XML database
        for country in xml_db.findall(".//country"):
            for circuit in country.findall(".//circuit"):
                # Compare case-insensitive and strip spaces
                if (
                    circuit.get("name").strip().lower() == track_name.strip().lower()
                    or circuit.get("name").strip().lower()
                    == track_name.strip().lower() + " "
                    or circuit.get("name").strip().lower()
                    == " " + track_name.strip().lower()
                ):
                    # Extract circuit attributes
                    for attr in ["name", "min", "max", "length", "gatewidth"]:
                        if attr in circuit.attrib:
                            track_info[attr] = circuit.get(attr)

                    # Extract splitinfo
                    splitinfo_elem = circuit.find("splitinfo")
                    if splitinfo_elem is not None:
                        for split in splitinfo_elem:
                            split_data = {
                                attr: split.get(attr) for attr in split.attrib
                            }
                            track_info["splitinfo"].append(split_data)
                    
                    log_message(f"Found track info for {track_name}: attributes={list(track_info.keys())}")
                    return track_info
        
        log_message(f"No track info found in database for: {track_name}")
    except Exception as e:
        log_message(f"Warning: Error extracting track info for {track_name}: {e}")

    return track_info


def find_track_data_file(track_name, root_dir):
    """Find the corresponding data file for a track"""
    # Try to find .txt, .csv, or .vbo files with the track name
    data_files = []

    # Clean up track name for better matching (remove special characters)
    clean_track_name = re.sub(r"[^a-zA-Z0-9]", " ", track_name).strip()
    clean_track_name = re.sub(
        r"\s+", " ", clean_track_name
    )  # Replace multiple spaces with single space

    log_message(f"Looking for track data file for: {track_name} (cleaned: {clean_track_name})")

    # Create variations of the track name for better matching
    track_variations = [
        track_name,  # Original name
        clean_track_name,  # Cleaned name
        clean_track_name.replace(" ", ""),  # No spaces
        re.sub(r"[^a-zA-Z0-9]", "", track_name),  # Alphanumeric only
    ]

    # Add common abbreviations and track variations
    if "International" in track_name:
        track_variations.append(track_name.replace("International", "Int"))
    if "Circuit" in track_name:
        track_variations.append(track_name.replace("Circuit", ""))
    if "Raceway" in track_name:
        track_variations.append(track_name.replace("Raceway", ""))

    # Look for files with these track name variations
    for track_var in set(track_variations):  # Use set to remove duplicates
        for ext in [".txt", ".csv", ".vbo"]:
            # Look for exact matches first (higher priority)
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    # Exact match for track name with extension
                    if file.lower() == f"{track_var.lower()}{ext}":
                        log_message(f"Found exact match: {os.path.join(root, file)}")
                        return os.path.join(root, file)

                    # Partial match
                    if track_var.lower() in file.lower() and file.lower().endswith(ext):
                        data_files.append(os.path.join(root, file))

    if data_files:
        log_message(f"Found partial match: {data_files[0]}")
        return data_files[0]

    # Look for .CIR files that might contain track data
    cir_files_dir = os.path.join(root_dir, "CIR Files")
    if os.path.exists(cir_files_dir):
        log_message(f"Checking CIR files directory: {cir_files_dir}")
        for root, dirs, files in os.walk(cir_files_dir):
            for file in files:
                if file.lower() == f"{track_name.lower()}.cir":
                    # Check if the CIR file contains track data
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


def parse_track_data_file(data_path, xml_db, track_name, cfg_file=None):
    """Parse a track data file in RaceLogic format, handling different file variations"""
    log_message(f"Parsing track data file: {data_path}")
    
    # Determine hemisphere from CFG file
    is_eastern = None
    if cfg_file:
        is_eastern = get_cfg_hemisphere(cfg_file)
        log_message(f"  Using CFG file hemisphere: {'Eastern' if is_eastern else 'Western'}")
    
    try:
        with open(data_path, "r", errors="ignore") as f:
            content = f.read()

        # Check if this is in the expected RaceLogic format
        if "[data]" in content:
            log_message("Found RaceLogic format with [data] section")
            # Found RaceLogic format, parse it
            lines = content.strip().split("\n")

            # Find where the data section begins
            data_start_index = -1
            for i, line in enumerate(lines):
                if line.strip() == "[data]":
                    data_start_index = i + 1
                    break

            if data_start_index == -1:
                log_message(f"Warning: Data section not found in {data_path}")
                return []

            # Parse the column names to determine indices
            column_indices = {}
            col_names = None

            # Look for column names section
            for i in range(data_start_index - 10, data_start_index):
                if (
                    i >= 0
                    and i < len(lines)
                    and lines[i].strip().startswith("[column names]")
                ):
                    if i + 1 < len(lines):
                        col_names = lines[i + 1].split()
                        for j, name in enumerate(col_names):
                            column_indices[name.lower()] = j
                        log_message(f"Found column names: {col_names}")
                    break

            # If column names weren't found, try to detect format from the data
            if not column_indices and data_start_index < len(lines):
                # Check the first data line
                first_data = lines[data_start_index].strip()
                parts = [part for part in first_data.split(" ") if part.strip()]
                log_message(f"First data line: {first_data}")

                # Simple format detection based on number of fields and patterns
                if len(parts) >= 2:
                    # Most common format: lat long (as seen in your example)
                    column_indices = {"lat": 0, "long": 1}
                    log_message("Detected format: lat long")
                elif len(parts) >= 7:
                    # Typical format with satellite data and more fields
                    column_indices = {
                        "lat": 2,
                        "long": 3,
                        "velocity": 4,
                        "heading": 5,
                        "height": 6,
                    }
                    log_message("Detected satellite data format")

            if not column_indices:
                log_message(f"Warning: Could not determine data format in {data_path}")
                return []

            # If hemisphere wasn't determined from CFG file, analyze sample data
            if is_eastern is None:
                # Check first few data rows to determine if we need to negate longitudes
                data_samples = []
                for i in range(data_start_index, min(data_start_index + 10, len(lines))):
                    line = lines[i].strip()
                    if not line:
                        continue

                    parts = [part for part in line.split(" ") if part.strip()]
                    if len(parts) < 2:
                        continue

                    lat_idx = column_indices.get("lat", 0)
                    long_idx = column_indices.get("long", 1)
                    
                    if lat_idx >= len(parts) or long_idx >= len(parts):
                        continue

                    data_samples.append(parts[long_idx])

                # Determine hemisphere by checking if most longitudes start with '-'
                negative_count = sum(1 for sample in data_samples if sample.startswith("-"))
                
                if negative_count > len(data_samples) / 2:
                    is_eastern = False  # Western hemisphere (most samples negative)
                    log_message(f"Determined Western hemisphere from data samples (negative longitudes)")
                else:
                    # If most values don't start with '-', try to determine from other sources
                    if data_samples:
                        try:
                            sample_lat = float(data_samples[0].replace("+", ""))
                            sample_long = float(data_samples[0].replace("-", "").replace("+", ""))
                            west_hemisphere = determine_hemisphere(
                                track_name, 
                                xml_db, 
                                sample_lat, 
                                sample_long,
                                cfg_file
                            )
                            is_eastern = not west_hemisphere
                            log_message(f"Determined hemisphere from database/geography: {'Western' if west_hemisphere else 'Eastern'}")
                        except Exception as e:
                            log_message(f"Error determining hemisphere from sample: {e}")
                            is_eastern = False  # Default to Western for safety

            # Parse data rows
            parsed_data = []
            for i in range(data_start_index, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue

                # Split the line by spaces and filter out empty parts
                parts = [part for part in line.split(" ") if part.strip()]
                if len(parts) < 2:  # Need at least lat/long
                    continue

                # Extract latitude and longitude based on column indices
                try:
                    lat_idx = column_indices.get("lat", 0)  # Default to first position
                    long_idx = column_indices.get("long", 1)  # Default to second position

                    if lat_idx >= len(parts) or long_idx >= len(parts):
                        continue  # Skip if indices are out of range

                    lat = parts[lat_idx].replace("+", "")
                    long = parts[long_idx].replace("+", "")

                    # Convert to float
                    lat = float(lat.replace(",", "."))
                    long = float(long.replace(",", "."))

                    # Convert RaceLogic minutes to decimal degrees
                    lat_degrees = convert_to_decimal_degrees(lat)

                    # For longitude, the sign handling depends on the determined hemisphere
                    if is_eastern:  # Eastern hemisphere
                        if parts[long_idx].startswith("-"):
                            # In RaceLogic data format, negative longitude in the CIR file
                            # but positive hemisphere (CFG says eastern) means POSITIVE decimal degrees
                            long_degrees = convert_to_decimal_degrees(abs(long))
                            if i == data_start_index:
                                log_message(f"Eastern hemisphere with negative RaceLogic data - converting to positive decimal: {long_degrees}")
                        else:
                            long_degrees = convert_to_decimal_degrees(long)
                    else:  # Western hemisphere
                        if parts[long_idx].startswith("-"):
                            long_degrees = -convert_to_decimal_degrees(abs(long))
                        else:
                            # Positive values in a western context should be negative in decimal
                            long_degrees = -convert_to_decimal_degrees(long)
                            if i == data_start_index:
                                log_message(f"Western hemisphere with positive RaceLogic data - converting to negative decimal: {long_degrees}")

                    # Extract other data points if available
                    velocity = heading = height = 0

                    if "velocity" in column_indices and column_indices["velocity"] < len(parts):
                        velocity = float(parts[column_indices["velocity"]].replace(",", "."))

                    if "heading" in column_indices and column_indices["heading"] < len(parts):
                        heading = float(parts[column_indices["heading"]].replace(",", "."))

                    if "height" in column_indices and column_indices["height"] < len(parts):
                        height = float(parts[column_indices["height"]].replace(",", ".").replace("+", ""))

                    parsed_data.append(
                        {
                            "lat": lat_degrees,
                            "long": long_degrees,
                            "velocity": velocity,
                            "heading": heading,
                            "height": height,
                        }
                    )
                except (ValueError, IndexError) as e:
                    log_message(f"Error parsing line {i} in {data_path}: {e}, parts={parts}")
                    continue  # Skip problematic lines

            log_message(f"Successfully parsed {len(parsed_data)} points from {data_path}")
            return parsed_data
        else:
            log_message(f"File does not contain [data] section: {data_path}")
    except Exception as e:
        log_message(f"Error parsing file {data_path}: {e}")

    return []


def generate_kml(parsed_data, track_info):
    """Generate KML content from parsed data"""
    track_name = track_info.get("name", "Unknown Track")
    log_message(f"Generating KML for track: {track_name} with {len(parsed_data)} points")

    # Check if we have start/finish line coordinates in the track info
    start_finish_coords = []
    for split in track_info.get("splitinfo", []):
        if split.get("name") == "Start / Finish" or split.get("name") == "Start/Finish":
            if "lat" in split and "long" in split:
                sf_lat = convert_to_decimal_degrees(
                    float(split["lat"].replace("+", ""))
                )
                
                # Determine hemisphere based on the sign in the database
                west_hemisphere = split["long"].startswith("-")
                sf_long = -convert_to_decimal_degrees(
                    float(split["long"].replace("-", "").replace("+", ""))
                ) if west_hemisphere else convert_to_decimal_degrees(
                    float(split["long"].replace("+", ""))
                )
                
                start_finish_coords.append((sf_long, sf_lat))
                log_message(f"Added start/finish coordinate: ({sf_lat}, {sf_long})")

    # Create the KML output
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{track_name} Track Map</name>
    <description>Converted from RaceLogic data</description>
    
    <!-- Track path style -->
    <Style id="trackStyle">
      <LineStyle>
        <color>ff0000ff</color>
        <width>4</width>
      </LineStyle>
    </Style>
    
    <!-- Start/Finish line style -->
    <Style id="startFinishStyle">
      <LineStyle>
        <color>ff00ff00</color>
        <width>6</width>
      </LineStyle>
    </Style>
    
    <!-- Track Path Placemark -->
    <Placemark>
      <name>Track Path</name>
      <styleUrl>#trackStyle</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>"""

    # Add each coordinate point
    for point in parsed_data:
        kml += (
            f"\n          {point['long']:.8f},{point['lat']:.8f},{point['height']:.1f}"
        )

    kml += """
        </coordinates>
      </LineString>
    </Placemark>
    """

    # Add Start/Finish line if we have the coordinates
    if len(start_finish_coords) == 1:
        sf_long, sf_lat = start_finish_coords[0]
        kml += f"""
        <!-- Start/Finish Marker Placemark -->
        <Placemark>
        <name>Start / Finish</name>
        <styleUrl>#startFinishStyle</styleUrl>
        <Point>
            <coordinates>{sf_long:.8f},{sf_lat:.8f},0</coordinates>
        </Point>
        </Placemark>"""

    kml += """
    </Document>
    </kml>"""

    return kml



def create_kmz(kml_content, output_path):
    """Create a KMZ file (zipped KML) for Google Earth"""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        log_message(f"Creating KMZ file: {output_path}")

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as kmz_file:
            kmz_file.writestr("doc.kml", kml_content)
        log_message(f"Successfully created KMZ file")
        return True
    except Exception as e:
        log_message(f"Error creating KMZ file {output_path}: {e}")
        return False


def process_track(track_params):
    """Process a single track file and convert to KMZ"""
    cir_path, output_dir, root_dir, xml_db = track_params

    try:
        # Get track name from .CIR file
        track_name = os.path.basename(cir_path).replace(".CIR", "").replace(".cir", "")
        log_message(f"Processing track: {track_name} from {cir_path}")

        # Create output file path
        country_dir = os.path.basename(os.path.dirname(cir_path))
        output_path = os.path.join(output_dir, country_dir, track_name + ".kmz")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        log_message(f"Output path: {output_path}")

        # If the output file already exists, skip
        if os.path.exists(output_path):
            log_message(f"Skipping {track_name} (already exists)")
            return True

        # Get track info from database
        track_info = get_track_info_from_database(track_name, xml_db)

        # Find corresponding CFG file
        cfg_files = []

        # Try to find in the country-specific DataBaseTrackmapFiles directory
        country_trackmaps_dir = os.path.join(
            root_dir, "DataBaseTrackmapFiles", country_dir, "TRACKMAPS"
        )
        if os.path.exists(country_trackmaps_dir):
            log_message(f"Checking country trackmaps directory: {country_trackmaps_dir}")
            # Use case-insensitive file search (compatible with older Python versions)
            cfg_pattern = os.path.join(country_trackmaps_dir, f"{track_name}.CFG")
            for file in os.listdir(country_trackmaps_dir):
                if file.lower() == f"{track_name.lower()}.cfg":
                    cfg_files.append(os.path.join(country_trackmaps_dir, file))
                    log_message(f"Found country-specific CFG file: {cfg_files[-1]}")

        # If not found, try the root TRACKMAPS directory
        if not cfg_files:
            trackmaps_dir = os.path.join(root_dir, "DataBaseTrackmapFiles", "TRACKMAPS")
            if os.path.exists(trackmaps_dir):
                log_message(f"Checking root trackmaps directory: {trackmaps_dir}")
                for file in os.listdir(trackmaps_dir):
                    if file.lower() == f"{track_name.lower()}.cfg":
                        cfg_files.append(os.path.join(trackmaps_dir, file))
                        log_message(f"Found root CFG file: {cfg_files[-1]}")

        # Get the CFG file if available
        cfg_file = cfg_files[0] if cfg_files else None

        # Find track data file
        data_file = find_track_data_file(track_name, root_dir)

        # Check if the CIR file itself contains track data
        cir_has_data = False
        try:
            with open(cir_path, "r", errors="ignore") as f:
                content = f.read()
                if "[data]" in content:
                    data_file = cir_path
                    cir_has_data = True
                    log_message(f"CIR file contains track data")
        except Exception as e:
            log_message(f"Error checking CIR file for data: {e}")

        # Parse track data or generate synthetic data
        data_source = "unknown"
        parsed_data = []

        if data_file:
            log_message(f"Parsing data file: {data_file}")
            parsed_data = parse_track_data_file(data_file, xml_db, track_name, cfg_file)
            data_source = "real" if not cir_has_data else "cir"
            log_message(f"Data source: {data_source}")

        # Generate KML if we have any data
        if parsed_data:
            log_message(f"Generating KML with {len(parsed_data)} data points")
            kml_content = generate_kml(parsed_data, track_info)

            # Create KMZ
            if create_kmz(kml_content, output_path):
                log_message(f"Converted {track_name} (using {data_source} data)")
                return True
            else:
                log_message(f"Warning: Failed to create KMZ for {track_name}")
        else:
            # Log the failure and continue
            failure_log = os.path.join(output_dir, "conversion_failures.log")
            log_message(f"No valid data found for {track_name} - logging to {failure_log}")
            with open(failure_log, "a") as log_file:
                log_file.write(f"{country_dir}/{track_name}: No valid data found\n")
            log_message(f"Warning: No valid data found for {track_name} - logged failure")

            return False
    except Exception as e:
        # Log the exception
        error_log = os.path.join(output_dir, "conversion_errors.log") 
        log_message(f"Error processing {cir_path}: {e}")
        log_message(f"Error traceback: {traceback.format_exc()}")
        with open(error_log, "a") as log_file:
            log_file.write(f"{country_dir}/{track_name}: {str(e)}\n")
        return False


def find_database_file(root_dir):
    """Find the XML database file that contains track information"""
    log_message(f"Searching for XML database file in {root_dir}")
    # Check the known location first
    known_path = os.path.join(
        root_dir, "Start Finish Database", "StartFinishDataBase.xml"
    )
    if os.path.exists(known_path):
        log_message(f"Checking known database location: {known_path}")
        try:
            tree = ET.parse(known_path)
            root = tree.getroot()
            log_message(f"Found track database at known location: {known_path}")
            return root
        except Exception as e:
            log_message(f"Error parsing known XML file {known_path}: {e}")

    # If not found, search recursively
    log_message(f"Searching recursively for database file")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".xml") and (
                "database" in file.lower() or "db" in file.lower()
            ):
                xml_file = os.path.join(root, file)
                log_message(f"Found potential database file: {xml_file}")
                try:
                    tree = ET.parse(xml_file)
                    xml_root = tree.getroot()
                    # Check if it looks like the track database
                    if (
                        xml_root.tag.endswith("Database")
                        or xml_root.find(".//circuit") is not None
                    ):
                        log_message(f"Found track database through search: {xml_file}")
                        return xml_root
                except Exception as e:
                    log_message(f"Error parsing XML file {xml_file}: {e}")

    log_message(f"No database file found!")
    return None


def process_all_tracks(input_dir, output_dir, num_processes=None):
    """Process all track files in the RaceLogic folder structure"""
    log_message(f"Starting to process all tracks: input={input_dir}, output={output_dir}")
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize log files
    failure_log = os.path.join(output_dir, "conversion_failures.log")
    error_log = os.path.join(output_dir, "conversion_errors.log")
    
    log_message(f"Initializing log files: {failure_log}, {error_log}")
    with open(failure_log, "w") as log_file:
        log_file.write(
            f"# Track conversion failures - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        log_file.write("# Format: country/track_name: reason\n\n")

    with open(error_log, "w") as log_file:
        log_file.write(
            f"# Track conversion errors - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        log_file.write("# Format: country/track_name: error message\n\n")

    # Load the database file
    xml_db = find_database_file(input_dir)

    # Find all .CIR files
    cir_files_dir = os.path.join(input_dir, "CIR Files")
    log_message(f"Looking for CIR Files directory: {cir_files_dir}")
    if not os.path.exists(cir_files_dir):
        log_message(f"Error: CIR Files directory not found: {cir_files_dir}")
        return False

    # Find all .CIR files recursively
    cir_files = []
    log_message(f"Starting recursive search for CIR files")
    for root, dirs, files in os.walk(cir_files_dir):
        for file in files:
            if file.lower().endswith(".cir"):
                cir_files.append(os.path.join(root, file))

    log_message(f"Found {len(cir_files)} CIR files to process")

    # Prepare parameters for processing
    track_params = [
                        (cir_file, output_dir, input_dir, xml_db)
                        for cir_file in cir_files
                        if os.path.basename(cir_file) not in SKIP_TRACKS
                    ]

    # Track progress
    total_tracks = len(track_params)
    success_count = 0
    error_count = 0

    # Determine number of processes to use
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)

    # Log start time
    start_time = datetime.now()
    log_message(
        f"Starting conversion at {start_time.strftime('%H:%M:%S')} using {num_processes} processes..."
    )

    # For debugging purposes, let's process tracks sequentially first
    log_message("Using sequential processing for debugging")
    results = []
    for params in track_params:  # Limit to first 10 tracks for testing
        try:
            result = process_track(params)
            results.append(result)
        except Exception as e:
            log_message(f"Error in process_track: {e}")
            log_message(f"Traceback: {traceback.format_exc()}")
            results.append(False)

    # Count successes and errors
    success_count = sum(1 for result in results if result)
    error_count = len(results) - success_count

    # Log end time
    end_time = datetime.now()
    duration = end_time - start_time

    # Create summary report
    summary_file_path = os.path.join(output_dir, "conversion_summary.txt")
    log_message(f"Creating summary report: {summary_file_path}")
    with open(summary_file_path, "w") as summary_file:
        summary_file.write(f"# RaceLogic Track Conversion Summary\n\n")
        summary_file.write(f"Date: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary_file.write(f"Duration: {duration}\n\n")
        summary_file.write(f"Total tracks processed: {len(results)}\n")
        summary_file.write(f"Successfully converted: {success_count}\n")
        summary_file.write(f"Failed conversions: {error_count}\n\n")

        # Calculate success rate
        success_rate = (success_count / len(results)) * 100 if len(results) > 0 else 0
        summary_file.write(f"Success rate: {success_rate:.2f}%\n\n")

        # Add information about failure log
        if error_count > 0:
            summary_file.write(
                "See conversion_failures.log and conversion_errors.log for details on failed tracks.\n"
            )

    log_message(f"\nConversion complete at {end_time.strftime('%H:%M:%S')}")
    log_message(f"Duration: {duration}")
    log_message(
        f"Results: {success_count} files converted successfully, {error_count} failed"
    )
    log_message(f"Success rate: {success_rate:.2f}%")
    log_message(f"See {summary_file_path} for details")

    return success_count, error_count


if __name__ == "__main__":
    log_message("Script main block started")
    
    if len(sys.argv) != 3:
        log_message("Error: Invalid command line arguments")
        print(
            "Usage: python racelogic_batch_converter.py input_root_dir output_root_dir"
        )
        sys.exit(1)

    input_root_dir = sys.argv[1]
    output_root_dir = sys.argv[2]

    log_message(f"Input directory: {input_root_dir}")
    log_message(f"Output directory: {output_root_dir}")

    if not os.path.exists(input_root_dir):
        log_message(f"Error: Input directory not found: {input_root_dir}")
        print(f"Error: Input directory not found: {input_root_dir}")
        sys.exit(1)

    # Process all tracks
    try:
        log_message("Starting process_all_tracks")
        process_all_tracks(input_root_dir, output_root_dir)
        log_message("process_all_tracks completed")
    except Exception as e:
        log_message(f"Unhandled exception in process_all_tracks: {e}")
        log_message(f"Traceback: {traceback.format_exc()}")
        print(f"Error: {e}")
        sys.exit(1)