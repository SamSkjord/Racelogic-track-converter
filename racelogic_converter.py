"""
RaceLogic Track Converter

Imports RaceLogic track data from import/Racelogic/ folder:
- Converts .CIR boundary files to KMZ format
- Updates SQLite database with track metadata
- Removes source files after successful import

Input:  import/Racelogic/  (RaceLogic folder structure)
Output: tracks/racelogic/{country}/{track}.kmz + tracks/racelogic.db

Usage:
    python racelogic_converter.py
"""

import sys
import os
import shutil
from datetime import datetime
import traceback

from tracks_db import TracksDB, import_racelogic_xml
import racelogic_parser as parser
import racelogic_boundary as boundary
import racelogic_kml as kml

# =============================================================================
# Configuration
# =============================================================================

SKIP_TRACKS = {
    "Falkenberg.CIR",  # broken version in the china directory
    "test track.CIR",  # Autobahn Country Club without the start/finish line
}

LOG_FILE_PATH = "racelogic_debug.log"


def log_message(message):
    with open(LOG_FILE_PATH, "a") as log:
        log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


# Initialize logging in submodules
parser.set_logger(log_message)
boundary.set_logger(log_message)
kml.set_logger(log_message)

log_message("Script started")
log_message(f"Python version: {sys.version}")


# =============================================================================
# Track Processing
# =============================================================================

def process_track(track_params):
    """Process a single track file and convert to KMZ."""
    cir_path, output_dir, root_dir, xml_db = track_params

    try:
        track_name = os.path.basename(cir_path).replace(".CIR", "").replace(".cir", "")
        log_message(f"Processing track: {track_name} from {cir_path}")

        # Skip combo tracks
        if parser.is_combo_track(track_name, xml_db):
            log_message(f"Skipping {track_name} (combo track)")
            return True

        # Create output path
        country_dir = os.path.basename(os.path.dirname(cir_path))
        output_path = os.path.join(output_dir, country_dir, track_name.strip() + ".kmz")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        log_message(f"Output path: {output_path}")

        # Get track info from database
        track_info = parser.get_track_info_from_database(track_name, xml_db)

        # Find CFG file for hemisphere detection
        cfg_file = None
        country_trackmaps_dir = os.path.join(root_dir, "DataBaseTrackmapFiles", country_dir, "TRACKMAPS")
        if os.path.exists(country_trackmaps_dir):
            for file in os.listdir(country_trackmaps_dir):
                if file.lower() == f"{track_name.lower()}.cfg":
                    cfg_file = os.path.join(country_trackmaps_dir, file)
                    log_message(f"Found CFG file: {cfg_file}")
                    break

        # Check if CIR file contains track data
        data_file = None
        try:
            with open(cir_path, "r", errors="ignore") as f:
                content = f.read()
                if "[data]" in content:
                    data_file = cir_path
                    log_message("CIR file contains track data")
        except Exception as e:
            log_message(f"Error checking CIR file: {e}")

        # Parse track data
        parsed_data = []
        if data_file:
            log_message(f"Parsing data file: {data_file}")
            parsed_data = parser.parse_track_data_file(data_file, xml_db, track_name, cfg_file)

        # Generate KML if we have data
        if parsed_data:
            log_message(f"Generating KML with {len(parsed_data)} data points")
            kml_content = kml.generate_kml(parsed_data, track_info)

            if kml.create_kmz(kml_content, output_path):
                log_message(f"Converted {track_name}")
                return True
            else:
                log_message(f"Warning: Failed to create KMZ for {track_name}")
        else:
            failure_log = os.path.join(output_dir, "conversion_failures.log")
            log_message(f"No valid data found for {track_name}")
            with open(failure_log, "a") as log_file:
                log_file.write(f"{country_dir}/{track_name}: No valid data found\n")
            return False

    except Exception as e:
        error_log = os.path.join(output_dir, "conversion_errors.log")
        log_message(f"Error processing {cir_path}: {e}")
        log_message(f"Error traceback: {traceback.format_exc()}")
        with open(error_log, "a") as log_file:
            log_file.write(f"{cir_path}: {str(e)}\n")
        return False


def process_all_tracks(input_dir, output_dir):
    """Process all track files in the RaceLogic folder structure."""
    log_message(f"Starting to process all tracks: input={input_dir}, output={output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize log files
    failure_log = os.path.join(output_dir, "conversion_failures.log")
    error_log = os.path.join(output_dir, "conversion_errors.log")

    with open(failure_log, "w") as f:
        f.write(f"# Track conversion failures - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    with open(error_log, "w") as f:
        f.write(f"# Track conversion errors - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Load the database file
    xml_db = parser.find_database_file(input_dir)

    # Find all CIR files
    cir_files_dir = os.path.join(input_dir, "CIR Files")
    log_message(f"Looking for CIR Files directory: {cir_files_dir}")
    if not os.path.exists(cir_files_dir):
        log_message(f"Error: CIR Files directory not found: {cir_files_dir}")
        return 0, 0

    cir_files = []
    for root, dirs, files in os.walk(cir_files_dir):
        for file in files:
            if file.lower().endswith(".cir"):
                cir_files.append(os.path.join(root, file))

    log_message(f"Found {len(cir_files)} CIR files to process")

    # Prepare parameters
    track_params = [
        (cir_file, output_dir, input_dir, xml_db)
        for cir_file in cir_files
        if os.path.basename(cir_file) not in SKIP_TRACKS
    ]

    # Process tracks
    start_time = datetime.now()
    log_message(f"Starting conversion at {start_time.strftime('%H:%M:%S')}")

    results = []
    for params in track_params:
        try:
            result = process_track(params)
            results.append(result)
        except Exception as e:
            log_message(f"Error in process_track: {e}")
            results.append(False)

    success_count = sum(1 for r in results if r)
    error_count = len(results) - success_count

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    success_rate = (success_count / len(results)) * 100 if results else 0

    summary_path = os.path.join(output_dir, "conversion_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"# RaceLogic Track Conversion Summary\n\n")
        f.write(f"Date: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration}\n\n")
        f.write(f"Total tracks: {len(results)}\n")
        f.write(f"Converted: {success_count}\n")
        f.write(f"Failed: {error_count}\n")
        f.write(f"Success rate: {success_rate:.1f}%\n")

    log_message(f"Conversion complete: {success_count} converted, {error_count} failed")
    return success_count, error_count


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    log_message("Script main block started")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_root_dir = os.path.join(script_dir, "import", "Racelogic")
    output_root_dir = os.path.join(script_dir, "tracks", "racelogic")
    db_path = os.path.join(script_dir, "tracks", "racelogic.db")

    print("RaceLogic Track Converter")
    print(f"  Input:  {input_root_dir}")
    print(f"  Output: {output_root_dir}")

    if not os.path.exists(input_root_dir):
        print("No import/Racelogic/ folder found.")
        sys.exit(0)

    cir_dir = os.path.join(input_root_dir, "CIR Files")
    if not os.path.exists(cir_dir):
        print("No CIR Files/ subfolder in import/Racelogic/")
        sys.exit(0)

    cir_count = sum(1 for root, dirs, files in os.walk(cir_dir)
                    for f in files if f.lower().endswith('.cir'))
    if cir_count == 0:
        print("No .CIR files to import.")
        sys.exit(0)

    print(f"  Found: {cir_count} CIR files")

    try:
        log_message("Starting process_all_tracks")
        success, errors = process_all_tracks(input_root_dir, output_root_dir)
        log_message("process_all_tracks completed")

        # Update database from XML
        xml_path = os.path.join(input_root_dir, "Start Finish Database", "StartFinishDataBase.xml")
        if os.path.exists(xml_path):
            print(f"\nUpdating database...")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            with TracksDB(db_path) as db:
                stats = import_racelogic_xml(db, xml_path, replace=True)
                print(f"  Database: {stats['imported']} added, {stats['replaced']} replaced")

        # Clean up import folder if all succeeded
        if errors == 0 and success > 0:
            print(f"\nCleaning up import/Racelogic/...")
            shutil.rmtree(input_root_dir)
            print("  Removed import/Racelogic/")

        print(f"\nDone: {success} converted, {errors} errors")

    except Exception as e:
        log_message(f"Unhandled exception: {e}")
        log_message(f"Traceback: {traceback.format_exc()}")
        print(f"Error: {e}")
        sys.exit(1)
