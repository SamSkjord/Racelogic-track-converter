"""
KML/KMZ generation for RaceLogic track data.

Handles:
- KML document generation with boundaries and markers
- Start/finish line generation
- KMZ packaging
"""

import os
import zipfile
import math

from tracks_db import haversine_distance
from racelogic_parser import convert_to_decimal_degrees
from racelogic_boundary import detect_and_split_boundaries


def log_message(message):
    """Logging stub - replaced at runtime by main module."""
    pass


def set_logger(logger_func):
    """Set the logging function to use."""
    global log_message
    log_message = logger_func


# =============================================================================
# Point Finding Utilities
# =============================================================================

def find_closest_point(boundary_points, target_lat, target_lon):
    """Find the point in a boundary closest to a target coordinate."""
    if not boundary_points:
        return None

    closest_point = None
    min_dist = float('inf')

    for point in boundary_points:
        dist = haversine_distance(target_lat, target_lon, point['lat'], point['long'])
        if dist < min_dist:
            min_dist = dist
            closest_point = point

    return closest_point


def find_closest_point_with_index(boundary_points, target_lat, target_lon):
    """Find the point and its index in a boundary closest to a target coordinate."""
    if not boundary_points:
        return None, -1

    closest_point = None
    closest_idx = -1
    min_dist = float('inf')

    for i, point in enumerate(boundary_points):
        dist = haversine_distance(target_lat, target_lon, point['lat'], point['long'])
        if dist < min_dist:
            min_dist = dist
            closest_point = point
            closest_idx = i

    return closest_point, closest_idx


# =============================================================================
# Perpendicular Line Generation
# =============================================================================

def generate_perpendicular_line_at_coord(boundary_points, direction_idx, center_lat, center_lon, half_width_meters=8.0):
    """
    Generate a line perpendicular to the track direction at a specific coordinate.

    Uses boundary_points[direction_idx] and neighbors to determine track direction,
    but places the line at (center_lat, center_lon).

    Returns:
        Tuple of two points representing the line endpoints
    """
    if not boundary_points or direction_idx < 0 or direction_idx >= len(boundary_points):
        return None, None

    prev_idx = max(0, direction_idx - 1)
    next_idx = min(len(boundary_points) - 1, direction_idx + 1)

    prev_point = boundary_points[prev_idx]
    next_point = boundary_points[next_idx]

    dlat = next_point['lat'] - prev_point['lat']
    dlon = next_point['long'] - prev_point['long']

    # Perpendicular direction
    perp_lat = -dlon
    perp_lon = dlat

    # Convert to meters
    lat_rad = math.radians(center_lat)
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * math.cos(lat_rad)

    perp_lat_m = perp_lat * meters_per_deg_lat
    perp_lon_m = perp_lon * meters_per_deg_lon

    length = math.sqrt(perp_lat_m**2 + perp_lon_m**2)
    if length < 0.001:
        return None, None

    perp_lat_m /= length
    perp_lon_m /= length

    offset_lat = (perp_lat_m * half_width_meters) / meters_per_deg_lat
    offset_lon = (perp_lon_m * half_width_meters) / meters_per_deg_lon

    point1 = {'lat': center_lat + offset_lat, 'long': center_lon + offset_lon, 'height': 0}
    point2 = {'lat': center_lat - offset_lat, 'long': center_lon - offset_lon, 'height': 0}

    return point1, point2


# =============================================================================
# KML Generation
# =============================================================================

def generate_kml(parsed_data, track_info):
    """
    Generate KML content from parsed track data.

    Automatically detects and splits dual boundaries, adds start/finish
    markers and lines perpendicular to the track direction.

    Args:
        parsed_data: List of points with 'lat', 'long', 'height' keys
        track_info: Dict with 'name' and 'splitinfo' (list of S/F markers)
                   Each splitinfo entry has 'name', 'lat', 'long' keys

    Returns:
        KML document as string, ready for packaging into KMZ
    """
    track_name = track_info.get("name", "Unknown Track")
    log_message(f"Generating KML for track: {track_name} with {len(parsed_data)} points")

    boundaries = detect_and_split_boundaries(parsed_data)

    start_coord = None
    finish_coord = None

    for split in track_info.get("splitinfo", []):
        name = split.get("name", "").strip()
        if "lat" not in split or "long" not in split:
            continue

        lat = convert_to_decimal_degrees(float(split["lat"].replace("+", "")))
        long_raw = float(split["long"].replace("+", "").replace("-", ""))
        long = -convert_to_decimal_degrees(long_raw) if split["long"].startswith("-") else convert_to_decimal_degrees(long_raw)

        if name in ("Start / Finish", "Start/Finish"):
            start_coord = (long, lat)
        elif name == "Finish":
            finish_coord = (long, lat)

    # Start KML document
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{track_name} Track Map</name>
    <description>Converted from RaceLogic data</description>

    <Style id="outerBoundaryStyle">
      <LineStyle><color>ff0000ff</color><width>4</width></LineStyle>
    </Style>
    <Style id="innerBoundaryStyle">
      <LineStyle><color>ffff0000</color><width>4</width></LineStyle>
    </Style>
    <Style id="startFinishStyle">
      <LineStyle><color>ff00ff00</color><width>6</width></LineStyle>
    </Style>
    <Style id="startFinishLineStyle">
      <LineStyle><color>ff00ff00</color><width>3</width></LineStyle>
    </Style>
    <Style id="centerlineStyle">
      <LineStyle><color>ff00ffff</color><width>5</width></LineStyle>
    </Style>"""

    if boundaries['type'] == 'dual':
        log_message(f"Generating KML with dual boundaries")

        # Outer boundary
        kml += """

    <Placemark>
      <name>Outer Boundary</name>
      <styleUrl>#outerBoundaryStyle</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>"""

        for point in boundaries['boundary1']:
            kml += f"\n          {point['long']:.8f},{point['lat']:.8f},{point.get('height', 0):.1f}"

        kml += """
        </coordinates>
      </LineString>
    </Placemark>"""

        # Inner boundary
        kml += """

    <Placemark>
      <name>Inner Boundary</name>
      <styleUrl>#innerBoundaryStyle</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>"""

        for point in boundaries['boundary2']:
            kml += f"\n          {point['long']:.8f},{point['lat']:.8f},{point.get('height', 0):.1f}"

        kml += """
        </coordinates>
      </LineString>
    </Placemark>"""

        # Start/finish line across track
        if start_coord:
            sf_outer = find_closest_point(boundaries['boundary1'], start_coord[1], start_coord[0])
            sf_inner = find_closest_point(boundaries['boundary2'], start_coord[1], start_coord[0])

            if sf_outer and sf_inner:
                kml += f"""

    <Placemark>
      <name>{'Start' if finish_coord else 'Start / Finish'} Line</name>
      <styleUrl>#startFinishLineStyle</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>
          {sf_outer['long']:.8f},{sf_outer['lat']:.8f},0
          {sf_inner['long']:.8f},{sf_inner['lat']:.8f},0
        </coordinates>
      </LineString>
    </Placemark>"""

    else:
        # Single boundary
        log_message(f"Generating KML with single boundary")

        kml += """

    <Placemark>
      <name>Track Path</name>
      <styleUrl>#outerBoundaryStyle</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>"""

        for point in boundaries['boundary1']:
            kml += f"\n          {point['long']:.8f},{point['lat']:.8f},{point.get('height', 0):.1f}"

        kml += """
        </coordinates>
      </LineString>
    </Placemark>"""

        # Start/finish lines for single-boundary tracks
        if start_coord:
            _, start_idx = find_closest_point_with_index(boundaries['boundary1'], start_coord[1], start_coord[0])
            if start_idx >= 0:
                p1, p2 = generate_perpendicular_line_at_coord(
                    boundaries['boundary1'], start_idx,
                    start_coord[1], start_coord[0],
                    half_width_meters=8.0
                )
                if p1 and p2:
                    kml += f"""

    <Placemark>
      <name>{'Start Line' if finish_coord else 'Start / Finish Line'}</name>
      <styleUrl>#startFinishLineStyle</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>
          {p1['long']:.8f},{p1['lat']:.8f},0
          {p2['long']:.8f},{p2['lat']:.8f},0
        </coordinates>
      </LineString>
    </Placemark>"""

        if finish_coord:
            _, finish_idx = find_closest_point_with_index(boundaries['boundary1'], finish_coord[1], finish_coord[0])
            if finish_idx >= 0:
                p1, p2 = generate_perpendicular_line_at_coord(
                    boundaries['boundary1'], finish_idx,
                    finish_coord[1], finish_coord[0],
                    half_width_meters=8.0
                )
                if p1 and p2:
                    kml += f"""

    <Placemark>
      <name>Finish Line</name>
      <styleUrl>#startFinishLineStyle</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>
          {p1['long']:.8f},{p1['lat']:.8f},0
          {p2['long']:.8f},{p2['lat']:.8f},0
        </coordinates>
      </LineString>
    </Placemark>"""

    # Add start/finish point markers
    if start_coord:
        kml += f"""

    <Placemark>
      <name>{'Start' if finish_coord else 'Start / Finish'}</name>
      <styleUrl>#startFinishStyle</styleUrl>
      <Point>
        <coordinates>{start_coord[0]:.8f},{start_coord[1]:.8f},0</coordinates>
      </Point>
    </Placemark>"""

    if finish_coord:
        kml += f"""

    <Placemark>
      <name>Finish</name>
      <styleUrl>#startFinishStyle</styleUrl>
      <Point>
        <coordinates>{finish_coord[0]:.8f},{finish_coord[1]:.8f},0</coordinates>
      </Point>
    </Placemark>"""

    kml += """
  </Document>
</kml>"""

    return kml


def create_kmz(kml_content, output_path):
    """Create a KMZ file (zipped KML) for Google Earth."""
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        log_message(f"Creating KMZ file: {output_path}")

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as kmz_file:
            kmz_file.writestr("doc.kml", kml_content)
        log_message("Successfully created KMZ file")
        return True
    except Exception as e:
        log_message(f"Error creating KMZ file {output_path}: {e}")
        return False
