#!/usr/bin/env python3
"""
Generate track map PNG images from RaceLogic data.
Creates transparent PNGs with track shaded in aqua and S/F line marked.
"""

import sys
import os
from PIL import Image, ImageDraw

# Import from main converter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from racelogic_converter import (
    find_database_file, get_track_info_from_database,
    parse_track_data_file, detect_and_split_boundaries,
    find_closest_point, haversine_distance
)


def coords_to_pixels(points, width, height, padding=20):
    """Convert GPS coordinates to pixel coordinates."""
    if not points:
        return []

    lats = [p['lat'] for p in points]
    lons = [p['long'] for p in points]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Add small buffer to prevent edge clipping
    lat_range = max_lat - min_lat or 0.0001
    lon_range = max_lon - min_lon or 0.0001

    # Calculate scale to fit in image while maintaining aspect ratio
    # Approximate aspect ratio correction for latitude
    lat_center = (min_lat + max_lat) / 2
    import math
    lon_scale = math.cos(math.radians(lat_center))

    effective_lon_range = lon_range * lon_scale

    scale_x = (width - 2 * padding) / effective_lon_range
    scale_y = (height - 2 * padding) / lat_range
    scale = min(scale_x, scale_y)

    pixels = []
    for p in points:
        x = padding + (p['long'] - min_lon) * lon_scale * scale
        y = padding + (max_lat - p['lat']) * scale  # Flip Y axis
        pixels.append((int(x), int(y)))

    return pixels, (min_lat, max_lat, min_lon, max_lon, scale, lon_scale, padding)


def generate_track_image(track_name, root_dir, output_path, width=800, height=800):
    """Generate a track map PNG."""

    xml_db = find_database_file(root_dir)
    track_info = get_track_info_from_database(track_name, xml_db)

    # Find CIR file
    cir_path = None
    cir_dir = os.path.join(root_dir, "CIR Files")
    for country in os.listdir(cir_dir):
        country_path = os.path.join(cir_dir, country)
        if os.path.isdir(country_path):
            for f in os.listdir(country_path):
                name = f.replace(".CIR", "").replace(".cir", "")
                if name.lower() == track_name.lower():
                    cir_path = os.path.join(country_path, f)
                    break
        if cir_path:
            break

    if not cir_path:
        print(f"Could not find CIR file for {track_name}")
        return False

    # Parse track data
    parsed_data = parse_track_data_file(cir_path, xml_db, track_name, None)
    if not parsed_data:
        print(f"Could not parse track data for {track_name}")
        return False

    boundaries = detect_and_split_boundaries(parsed_data)

    # Create image with transparency
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Aqua color with some transparency
    track_color = (0, 255, 255, 180)  # RGBA - aqua
    sf_color = (255, 255, 255, 255)   # White for S/F line

    if boundaries['type'] == 'dual':
        # Draw track as thick strokes for both boundaries
        outer = boundaries['boundary1']
        inner = boundaries['boundary2']

        # Combine all points for scaling
        all_points = outer + inner

        # Convert to pixels
        outer_px, transform = coords_to_pixels(outer, width, height)

        # Use same transform for inner boundary
        min_lat, max_lat, min_lon, max_lon, scale, lon_scale, padding = transform
        inner_px = []
        for p in inner:
            x = padding + (p['long'] - min_lon) * lon_scale * scale
            y = padding + (max_lat - p['lat']) * scale
            inner_px.append((int(x), int(y)))

        # Generate centerline by finding closest point on inner for each outer point
        # This handles boundaries that aren't aligned at same track positions
        from racelogic_converter import resample_boundary, haversine_distance

        # Resample outer to reasonable number of points
        outer_resampled = resample_boundary(outer, 500)

        # For each outer point, find closest inner point and average
        centerline_px = []
        for o in outer_resampled:
            # Find closest point on inner boundary
            min_dist = float('inf')
            closest_inner = None
            for n in inner:
                dist = haversine_distance(o['lat'], o['long'], n['lat'], n['long'])
                if dist < min_dist:
                    min_dist = dist
                    closest_inner = n

            if closest_inner:
                # Average the two points
                avg_lat = (o['lat'] + closest_inner['lat']) / 2
                avg_lon = (o['long'] + closest_inner['long']) / 2

                x = padding + (avg_lon - min_lon) * lon_scale * scale
                y = padding + (max_lat - avg_lat) * scale
                centerline_px.append((int(x), int(y)))

        # Draw centerline first (thick, to fill gaps)
        if len(centerline_px) >= 2:
            draw.line(centerline_px, fill=track_color, width=30, joint="curve")

        # Draw boundaries on top (thinner, for definition)
        boundary_width = 8
        if len(outer_px) >= 2:
            draw.line(outer_px, fill=track_color, width=boundary_width, joint="curve")
        if len(inner_px) >= 2:
            draw.line(inner_px, fill=track_color, width=boundary_width, joint="curve")

        # Draw S/F line
        start_coord = None
        for split in track_info.get("splitinfo", []):
            if "Start" in split.get("name", ""):
                start_coord = (split.get("long"), split.get("lat"))
                break

        if start_coord:
            sf_lat = float(start_coord[1]) / 60  # Convert from RaceLogic minutes
            sf_lon = float(start_coord[0]) / 60
            sf_outer = find_closest_point(outer, sf_lat, sf_lon)
            sf_inner = find_closest_point(inner, sf_lat, sf_lon)

            if sf_outer and sf_inner:
                # Convert S/F points to pixels
                sf_outer_x = padding + (sf_outer['long'] - min_lon) * lon_scale * scale
                sf_outer_y = padding + (max_lat - sf_outer['lat']) * scale
                sf_inner_x = padding + (sf_inner['long'] - min_lon) * lon_scale * scale
                sf_inner_y = padding + (max_lat - sf_inner['lat']) * scale

                draw.line([(int(sf_outer_x), int(sf_outer_y)),
                          (int(sf_inner_x), int(sf_inner_y))],
                         fill=sf_color, width=3)

    else:
        # Single boundary - just draw the outline
        boundary = boundaries['boundary1']
        boundary_px, transform = coords_to_pixels(boundary, width, height)

        if len(boundary_px) >= 2:
            draw.line(boundary_px, fill=track_color, width=5)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, 'PNG')
    print(f"Saved: {output_path}")
    return True


if __name__ == "__main__":
    root_dir = "/Users/sam/git/Racelogic-tracks"

    # Test with a few tracks
    test_tracks = ["Croft", "Castle Combe", "Bathurst", "Oran Park Raceway", "Goodwood Festival of Speed"]

    for track in test_tracks:
        output = f"/Users/sam/git/Racelogic-track-converter/trackmaps/{track}.png"
        generate_track_image(track, root_dir, output)
