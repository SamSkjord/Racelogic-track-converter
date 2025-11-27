"""
Boundary processing for RaceLogic track data.

Handles:
- Detecting and splitting concatenated boundaries
- Removing crossing artifacts
- Closing boundary loops
- Generating centerlines
"""

import math

from tracks_db import haversine_distance

# =============================================================================
# Constants
# =============================================================================

BOUNDARY_SPLIT_THRESHOLD_M = 20.0  # Max distance to consider as "returning to start"
MIN_BOUNDARY_POINTS = 50  # Minimum points per boundary for dual detection
CROSSING_ANGLE_THRESHOLD_DEG = 50  # Angle change indicating crossing artifact
MAX_LOOP_GAP_METERS = 50  # Maximum gap to auto-close


def log_message(message):
    """Logging stub - replaced at runtime by main module."""
    pass


def set_logger(logger_func):
    """Set the logging function to use."""
    global log_message
    log_message = logger_func


# =============================================================================
# Boundary Loop Handling
# =============================================================================

def close_boundary_loop(boundary, max_gap_meters=MAX_LOOP_GAP_METERS):
    """
    Ensure a boundary forms a closed loop by connecting end to start if needed.

    Args:
        boundary: list of points
        max_gap_meters: maximum gap to close (don't close if gap is too large)

    Returns:
        Boundary with loop closed (last point connects to first)
    """
    if len(boundary) < 3:
        return boundary

    first = boundary[0]
    last = boundary[-1]

    gap = haversine_distance(first['lat'], first['long'], last['lat'], last['long'])

    if gap > 1 and gap < max_gap_meters:
        boundary = boundary + [first.copy()]
        log_message(f"Closed boundary loop (gap was {gap:.1f}m)")

    return boundary


def remove_crossing_artifacts(boundary, position='start', angle_threshold=CROSSING_ANGLE_THRESHOLD_DEG):
    """
    Remove crossing artifacts from a boundary.

    These are points where the path makes a sharp turn (near 90 degrees)
    indicating the crossing from one side of the track to the other.

    Args:
        boundary: list of points
        position: 'start' or 'end' - where to look for artifacts
        angle_threshold: minimum angle change (degrees) to consider as crossing artifact

    Returns:
        Cleaned boundary with crossing artifacts removed
    """
    if len(boundary) < 10:
        return boundary

    def calc_angle_change(p_prev, p_curr, p_next):
        v1_lat = p_curr['lat'] - p_prev['lat']
        v1_lon = p_curr['long'] - p_prev['long']
        v2_lat = p_next['lat'] - p_curr['lat']
        v2_lon = p_next['long'] - p_curr['long']

        dot = v1_lat * v2_lat + v1_lon * v2_lon
        mag1 = math.sqrt(v1_lat**2 + v1_lon**2)
        mag2 = math.sqrt(v2_lat**2 + v2_lon**2)

        if mag1 < 0.0000001 or mag2 < 0.0000001:
            return 0

        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    search_count = min(20, len(boundary) // 10)

    if position == 'start':
        trim_to = 0
        for i in range(1, min(search_count, len(boundary) - 1)):
            angle = calc_angle_change(boundary[i-1], boundary[i], boundary[i+1])
            if angle > angle_threshold:
                trim_to = i + 1
                log_message(f"Found crossing artifact at start, index {i}, angle {angle:.1f}°")

        if trim_to > 0:
            return boundary[trim_to:]

    elif position == 'end':
        trim_from = len(boundary)
        for i in range(len(boundary) - 2, max(len(boundary) - search_count, 0), -1):
            angle = calc_angle_change(boundary[i-1], boundary[i], boundary[i+1])
            if angle > angle_threshold:
                trim_from = i
                log_message(f"Found crossing artifact at end, index {i}, angle {angle:.1f}°")

        if trim_from < len(boundary):
            return boundary[:trim_from]

    return boundary


# =============================================================================
# Boundary Detection & Splitting
# =============================================================================

def detect_and_split_boundaries(parsed_data, threshold_meters=BOUNDARY_SPLIT_THRESHOLD_M, min_boundary_points=MIN_BOUNDARY_POINTS):
    """
    Detect if track data contains two concatenated boundaries and split them.

    RaceLogic tracks often contain inner and outer boundaries stored as one
    continuous path that traces one boundary, crosses to the other side,
    and traces back.

    Returns:
        dict with keys:
        - 'type': 'single' or 'dual'
        - 'boundary1': list of points (outer boundary or single path)
        - 'boundary2': list of points (inner boundary) or None
        - 'split_index': index where split occurred or None
    """
    if len(parsed_data) < min_boundary_points * 2:
        log_message(f"Track has {len(parsed_data)} points, treating as single boundary")
        return {
            'type': 'single',
            'boundary1': parsed_data,
            'boundary2': None,
            'split_index': None
        }

    start_lat = parsed_data[0]['lat']
    start_lon = parsed_data[0]['long']

    # Search for a point that returns close to start (not at the very end)
    search_start = int(len(parsed_data) * 0.1)
    search_end = int(len(parsed_data) * 0.9)

    best_match_idx = None
    best_match_dist = float('inf')

    for i in range(search_start, search_end):
        dist = haversine_distance(
            start_lat, start_lon,
            parsed_data[i]['lat'], parsed_data[i]['long']
        )

        if dist < threshold_meters and dist < best_match_dist:
            best_match_dist = dist
            best_match_idx = i

    if best_match_idx is not None:
        log_message(f"Detected dual boundaries: split at index {best_match_idx} "
                   f"(distance to start: {best_match_dist:.2f}m)")

        boundary1 = parsed_data[:best_match_idx + 1]
        boundary2 = parsed_data[best_match_idx:]

        # Clean up crossing artifacts
        boundary1 = remove_crossing_artifacts(boundary1, position='end')
        boundary2 = remove_crossing_artifacts(boundary2, position='start')

        # Close the boundary loops
        boundary1 = close_boundary_loop(boundary1)
        boundary2 = close_boundary_loop(boundary2)

        log_message(f"Split boundaries: boundary1={len(boundary1)} pts, "
                   f"boundary2={len(boundary2)} pts")

        return {
            'type': 'dual',
            'boundary1': boundary1,
            'boundary2': boundary2,
            'split_index': best_match_idx
        }
    else:
        log_message("No boundary split detected, treating as single boundary")
        return {
            'type': 'single',
            'boundary1': parsed_data,
            'boundary2': None,
            'split_index': None
        }


# =============================================================================
# Centerline Generation
# =============================================================================

def resample_boundary(boundary_points, num_points):
    """
    Resample a boundary to have a specific number of equally-spaced points.
    Uses linear interpolation along the path.
    """
    if len(boundary_points) < 2:
        return boundary_points

    # Calculate cumulative distance along the boundary
    distances = [0.0]
    for i in range(1, len(boundary_points)):
        d = haversine_distance(
            boundary_points[i-1]['lat'], boundary_points[i-1]['long'],
            boundary_points[i]['lat'], boundary_points[i]['long']
        )
        distances.append(distances[-1] + d)

    total_distance = distances[-1]
    if total_distance < 1:
        return boundary_points

    # Generate equally-spaced target distances
    target_distances = [total_distance * i / (num_points - 1) for i in range(num_points)]

    # Interpolate points at target distances
    resampled = []
    boundary_idx = 0

    for target_dist in target_distances:
        while boundary_idx < len(distances) - 1 and distances[boundary_idx + 1] < target_dist:
            boundary_idx += 1

        if boundary_idx >= len(boundary_points) - 1:
            resampled.append(boundary_points[-1].copy())
        else:
            seg_start_dist = distances[boundary_idx]
            seg_end_dist = distances[boundary_idx + 1]
            seg_length = seg_end_dist - seg_start_dist

            if seg_length < 0.001:
                t = 0
            else:
                t = (target_dist - seg_start_dist) / seg_length

            p1 = boundary_points[boundary_idx]
            p2 = boundary_points[boundary_idx + 1]

            resampled.append({
                'lat': p1['lat'] + t * (p2['lat'] - p1['lat']),
                'long': p1['long'] + t * (p2['long'] - p1['long']),
                'height': p1.get('height', 0) + t * (p2.get('height', 0) - p1.get('height', 0))
            })

    return resampled


def generate_centerline(boundary1, boundary2, num_points=200):
    """
    Generate a centerline by averaging corresponding points from two boundaries.

    Both boundaries are resampled to have the same number of points.
    Auto-detects and corrects if boundaries are traced in opposite directions.
    """
    log_message(f"Generating centerline from {len(boundary1)} and {len(boundary2)} boundary points")

    # Check if boundaries are traced in opposite directions
    sample_pts = 5
    b1_sample = resample_boundary(boundary1, sample_pts)
    b2_sample = resample_boundary(boundary2, sample_pts)
    b2_rev_sample = resample_boundary(list(reversed(boundary2)), sample_pts)

    avg_normal = sum(
        haversine_distance(b1_sample[i]['lat'], b1_sample[i]['long'],
                          b2_sample[i]['lat'], b2_sample[i]['long'])
        for i in range(sample_pts)
    ) / sample_pts

    avg_reversed = sum(
        haversine_distance(b1_sample[i]['lat'], b1_sample[i]['long'],
                          b2_rev_sample[i]['lat'], b2_rev_sample[i]['long'])
        for i in range(sample_pts)
    ) / sample_pts

    if avg_reversed < avg_normal:
        log_message(f"Boundaries traced in opposite directions - reversing boundary2")
        boundary2 = list(reversed(boundary2))

    # Resample both boundaries
    b1_resampled = resample_boundary(boundary1, num_points)
    b2_resampled = resample_boundary(boundary2, num_points)

    # Calculate distances between corresponding points
    distances = []
    for i in range(num_points):
        d = haversine_distance(
            b1_resampled[i]['lat'], b1_resampled[i]['long'],
            b2_resampled[i]['lat'], b2_resampled[i]['long']
        )
        distances.append(d)

    # Find median distance (typical track width)
    sorted_distances = sorted(distances)
    median_width = sorted_distances[len(sorted_distances) // 2]
    min_valid_width = median_width * 0.3

    log_message(f"Track width - median: {median_width:.1f}m, min valid: {min_valid_width:.1f}m")

    # Generate centerline with interpolation across crossing artifacts
    centerline = []
    for i in range(num_points):
        p1 = b1_resampled[i]
        p2 = b2_resampled[i]

        if distances[i] >= min_valid_width:
            centerline.append({
                'lat': (p1['lat'] + p2['lat']) / 2,
                'long': (p1['long'] + p2['long']) / 2,
                'height': (p1.get('height', 0) + p2.get('height', 0)) / 2
            })
        else:
            # Crossing artifact - interpolate from nearest valid points
            prev_valid = next_valid = None

            for j in range(i - 1, -1, -1):
                if distances[j] >= min_valid_width:
                    prev_valid = j
                    break

            for j in range(i + 1, num_points):
                if distances[j] >= min_valid_width:
                    next_valid = j
                    break

            if prev_valid is None:
                for j in range(num_points - 1, i, -1):
                    if distances[j] >= min_valid_width:
                        prev_valid = j - num_points
                        break

            if next_valid is None:
                for j in range(0, i):
                    if distances[j] >= min_valid_width:
                        next_valid = j + num_points
                        break

            if prev_valid is not None and next_valid is not None:
                prev_idx = prev_valid % num_points
                next_idx = next_valid % num_points

                span = next_valid - prev_valid
                t = (i - prev_valid) / span if span != 0 else 0.5

                prev_center_lat = (b1_resampled[prev_idx]['lat'] + b2_resampled[prev_idx]['lat']) / 2
                prev_center_lon = (b1_resampled[prev_idx]['long'] + b2_resampled[prev_idx]['long']) / 2
                next_center_lat = (b1_resampled[next_idx]['lat'] + b2_resampled[next_idx]['lat']) / 2
                next_center_lon = (b1_resampled[next_idx]['long'] + b2_resampled[next_idx]['long']) / 2

                centerline.append({
                    'lat': prev_center_lat + t * (next_center_lat - prev_center_lat),
                    'long': prev_center_lon + t * (next_center_lon - prev_center_lon),
                    'height': (p1.get('height', 0) + p2.get('height', 0)) / 2
                })
            else:
                centerline.append({
                    'lat': (p1['lat'] + p2['lat']) / 2,
                    'long': (p1['long'] + p2['long']) / 2,
                    'height': (p1.get('height', 0) + p2.get('height', 0)) / 2
                })

    log_message(f"Generated centerline with {len(centerline)} points")
    return centerline
