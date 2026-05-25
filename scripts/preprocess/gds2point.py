"""Convert GDSII layout files to 4D point cloud representations.

For each hotspot/non-hotspot marker window in the GDS file, this script:
  1. Boolean-clips Metal1 polygon shapes against the marker bounding box.
  2. Extracts vertices and computes outward-facing unit normal vectors.
  3. Produces fixed-size point sets via vertex anchoring + edge interpolation.
  4. Normalizes coordinates to [0, 1] and saves as .npy files.

Output format: each .npy file has shape [target_num, 4] with columns (x, y, nx, ny).

Usage:
    python gds2point.py --data_root ./data/ICCAD2012/raw --output_root ./data/ICCAD2012/points_4d

Expected GDS structure:
    <data_root>/<chip_name>/train.gds
    <data_root>/<chip_name>/test.gds

GDS layers (ICCAD 2012 convention):
    Layer 10/0: Metal1 polygon shapes
    Layer 21/0: Hotspot (HS) marker bounding boxes (1200x1200)
    Layer 23/0: Non-hotspot (NHS) marker bounding boxes (1200x1200)
"""

import klayout.db as db
import numpy as np
import os
import argparse
from tqdm import tqdm
import math

np.random.seed(42)


def get_vertices_and_perfect_fill_with_normals(polygons, target_num):
    """Vertex-anchored sampling with edge interpolation and normal vector computation.

    Sampling strategy:
      - If vertices >= target_num: random downsample.
      - If vertices < target_num: keep all vertices, fill remaining by interpolating
        along polygon edges (proportional to edge length), then pad/truncate.

    Args:
        polygons: list of KLayout polygon objects (clipped to a window).
        target_num: desired number of output points.

    Returns:
        np.ndarray of shape [target_num, 4] -> (x, y, nx, ny).
        Coordinates are in absolute GDSII integer DBU units.
        Normals are outward-facing unit vectors.
    """
    all_points_data = []
    edges_info = []
    total_perimeter = 0.0

    # --- 1. Extract geometry and compute normals ---
    for poly in polygons:
        hull = [[p.x, p.y] for p in poly.each_point_hull()]

        # Force closure check (KLayout hulls usually don't repeat first/last point)
        if len(hull) > 1 and hull[0] == hull[-1]:
            hull.pop()

        n_pts = len(hull)
        if n_pts < 3:
            continue  # Skip degenerate shapes

        # Pre-compute edge attributes
        poly_edge_normals = []
        poly_edge_lengths = []
        poly_edge_vecs = []

        for i in range(n_pts):
            p1 = hull[i]
            p2 = hull[(i + 1) % n_pts]  # Cyclic closure

            dx = float(p2[0] - p1[0])
            dy = float(p2[1] - p1[1])
            length = math.sqrt(dx * dx + dy * dy)

            if length == 0:
                poly_edge_normals.append(np.array([1.0, 0.0]))
                poly_edge_lengths.append(0.0)
                poly_edge_vecs.append(np.array([0.0, 0.0]))
                continue

            # Normal vector computation:
            # Assumes counter-clockwise (CCW) winding (GDSII standard).
            # Edge vector: (dx, dy)
            # Outward normal: (dy, -dx) normalized to unit length.
            nx = dy / length
            ny = -dx / length

            normal = np.array([nx, ny], dtype=np.float32)
            vec = np.array([dx, dy], dtype=np.float32)

            poly_edge_normals.append(normal)
            poly_edge_lengths.append(length)
            poly_edge_vecs.append(vec)

            # Record edge info for interpolation fill
            edges_info.append({
                'start': np.array(p1, dtype=np.float32),
                'vec': vec,
                'length': length,
                'normal': normal
            })
            total_perimeter += length

        # --- Extract vertices with averaged normals ---
        for i in range(n_pts):
            # Vertex i connects "previous edge i-1" and "current edge i"
            # poly_edge_normals[i] corresponds to p[i] -> p[i+1]
            prev_idx = (i - 1 + n_pts) % n_pts
            curr_idx = i

            n_prev = poly_edge_normals[prev_idx]
            n_curr = poly_edge_normals[curr_idx]

            # Vertex normal = average of adjacent edge normals (bisector direction)
            avg_nx = n_prev[0] + n_curr[0]
            avg_ny = n_prev[1] + n_curr[1]

            # Normalize
            mag = math.sqrt(avg_nx * avg_nx + avg_ny * avg_ny)
            if mag > 1e-6:
                avg_nx /= mag
                avg_ny /= mag

            all_points_data.append([float(hull[i][0]), float(hull[i][1]), avg_nx, avg_ny])

    points_arr = np.array(all_points_data, dtype=np.float32)
    num_vertices = len(points_arr)

    # Empty slice
    if num_vertices == 0:
        return np.zeros((target_num, 4), dtype=np.float32)

    final_points = None

    # --- 2. Sampling decision ---

    # Case A: Too many vertices -> random downsample
    if num_vertices >= target_num:
        choice_idx = np.random.choice(num_vertices, target_num, replace=False)
        final_points = points_arr[choice_idx]

    # Case B: Keep vertices, fill remaining space with edge interpolation
    else:
        final_points_list = [points_arr]
        points_to_insert = target_num - num_vertices

        if points_to_insert > 0 and total_perimeter > 0:
            inserted_data = []

            current_perimeter_sum = 0.0
            points_allocated_so_far = 0

            for edge in edges_info:
                current_perimeter_sum += edge['length']
                expected_total = (current_perimeter_sum / total_perimeter) * points_to_insert
                count_for_this_edge = int(round(expected_total)) - points_allocated_so_far

                if count_for_this_edge > 0:
                    # Interpolate along edge
                    t_values = np.linspace(0, 1, count_for_this_edge + 2)[1:-1]

                    # Edge normal is constant along the entire edge
                    edge_normal = edge['normal']

                    for t in t_values:
                        new_xy = edge['start'] + t * edge['vec']
                        inserted_data.append([new_xy[0], new_xy[1], edge_normal[0], edge_normal[1]])

                points_allocated_so_far += count_for_this_edge

            if len(inserted_data) > 0:
                final_points_list.append(np.array(inserted_data, dtype=np.float32))

        # Merge
        all_points = np.concatenate(final_points_list, axis=0)

        # Pad / truncate
        current_count = len(all_points)
        delta = target_num - current_count

        if delta > 0:
            pad_idx = np.random.choice(current_count, delta, replace=True)
            padding = all_points[pad_idx]
            final_points = np.vstack([all_points, padding])
        elif delta < 0:
            final_points = all_points[:target_num]
        else:
            final_points = all_points

    # --- 3. Shuffle ---
    np.random.shuffle(final_points)

    return final_points


def extract_and_save_pointcloud(window_boxes, polygons_region, save_dir,
                                 progress_label="", target_points=1024):
    """Extract and save point clouds for all marker windows.

    Args:
        window_boxes: sorted list of KLayout Box objects (marker bounding boxes).
        polygons_region: KLayout Region of Metal1 polygons.
        save_dir: output directory for .npy files.
        progress_label: "HS" or "NHS" (used in filenames).
        target_points: number of points per sample.
    """
    os.makedirs(save_dir, exist_ok=True)

    for i, bbox in enumerate(tqdm(window_boxes, desc="Processing " + progress_label)):
        # 1. Boolean clip (Region AND operation)
        clip_box_region = db.Region(bbox)
        clipped_shapes = polygons_region & clip_box_region
        polys_in_box = [shape for shape in clipped_shapes.each()]

        # 2. Vertex-anchored sampling (absolute coordinates)
        raw_points = get_vertices_and_perfect_fill_with_normals(polys_in_box, target_points)

        # 3. Normalize to [0, 1]
        if len(raw_points) > 0:
            w = bbox.width()
            h = bbox.height()

            if w == 0:
                w = 1.0
            if h == 0:
                h = 1.0

            # Translate to window origin
            raw_points[:, 0] -= bbox.left
            raw_points[:, 1] -= bbox.bottom

            # Scale to [0, 1]
            raw_points[:, 0] /= w
            raw_points[:, 1] /= h

        # 4. Save as .npy
        filename = f"{progress_label}_{i+1:05d}.npy"
        filepath = os.path.join(save_dir, filename)
        np.save(filepath, raw_points)


def clip_point_clouds(gds_file, save_dir, target_num=1024):
    """Process a single GDS file: extract point clouds for all markers.

    Args:
        gds_file: path to the GDS file.
        save_dir: output directory.
        target_num: number of points per sample.
    """
    ly = db.Layout()
    ly.read(gds_file)
    top_cell = ly.top_cell()

    # ICCAD 2012 layer definitions
    # Layer 10/0: Metal1 Shapes
    # Layer 21/0: Hotspot Markers
    # Layer 23/0: Non-Hotspot Markers
    polygon_layer = ly.layer(10, 0)
    hs_layer = ly.layer(21, 0)
    nhs_layer = ly.layer(23, 0)

    nhs_boxes = [shape.bbox() for shape in top_cell.shapes(nhs_layer)]
    hs_boxes = [shape.bbox() for shape in top_cell.shapes(hs_layer)]

    # Sort by (bottom, left) for deterministic processing order
    nhs_boxes.sort(key=lambda box: (box.bottom, box.left))
    hs_boxes.sort(key=lambda box: (box.bottom, box.left))

    # Build merged Region for efficient boolean operations
    polygons_region = db.Region(top_cell.shapes(polygon_layer))
    polygons_region.merge()

    print(f"File: {os.path.basename(gds_file)}")
    print(f" - Found {len(hs_boxes)} Hotspots")
    print(f" - Found {len(nhs_boxes)} Non-Hotspots")

    extract_and_save_pointcloud(nhs_boxes, polygons_region, save_dir, "NHS", target_points=target_num)
    extract_and_save_pointcloud(hs_boxes, polygons_region, save_dir, "HS", target_points=target_num)


def main():
    parser = argparse.ArgumentParser(
        description="Convert GDSII layouts to 4D point clouds (x, y, nx, ny)."
    )
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing chip subdirectories with GDS files.')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Output root directory for generated point clouds.')
    parser.add_argument('--target_points', type=int, default=1024,
                        help='Number of points per sample (default: 1024).')
    args = parser.parse_args()

    for chip in os.listdir(args.data_root):
        gds_dir = os.path.join(args.data_root, chip)
        if not os.path.isdir(gds_dir):
            continue

        save_dir = os.path.join(args.output_root, chip)
        train_dir = os.path.join(save_dir, "train")
        test_dir = os.path.join(save_dir, "test")

        train_gds = os.path.join(gds_dir, "train.gds")
        test_gds = os.path.join(gds_dir, "test.gds")

        if os.path.exists(train_gds):
            print(f"Processing Train: {chip}")
            clip_point_clouds(train_gds, train_dir, target_num=args.target_points)

        if os.path.exists(test_gds):
            print(f"Processing Test: {chip}")
            clip_point_clouds(test_gds, test_dir, target_num=args.target_points)


if __name__ == "__main__":
    main()
