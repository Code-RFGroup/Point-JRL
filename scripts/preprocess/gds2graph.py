"""Convert GDSII layout files to graph representations for GNN-based hotspot detection.

For each marker window, this script:
  1. Boolean-clips Metal1 polygons against the marker bounding box.
  2. Decomposes each clipped polygon into rectangles (Manhattan decomposition).
  3. Each rectangle becomes a node with features [x1, y1, x2, y2] normalized to [0, 1].
  4. Builds two types of edges:
     - Internal edges (type 0): between rectangles from the same polygon (distance = 0).
     - External edges (type 1): between rectangles from different polygons within a
       distance threshold (default 65nm), with normalized distance as edge attribute.
  5. Saves as PyTorch Geometric Data objects (.pt files).

Usage:
    python gds2graph.py --data_root ./data/ICCAD2012/raw --output_root ./data/ICCAD2012/graphs

Expected GDS structure:
    <data_root>/<chip_name>/train.gds
    <data_root>/<chip_name>/test.gds

GDS layers (ICCAD 2012 convention):
    Layer 10/0: Metal1 polygon shapes
    Layer 21/0: Hotspot (HS) marker bounding boxes (1200x1200)
    Layer 23/0: Non-hotspot (NHS) marker bounding boxes (1200x1200)
"""

import klayout.db as db
import torch
import os
import math
import argparse
from tqdm import tqdm
from torch_geometric.data import Data


def compute_rect_distance(box_a, box_b):
    """Compute Euclidean distance between two axis-aligned rectangles.
    Returns 0 if the rectangles overlap.
    """
    delta_x = max(0, max(box_a.left, box_b.left) - min(box_a.right, box_b.right))
    delta_y = max(0, max(box_a.bottom, box_b.bottom) - min(box_a.top, box_b.top))
    return math.sqrt(delta_x**2 + delta_y**2)


def process_clip_to_graph(clip_box, polygons_region, threshold_nm, dbu):
    """Convert a single marker window to a GNN graph.

    Args:
        clip_box: KLayout Box defining the marker window.
        polygons_region: KLayout Region of Metal1 polygons.
        threshold_nm: distance threshold in nanometers for external edges.
        dbu: database unit (meters per GDSII unit).

    Returns:
        PyTorch Geometric Data object, or None if no polygons found.
    """
    # 1. Boolean clip
    search_region = db.Region(clip_box)
    clipped_polys = polygons_region & search_region

    # 2. Polygon decomposition into rectangles
    rect_nodes = []
    poly_id_counter = 0

    for poly in clipped_polys.each():
        simple_shapes = poly.decompose_trapezoids()
        for simple_shape in simple_shapes:
            box = simple_shape.bbox()
            rect_nodes.append({'box': box, 'poly_id': poly_id_counter})
        poly_id_counter += 1

    num_nodes = len(rect_nodes)
    if num_nodes == 0:
        return None

    # Sort nodes by (bottom, left) for deterministic graph structure
    rect_nodes.sort(key=lambda item: (item['box'].bottom, item['box'].left))

    # 3. Build node features [x1, y1, x2, y2] normalized to [0, 1]
    w_scale = clip_box.width()
    h_scale = clip_box.height()
    if w_scale == 0:
        w_scale = 1.0
    if h_scale == 0:
        h_scale = 1.0

    node_feats = []
    for item in rect_nodes:
        b = item['box']
        x1 = (b.left - clip_box.left) / w_scale
        y1 = (b.bottom - clip_box.bottom) / h_scale
        x2 = (b.right - clip_box.left) / w_scale
        y2 = (b.top - clip_box.bottom) / h_scale
        node_feats.append([x1, y1, x2, y2])

    x = torch.tensor(node_feats, dtype=torch.float)

    # 4. Build edges
    edge_indices = []
    edge_attrs = []
    edge_types = []  # 0: Internal, 1: External

    threshold_dbu = threshold_nm / 1000.0 / dbu

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue

            node_i = rect_nodes[i]
            node_j = rect_nodes[j]

            # Internal edge (same polygon)
            if node_i['poly_id'] == node_j['poly_id']:
                edge_indices.append([i, j])
                edge_attrs.append([0.0])
                edge_types.append([0])

            # External edge (different polygon, distance < threshold)
            else:
                dist = compute_rect_distance(node_i['box'], node_j['box'])
                if dist < threshold_dbu:
                    norm_dist = dist / w_scale
                    edge_indices.append([i, j])
                    edge_attrs.append([norm_dist])
                    edge_types.append([1])

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        edge_type = torch.empty((0, 1), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        edge_type = torch.tensor(edge_types, dtype=torch.long)

    # 5. Assemble Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.edge_type = edge_type
    return data


def extract_and_save_graph(window_boxes, polygons_region, save_dir,
                           progress_label, dbu, threshold_nm=65):
    """Extract and save graphs for all marker windows.

    Args:
        window_boxes: sorted list of KLayout Box objects (marker bounding boxes).
        polygons_region: KLayout Region of Metal1 polygons.
        save_dir: output directory for .pt files.
        progress_label: "HS" or "NHS" (used in filenames and labels).
        dbu: database unit (meters per GDSII unit).
        threshold_nm: distance threshold for external edges (default: 65nm).
    """
    os.makedirs(save_dir, exist_ok=True)

    for i, bbox in enumerate(tqdm(window_boxes, desc="Processing " + progress_label)):
        graph_data = process_clip_to_graph(bbox, polygons_region, threshold_nm, dbu)

        filename = f"{progress_label}_{i+1:05d}.pt"
        filepath = os.path.join(save_dir, filename)

        if graph_data is not None:
            y_val = 1 if progress_label == "HS" else 0
            graph_data.y = torch.tensor([y_val], dtype=torch.long)
            torch.save(graph_data, filepath)
        else:
            # Save a dummy graph as placeholder to maintain index alignment
            # with point cloud files (which also save zero-filled arrays for empty clips)
            y_val = 1 if progress_label == "HS" else 0
            dummy_x = torch.zeros((1, 4), dtype=torch.float)
            dummy_edge_index = torch.empty((2, 0), dtype=torch.long)
            dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index)
            dummy_data.edge_attr = torch.empty((0, 1), dtype=torch.float)
            dummy_data.edge_type = torch.empty((0, 1), dtype=torch.long)
            dummy_data.y = torch.tensor([y_val], dtype=torch.long)
            torch.save(dummy_data, filepath)


def clip_graphs(gds_file, save_dir, threshold_nm=65):
    """Process a single GDS file: extract graphs for all markers.

    Args:
        gds_file: path to the GDS file.
        save_dir: output directory.
        threshold_nm: distance threshold for external edges (default: 65nm).
    """
    ly = db.Layout()
    ly.read(gds_file)
    top_cell = ly.top_cell()
    dbu = ly.dbu

    # ICCAD 2012 layer definitions
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

    extract_and_save_graph(nhs_boxes, polygons_region, save_dir, "NHS", dbu, threshold_nm)
    extract_and_save_graph(hs_boxes, polygons_region, save_dir, "HS", dbu, threshold_nm)


def main():
    parser = argparse.ArgumentParser(
        description="Convert GDSII layouts to PyTorch Geometric graph representations."
    )
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing chip subdirectories with GDS files.')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Output root directory for generated graph .pt files.')
    parser.add_argument('--threshold_nm', type=float, default=65.0,
                        help='Distance threshold in nm for external edges (default: 65).')
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
            clip_graphs(train_gds, train_dir, threshold_nm=args.threshold_nm)

        if os.path.exists(test_gds):
            print(f"Processing Test: {chip}")
            clip_graphs(test_gds, test_dir, threshold_nm=args.threshold_nm)


if __name__ == "__main__":
    main()
