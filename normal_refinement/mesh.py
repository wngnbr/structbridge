from pathlib import Path

import numpy as np
import trimesh


def _as_mesh(mesh_or_scene):
    if isinstance(mesh_or_scene, trimesh.Scene):
        geometries = tuple(mesh_or_scene.geometry.values())
        if not geometries:
            raise ValueError("Scene contains no geometry")
        return trimesh.util.concatenate(geometries)
    return mesh_or_scene


def _remove_bad_and_duplicate_faces(vertices, faces, area_eps):
    valid = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    faces = faces[valid]
    if len(faces) == 0:
        return faces

    distinct = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    faces = faces[distinct]
    if len(faces) == 0:
        return faces

    tri = vertices[faces]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    faces = faces[area2 > area_eps]
    if len(faces) == 0:
        return faces

    sorted_faces = np.sort(faces, axis=1)
    _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
    unique_idx.sort()
    return faces[unique_idx]


def _remove_unreferenced(vertices, faces):
    used = np.zeros(len(vertices), dtype=bool)
    used[faces.reshape(-1)] = True
    index = np.full(len(vertices), -1, dtype=np.int64)
    index[used] = np.arange(used.sum())
    return vertices[used], index[faces]


def _face_areas(vertices, faces):
    tri = vertices[faces]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def _edge_lengths(vertices, faces):
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    return np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)


def _weld_and_clean(vertices, faces, weld_tol, area_eps):
    faces = _remove_bad_and_duplicate_faces(vertices, faces, area_eps)
    if len(faces) == 0:
        raise ValueError("Mesh has no valid faces after cleanup")

    quantized = np.round(vertices / weld_tol).astype(np.int64)
    _, first_idx, inverse, counts = np.unique(
        quantized, axis=0, return_index=True, return_inverse=True, return_counts=True
    )

    welded_vertices = vertices[first_idx].copy()
    if np.any(counts > 1):
        sums = np.zeros((len(first_idx), 3), dtype=np.float64)
        np.add.at(sums, inverse, vertices)
        welded_vertices = sums / counts[:, None]

    faces = inverse[faces]
    faces = _remove_bad_and_duplicate_faces(welded_vertices, faces, area_eps)
    if len(faces) == 0:
        raise ValueError("Mesh has no valid faces after vertex welding")
    welded_vertices, faces = _remove_unreferenced(welded_vertices, faces)
    return welded_vertices, faces


def _target_max_edge(vertices, faces, target_vertices):
    surface_area = float(_face_areas(vertices, faces).sum())
    if surface_area <= 0:
        return 0.0

    # For near-equilateral triangles, A = sqrt(3) / 4 * edge^2 and F ~= 2V.
    # This gives a conservative edge length for roughly target_vertices samples.
    target_edge = np.sqrt(2.0 * surface_area / (np.sqrt(3.0) * max(target_vertices, 1)))

    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if bbox_diag > 0:
        target_edge = min(target_edge, bbox_diag * 0.035)
    return float(target_edge)


def _subdivide_long_edges(vertices, faces, target_vertices, max_iter=8):
    if target_vertices <= 0:
        lengths = _edge_lengths(vertices, faces)
        max_edge = float(lengths.max()) if len(lengths) else 0.0
        p95_edge = float(np.percentile(lengths, 95)) if len(lengths) else 0.0
        return vertices, faces, {
            "target_edge_len": 0.0,
            "subdivided": False,
            "max_edge_before": max_edge,
            "p95_edge_before": p95_edge,
            "max_edge_after": max_edge,
            "p95_edge_after": p95_edge,
        }

    target_edge = _target_max_edge(vertices, faces, target_vertices)
    if target_edge <= 0:
        return vertices, faces, {"target_edge_len": target_edge, "subdivided": False}

    lengths = _edge_lengths(vertices, faces)
    max_before = float(lengths.max()) if len(lengths) else 0.0
    p95_before = float(np.percentile(lengths, 95)) if len(lengths) else 0.0

    should_subdivide = len(vertices) < target_vertices or max_before > target_edge * 1.5
    if not should_subdivide:
        return vertices, faces, {
            "target_edge_len": target_edge,
            "subdivided": False,
            "max_edge_before": max_before,
            "p95_edge_before": p95_before,
            "max_edge_after": max_before,
            "p95_edge_after": p95_before,
        }

    new_vertices, new_faces = trimesh.remesh.subdivide_to_size(
        vertices, faces, max_edge=target_edge, max_iter=max_iter
    )
    new_vertices = np.asarray(new_vertices, dtype=np.float64)
    new_faces = np.asarray(new_faces, dtype=np.int64)

    lengths_after = _edge_lengths(new_vertices, new_faces)
    return new_vertices, new_faces, {
        "target_edge_len": target_edge,
        "subdivided": True,
        "max_edge_before": max_before,
        "p95_edge_before": p95_before,
        "max_edge_after": float(lengths_after.max()) if len(lengths_after) else 0.0,
        "p95_edge_after": float(np.percentile(lengths_after, 95)) if len(lengths_after) else 0.0,
    }


def preprocess_mesh(input_path, output_path, target_vertices=3000, weld_tol_ratio=1e-7, min_weld_tol=1e-8):
    """Clean and lightly regularize a mesh for normal-driven optimization.

    The preprocessing is intentionally conservative: it welds coincident seams and
    removes invalid faces, then subdivides long edges/large faces to provide
    enough reasonably uniform vertices. It does not fill holes, run Poisson/SDF
    reconstruction, or decimate the input.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = _as_mesh(trimesh.load(str(input_path), force="mesh", process=False))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"Empty mesh: {input_path}")

    finite_vertices = np.isfinite(vertices).all(axis=1)
    if not finite_vertices.all():
        index = np.full(len(vertices), -1, dtype=np.int64)
        index[finite_vertices] = np.arange(finite_vertices.sum())
        valid_faces = finite_vertices[faces].all(axis=1)
        vertices = vertices[finite_vertices]
        faces = index[faces[valid_faces]]

    bbox_diag = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    weld_tol = max(float(bbox_diag) * weld_tol_ratio, min_weld_tol)
    area_eps = max(weld_tol * weld_tol * 1e-2, 1e-20)

    input_v, input_f = len(vertices), len(faces)
    vertices, faces = _weld_and_clean(vertices, faces, weld_tol, area_eps)
    clean_v, clean_f = len(vertices), len(faces)

    vertices, faces, subdiv_stats = _subdivide_long_edges(vertices, faces, target_vertices)
    faces = _remove_bad_and_duplicate_faces(vertices, faces, area_eps)
    vertices, faces = _remove_unreferenced(vertices, faces)

    clean = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    clean.export(str(output_path))

    stats = {
        "input_vertices": input_v,
        "input_faces": input_f,
        "clean_vertices": clean_v,
        "clean_faces": clean_f,
        "output_vertices": len(vertices),
        "output_faces": len(faces),
        "target_vertices": target_vertices,
        "weld_tol": weld_tol,
        "welded_vertices": input_v - clean_v,
        "removed_faces": input_f - clean_f,
    }
    stats.update(subdiv_stats)
    stats["added_vertices_by_subdivision"] = len(vertices) - clean_v
    stats["added_faces_by_subdivision"] = len(faces) - clean_f
    return output_path, stats


def sanitize_mesh(input_path, output_path, weld_tol_ratio=1e-7, min_weld_tol=1e-8):
    return preprocess_mesh(
        input_path,
        output_path,
        target_vertices=0,
        weld_tol_ratio=weld_tol_ratio,
        min_weld_tol=min_weld_tol,
    )
