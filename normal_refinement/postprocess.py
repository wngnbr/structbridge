import argparse
import open3d as o3d
import os
import trimesh
import numpy as np

def smooth_taubin(mesh_path, out_path, iterations):
    print(f"Loading mesh for Taubin smoothing: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    if not mesh.has_vertices():
        raise ValueError(f"Cannot smooth an empty or unreadable mesh: {mesh_path}")

    mesh.compute_vertex_normals()
    print(f"Applying Taubin smoothing with {iterations} iterations...")
    # Taubin smoothing parameters:
    # lambda_filter (positive, inward) and mu (negative, outward)
    # usually lambda = 0.5, mu = -0.53 are good defaults to preserve volume.
    mesh_out = mesh.filter_smooth_taubin(number_of_iterations=iterations)
    mesh_out.compute_vertex_normals()

    if not o3d.io.write_triangle_mesh(out_path, mesh_out):
        raise OSError(f"Failed to write smoothed mesh: {out_path}")
    print(f"Saved smoothed mesh to: {out_path}")

def smooth_laplacian(mesh_path, out_path, iterations):
    print(f"Loading mesh for Laplacian smoothing: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    if not mesh.has_vertices():
        print("Failed to load mesh or mesh is empty.")
        return

    mesh.compute_vertex_normals()
    print(f"Applying Laplacian smoothing with {iterations} iterations...")
    # Laplacian shrinks the mesh, only use for few iterations!
    mesh_out = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
    mesh_out.compute_vertex_normals()

    o3d.io.write_triangle_mesh(out_path, mesh_out)
    print(f"Saved smoothed mesh to: {out_path}")

def smooth_bilateral(mesh_path, out_path, iterations, sigma_length, sigma_angle):
    print(f"Loading mesh for Bilateral smoothing: {mesh_path}")
    mesh = trimesh.load(mesh_path, process=False)

    if not hasattr(mesh, 'vertices'):
         print("Failed to load mesh.")
         return

    print(f"Applying Bilateral smoothing with {iterations} iterations...")

    for it in range(iterations):
        vertices = mesh.vertices.copy()
        normals = mesh.vertex_normals.copy()
        new_vertices = np.zeros_like(vertices)

        for i, v in enumerate(vertices):
            neighbors = mesh.vertex_neighbors[i]
            if len(neighbors) == 0:
                new_vertices[i] = v
                continue

            neighbor_verts = vertices[neighbors]
            _neighbor_normals = normals[neighbors]

            distances = np.linalg.norm(neighbor_verts - v, axis=1)
            w_spatial = np.exp(-(distances ** 2) / (2 * (sigma_length ** 2)))

            offsets = neighbor_verts - v
            t = np.sum(offsets * normals[i], axis=1)
            w_normal = np.exp(-(t ** 2) / (2 * (sigma_angle ** 2)))

            weights = w_spatial * w_normal
            weights_sum = np.sum(weights)

            if weights_sum > 1e-6:
                shift = np.sum(weights * t) / weights_sum
                new_vertices[i] = v + normals[i] * shift
            else:
                new_vertices[i] = v

        mesh.vertices = new_vertices
        print(f"  Iteration {it+1}/{iterations} completed.")

    mesh.export(out_path)
    print(f"Saved smoothed mesh to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh Smoothing Post-processing")
    parser.add_argument("--input", type=str, required=True, help="Path to input optimized mesh (e.g. result.obj)")
    parser.add_argument("--output", type=str, required=True, help="Path to output smoothed mesh")
    parser.add_argument("--method", type=str, choices=["taubin", "laplacian", "bilateral"], default="taubin", help="Smoothing method")
    parser.add_argument("--iter", type=int, default=10, help="Number of smoothing iterations")
    parser.add_argument("--sigma_length", type=float, default=0.01, help="Spatial sigma for bilateral filter (relative to edge length)")
    parser.add_argument("--sigma_angle", type=float, default=0.005, help="Normal/Influence sigma for bilateral filter")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        print(f"Error: Input file does not exist: {input_path}")
        exit(1)

    if args.method == "taubin":
        smooth_taubin(input_path, output_path, args.iter)
    elif args.method == "laplacian":
        smooth_laplacian(input_path, output_path, args.iter)
    elif args.method == "bilateral":
        smooth_bilateral(input_path, output_path, args.iter, args.sigma_length, args.sigma_angle)
