import argparse
import time
import os

def simplify_mesh(input_path, output_path, target_faces=500000):
    try:
        import open3d as o3d
    except ImportError:
        print("Error: 'open3d' is not installed. Please run 'pip install open3d'")
        return

    if not os.path.exists(input_path):
        print(f"Error: Input file not found {input_path}")
        return

    print(f"[1/4] Loading mesh from {input_path}...")
    mesh = o3d.io.read_triangle_mesh(input_path)

    current_faces = len(mesh.triangles)
    print(f"      Current face count: {current_faces}")

    if current_faces <= target_faces:
        print(f"      Mesh face count is already below or equal to target ({target_faces}). Outputting as is.")
        o3d.io.write_triangle_mesh(output_path, mesh)
        return

    print(f"[2/4] Simplifying mesh to {target_faces} faces using Quadric Edge Collapse...")
    start_time = time.time()

    # 核心算法：二次误差边缘折叠法，完美保留尖锐特征和局部细节
    simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))

    end_time = time.time()
    final_faces = len(simplified_mesh.triangles)
    print(f"      Simplification done in {end_time - start_time:.2f} seconds.")
    print(f"      Final face count: {final_faces}")

    print("[3/4] Recomputing final vertex and triangle normals...")
    # 重新计算渲染法向量，确保网格平滑过渡不出错
    simplified_mesh.compute_vertex_normals()
    simplified_mesh.compute_triangle_normals()

    print(f"[4/4] Saving detailed simplified mesh to {output_path}...")

    # 写出文件，保持纹理数据或格式一致性（如果原图带顶点色则保留）
    o3d.io.write_triangle_mesh(output_path, simplified_mesh)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplify 3D mesh while preserving high-frequency details (Quadric Error Metric).")
    parser.add_argument("input_path", type=str, help="Input mesh file path (.obj, .ply, .glb)")
    parser.add_argument("output_path", type=str, help="Output mesh file path (.obj, .ply, .glb)")
    parser.add_argument("--target_faces", "-t", type=int, default=500000, help="Target triangle face count (default: 500000)")

    args = parser.parse_args()
    simplify_mesh(args.input_path, args.output_path, args.target_faces)