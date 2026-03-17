import bpy
import os
import math
import numpy as np
import bmesh
import gc
import trimesh
import fpsample
import argparse  # 新增：命令行参数

def process_mesh(mesh_path, point_number, npz_output_path, sharpness_threshold):
    # 清空Blender场景，避免残留对象冲突
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # 导入OBJ模型
    bpy.ops.wm.obj_import(filepath=mesh_path)
    obj = bpy.context.selected_objects[0]

    # 选中锐边
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.edges_select_sharp(sharpness=sharpness_threshold)
    bpy.ops.object.mode_set(mode='OBJECT')

    # 获取锐边信息
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    sharp_edges = [edge for edge in bm.edges if edge.select]

    # 收集锐边数据
    sharp_edges_vertices = []
    link_normal1 = []
    link_normal2 = []
    vertices_set = set()
    
    for edge in sharp_edges:
        vertices_set.update(edge.verts[:])
        sharp_edges_vertices.append([edge.verts[0].index, edge.verts[1].index])
        normal1 = edge.link_faces[0].normal
        normal2 = edge.link_faces[1].normal
        link_normal1.append(normal1)
        link_normal2.append(normal2)

    # 顶点与法向量
    vertices = []
    vertices_normal = []
    for vertice in vertices_set:
        vertices.append(vertice.co)
        vertices_normal.append(vertice.normal)

    vertices = np.array(vertices)
    vertices_normal = np.array(vertices_normal)
    sharp_edges_count = len(sharp_edges)

    if sharp_edges_count > 0:
        sharp_edge_link_normal = np.concatenate([link_normal1, link_normal2], axis=1)
        sharp_edge_link_normal = np.nan_to_num(sharp_edge_link_normal)
        vertices_normal = np.nan_to_num(vertices_normal)

        # 加载模型用于采样
        mesh = trimesh.load(mesh_path, process=False)
        num_target_sharp_vertices = point_number // 2
        sharp_edges_vertices_pair = np.array(sharp_edges_vertices)
        sharp_vertices_pair = mesh.vertices[sharp_edges_vertices_pair]

        # 边法向量
        epsilon = 1e-4
        edge_normal = 0.5 * sharp_edge_link_normal[:, :3] + 0.5 * sharp_edge_link_normal[:, 3:]
        norms = np.linalg.norm(edge_normal, axis=1, keepdims=True)
        norms = np.where(norms > epsilon, norms, epsilon)
        edge_normal = edge_normal / norms

        # 锐边顶点集合
        known_vertices = np.concatenate([vertices, vertices_normal], axis=1)
        num_known_vertices = known_vertices.shape[0]

        # 插值补充顶点
        if num_known_vertices < num_target_sharp_vertices:
            num_new_vertices = num_target_sharp_vertices - num_known_vertices
            new_vertices = []
            if num_new_vertices > 0:
                rng = np.random.default_rng()
                ind = rng.choice(sharp_edges_count, num_new_vertices, replace=True)
                start_v = sharp_vertices_pair[ind, 0]
                end_v = sharp_vertices_pair[ind, 1]
                t = np.random.rand(num_new_vertices).reshape(-1, 1)
                new_pos = (1 - t) * start_v + t * end_v
                new_norm = 0.5 * sharp_edge_link_normal[ind, :3] + 0.5 * sharp_edge_link_normal[ind, 3:]
                new_norm /= np.linalg.norm(new_norm, axis=1, keepdims=True)
                new_vertices = np.concatenate([new_pos, new_norm], axis=1)
            target_vertices = np.concatenate([new_vertices, known_vertices], axis=0)
        else:
            target_vertices = known_vertices

        # 采样与FPS
        sharp_surface = target_vertices
        sharp_surface_points = sharp_surface[:, :3]

        # 表面粗采样
        coarse_surface_points, faces = mesh.sample(200000, return_index=True)
        normals = mesh.face_normals[faces]
        coarse_surface = np.concatenate([coarse_surface_points, normals], axis=1)

        # FPS采样
        kd_idx = fpsample.bucket_fps_kdline_sampling(coarse_surface_points, num_target_sharp_vertices, h=5)
        fps_coarse = coarse_surface[kd_idx].reshape(-1, 1, 6)

        if sharp_surface.shape[0] > num_target_sharp_vertices:
            kd_idx_sharp = fpsample.bucket_fps_kdline_sampling(sharp_surface_points, num_target_sharp_vertices, h=5)
            fps_sharp = sharp_surface[kd_idx_sharp].reshape(-1, 1, 6)
        else:
            fps_sharp = sharp_surface[:, None]

        # 清理异常值
        fps_sharp = np.nan_to_num(fps_sharp, nan=1, posinf=1, neginf=1)
        fps_coarse = np.nan_to_num(fps_coarse, nan=1, posinf=1, neginf=1)

        # 只保存NPZ
        np.savez(
            npz_output_path,
            fps_sharp_surface=fps_sharp.astype(np.float32),
            fps_coarse_surface=fps_coarse.astype(np.float32),
        )
        print(f"✅ 处理完成！NPZ已保存：\n{npz_output_path}")
    else:
        print("❌ 未检测到锐边，不生成NPZ")

    # 内存清理
    bm.free()
    bpy.data.objects.remove(obj, do_unlink=True)
    gc.collect()

# ====================== 命令行参数主程序 ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OBJ锐边采样生成NPZ工具")
    parser.add_argument("--input", "-i", required=True, type=str, help="输入OBJ文件路径")
    parser.add_argument("--output", "-o", required=True, type=str, help="输出NPZ文件路径")
    parser.add_argument("--angle", "-a", type=int, default=30, help="锐边角度阈值(度)，默认30")
    parser.add_argument("--points", "-p", type=int, default=2048, help="采样点数量，默认2048")

    args = parser.parse_args()

    INPUT_OBJ_PATH = args.input
    OUTPUT_NPZ_PATH = args.output
    ANGLE_THRESHOLD = args.angle
    POINT_NUMBER = args.points

    if not os.path.exists(INPUT_OBJ_PATH):
        print(f"❌ 文件不存在：{INPUT_OBJ_PATH}")
    else:
        os.makedirs(os.path.dirname(OUTPUT_NPZ_PATH), exist_ok=True)
        sharpness_threshold = math.radians(ANGLE_THRESHOLD)
        try:
            process_mesh(
                mesh_path=INPUT_OBJ_PATH,
                point_number=POINT_NUMBER,
                npz_output_path=OUTPUT_NPZ_PATH,
                sharpness_threshold=sharpness_threshold
            )
        except Exception as e:
            print(f"❌ 出错：{e}")