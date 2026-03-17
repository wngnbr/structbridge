import cubvh
import torch
import numpy as np
import trimesh
from diso import DiffDMC
import os
import argparse  # 用于命令行参数

def generate_dense_grid_points(
    bbox_min = np.array((-1.05, -1.05, -1.05)),
    bbox_max= np.array((1.05, 1.05, 1.05)),
    resolution = 512,
    indexing = "ij"
):
    """生成3D密集网格点（保留核心功能）"""
    x = np.linspace(bbox_min[0], bbox_max[0], resolution + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], resolution + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], resolution + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    xyz = xyz.reshape(-1, 3)
    grid_size = [resolution + 1, resolution + 1, resolution + 1]
    return xyz, grid_size

def process_single_glb(glb_path, output_obj_path, resolution=512):
    """
    处理单个glb文件并保存为obj
    :param glb_path: 输入glb文件路径
    :param output_obj_path: 输出obj文件路径
    :param resolution: 网格分辨率（默认512）
    """
    # 1. 生成网格点（GPU加速）
    grid_xyz, grid_size = generate_dense_grid_points(resolution=resolution)
    grid_xyz = torch.FloatTensor(grid_xyz).cuda()
    
    # 2. 加载并标准化模型
    mesh = trimesh.load(glb_path, force='mesh')
    vertices = mesh.vertices
    
    # 标准化到[-1, 1]立方体范围
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) / 2
    scale = 2.0 / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    
    # 3. 计算UDF（无符号距离场）
    f = cubvh.cuBVH(
        torch.as_tensor(vertices, dtype=torch.float32, device='cuda'),
        torch.as_tensor(mesh.faces, dtype=torch.float32, device='cuda')
    )
    grid_udf, _, _ = f.unsigned_distance(grid_xyz, return_uvw=False)
    grid_udf = grid_udf.view((grid_size[0], grid_size[1], grid_size[2]))
    
    # 4. DiffDMC重构网格
    eps = 2 / resolution
    diffdmc = DiffDMC(dtype=torch.float32).cuda()
    new_vertices, new_faces = diffdmc(grid_udf, isovalue=eps, normalize=False)
    
    # 5. 坐标还原并保存
    bbox_min = np.array((-1.05, -1.05, -1.05))
    bbox_size = 2.1  # 1.05 - (-1.05) = 2.1
    new_vertices = (new_vertices + 1) / grid_size[0] * bbox_size + bbox_min[0]
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
    
    # 保存为OBJ文件
    new_mesh = trimesh.Trimesh(
        vertices=new_vertices.cpu().numpy(),
        faces=new_faces.cpu().numpy()
    )
    new_mesh.export(output_obj_path)
    print(f"处理完成！模型已保存至: {output_obj_path}")

# 主程序：命令行参数版本
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLB转OBJ 3D网格重建工具")
    # 必选参数：输入GLB路径
    parser.add_argument("--input", "-i", required=True, type=str, help="输入GLB文件路径")
    # 必选参数：输出OBJ路径
    parser.add_argument("--output", "-o", required=True, type=str, help="输出OBJ文件路径")
    # 可选参数：分辨率
    parser.add_argument("--resolution", "-r", type=int, default=512, help="网格分辨率，默认512")

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误：输入文件不存在 -> {args.input}")
    else:
        try:
            process_single_glb(args.input, args.output, args.resolution)
        except Exception as e:
            print(f"处理出错：{e}")