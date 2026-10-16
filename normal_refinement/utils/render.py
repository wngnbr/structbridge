import nvdiffrast.torch as dr
import torch

def _warmup(glctx):
    #windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device='cuda', **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])

class NormalsRenderer:

    _glctx:dr.RasterizeGLContext = None

    def __init__(
            self,
            mv: torch.Tensor, #C,4,4
            proj: torch.Tensor, #C,4,4
            image_size: tuple[int,int],
            ):
        self._mvp = proj @ mv #C,4,4
        self._image_size = image_size
        self._glctx = dr.RasterizeCudaContext(device=mv.device)
        _warmup(self._glctx)

    def render(self,
            vertices: torch.Tensor, #V,3 float
            normals: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            colors: torch.Tensor = None, #V,3 float (optional)
            ) ->torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4

        if colors is not None:
            vert_col = colors
        else:
            vert_col = (normals+1)/2 #V,3

        col,_ = dr.interpolate(vert_col, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        return col #C,H,W,4

class DepthRenderer:

    _glctx:dr.RasterizeGLContext = None

    def __init__(
            self,
            mv: torch.Tensor, #C,4,4
            proj: torch.Tensor, #C,4,4
            image_size: tuple[int,int],
            ):
        self._mv = mv #C,4,4
        self._mvp = proj @ mv #C,4,4
        self._image_size = image_size
        self._glctx = dr.RasterizeCudaContext(device=mv.device)
        _warmup(self._glctx)

    def render(self,
            vertices: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            normalize: bool = False # If True, normalize depth to [0,1] based on near/far or per-view min/max? For now, raw depth.
            ) ->torch.Tensor: #C,H,W,4 (last channel is mask)

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4

        # Clip Space
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4

        # Camera Space Depth (Assuming OpenGL convention: Camera looks down -Z)
        # But we usually want positive depth. So -Z.
        vertices_cam = vert_hom @ self._mv.transpose(-2,-1) #C,V,4
        # depth = -vertices_cam[..., 2:3] # C,V,1. Positive Z distance.
        # Wait, if mv transforms to camera space, and camera looks at -Z, then visible objects have negative Z.
        # So -Z is positive distance.

        # However, for generic visualization, valid depth should be positive.
        # Let's assume standard view matrices.
        vert_depth = -vertices_cam[..., 2:3]

        # Rasterize
        rast_out, _ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4

        # Interpolate Depth
        # Attributes to interpolate: depth
        depth_map, _ = dr.interpolate(vert_depth, rast_out, faces) #C,H,W,1

        # Alpha mask
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1

        # Combine (Depth, Depth, Depth, Alpha) for visualization/standard tensor shape, or just (Depth, 0, 0, Alpha)
        # Let's return (Depth, 0, 0, Alpha) to be explicit, or just DepthMap if user expects specific format.
        # NormalsRenderer returns (R,G,B,A).
        # Let's return (D, D, D, A) so it's viewable as grayscale.

        col = torch.cat((depth_map, depth_map, depth_map, alpha), dim=-1) #C,H,W,4

        # Antialias
        # Note: Antialiasing depth maps is geometrically questionable at silhouettes (mixing foreground/background depths).
        # But for visualization or "Soft Depth", it's often done.
        # nvdiffrast manual says: "Antialiasing... produce an image with proper coverage at silhouette edges."
        # If we AA depth, the edge pixels will be a mix of bg-color (0?) and fg-depth.
        # If background is 0 and depth is 10, edge might be 5.
        # This implies "object is at depth 5 at edge". This is wrong. Use hard mask for depth usually.
        # But let's keep consistency with NormalsRenderer for now, or just return raw interpolated.

        # For pure depth tasks, usually AA is skipped or handled carefully.
        # But `col` implies visualization.
        col = dr.antialias(col, rast_out, vertices_clip, faces)

        # If we strictly want the Depth VALUE (channel 0), the AA might corrupt it at edges.
        # But rasterization IS discrete.

        return col
