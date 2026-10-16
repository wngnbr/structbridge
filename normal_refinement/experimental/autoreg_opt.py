import torch
import torch_scatter
from normal_refinement.core.opt import lerp_unbiased
from normal_refinement.core.remesh import calc_edges, calc_edge_length, calc_face_normals, calc_vertex_normals, calc_face_collapses, collapse_edges, split_edges, prepend_dummies, remove_dummies, pack, flip_edges

@torch.no_grad()
def remesh_autoreg(vertices_etc, faces, min_edgelen, max_edgelen, flip=True, max_vertices=1e6):
    vertices_etc,faces = prepend_dummies(vertices_etc,faces)
    vertices = vertices_etc[:,:3] #V,3
    nan_tensor = torch.tensor([torch.nan],device=min_edgelen.device)
    min_edgelen = torch.concat((nan_tensor,min_edgelen))
    max_edgelen = torch.concat((nan_tensor,max_edgelen))

    edges,face_to_edge = calc_edges(faces) #E,2 F,3
    edge_length = calc_edge_length(vertices,edges) #E
    face_normals = calc_face_normals(vertices,faces,normalize=False) #F,3
    vertex_normals = calc_vertex_normals(vertices,faces,face_normals) #V,3
    face_collapse = calc_face_collapses(vertices,faces,edges,face_to_edge,edge_length,face_normals,vertex_normals,min_edgelen,area_ratio=0.5)
    shortness = (1 - edge_length / min_edgelen[edges].mean(dim=-1)).clamp_min_(0)
    priority = face_collapse.float() + shortness

    # [FIX] Prevent collapsing edges connected to frozen vertices
    if vertices_etc.shape[1] >= 10:
        is_frozen = vertices_etc[:, 9] > 0.5
        edge_has_frozen = is_frozen[edges].any(dim=1)
        priority[edge_has_frozen] = 0.0

    vertices_etc,faces = collapse_edges(vertices_etc,faces,edges,priority)

    if vertices_etc.shape[0] < max_vertices + sum(vertices_etc.shape)*0: # just V < max
        edges,face_to_edge = calc_edges(faces) #E,2 F,3
        vertices = vertices_etc[:,:3] #V,3
        edge_length = calc_edge_length(vertices,edges) #E
        splits = edge_length > max_edgelen[edges].mean(dim=-1)

        # [FIX] Prevent splitting edges where BOTH vertices are frozen
        if vertices_etc.shape[1] >= 10:
            is_frozen = vertices_etc[:, 9] > 0.5
            both_frozen = is_frozen[edges].all(dim=1)
            splits = splits & (~both_frozen)

        vertices_etc,faces = split_edges(vertices_etc,faces,edges,face_to_edge,splits,pack_faces=False)

    vertices_etc,faces = pack(vertices_etc,faces)
    vertices = vertices_etc[:,:3]

    if flip:
        edges,_,edge_to_face = calc_edges(faces,with_edge_to_face=True) #E,2 F,3
        flip_edges(vertices,faces,edges,edge_to_face,with_border=False)

    return remove_dummies(vertices_etc,faces)

class AutoregMeshOptimizer:
    def __init__(self, vertices, faces, frozen_mask=None, lr=0.3, betas=(0.8, 0.8, 0), gammas=(0,0,0), nu_ref=0.3,
                 edge_len_lims=(.01, .15), edge_len_tol=.5, gain=.2, laplacian_weight=.02, ramp=1, grad_lim=10.,
                 remesh_interval=1, local_edgelen=True):
        self._vertices = vertices
        self._faces = faces
        self._lr = lr
        self._betas = betas
        self._gammas = gammas
        self._nu_ref = nu_ref
        self._edge_len_lims = edge_len_lims
        self._edge_len_tol = edge_len_tol
        self._gain = gain
        self._laplacian_weight = laplacian_weight
        self._ramp = ramp
        self._grad_lim = grad_lim
        self._remesh_interval = remesh_interval
        self._local_edgelen = local_edgelen
        self._step = 0

        self._init_frozen_mask = frozen_mask
        V = self._vertices.shape[0]
        # Allocate 10 columns for vertices_etc to hold frozen_mask
        self._vertices_etc = torch.zeros([V, 10], device=vertices.device)
        self._split_vertices_etc()
        self.vertices.copy_(vertices)
        if self._init_frozen_mask is not None:
            self._frozen_mask.copy_(self._init_frozen_mask)
        self._vertices.requires_grad_()
        self._ref_len.fill_(edge_len_lims[1])

    @property
    def vertices(self): return self._vertices
    @property
    def faces(self): return self._faces

    def _split_vertices_etc(self):
        self._vertices = self._vertices_etc[:,:3]
        self._m2 = self._vertices_etc[:,3]
        self._nu = self._vertices_etc[:,4]
        self._m1 = self._vertices_etc[:,5:8]
        self._ref_len = self._vertices_etc[:,8]
        self._frozen_mask = self._vertices_etc[:,9:10] # Column 9

        with_gammas = any(g!=0 for g in self._gammas)
        self._smooth = self._vertices_etc[:,:8] if with_gammas else self._vertices_etc[:,:3]

    def zero_grad(self):
        self._vertices.grad = None

    @torch.no_grad()
    def step(self):
        eps = 1e-8
        self._step += 1
        edges,_ = calc_edges(self._faces)
        E = edges.shape[0]
        edge_smooth = self._smooth[edges]
        neighbor_smooth = torch.zeros_like(self._smooth)
        torch_scatter.scatter_mean(src=edge_smooth.flip(dims=[1]).reshape(E*2,-1), index=edges.reshape(E*2,1), dim=0, out=neighbor_smooth)

        if self._gammas[0]: self._m1.lerp_(neighbor_smooth[:,5:8], self._gammas[0])
        if self._gammas[1]: self._m2.lerp_(neighbor_smooth[:,3], self._gammas[1])
        if self._gammas[2]: self._nu.lerp_(neighbor_smooth[:,4], self._gammas[2])

        laplace = self._vertices - neighbor_smooth[:,:3]
        grad = torch.addcmul(self._vertices.grad, laplace, self._nu[:,None], value=self._laplacian_weight)

        # APPLY FROZEN MASK
        grad = grad * (1.0 - self._frozen_mask)

        if self._step>1:
            grad_lim = self._m1.abs().mul_(self._grad_lim)
            grad.clamp_(min=-grad_lim, max=grad_lim)

        lerp_unbiased(self._m1, grad, self._betas[0], self._step)
        lerp_unbiased(self._m2, (grad**2).sum(dim=-1), self._betas[1], self._step)

        velocity = self._m1 / self._m2[:,None].sqrt().add_(eps)

        # APPLY FROZEN MASK TO VELOCITY TOO
        velocity = velocity * (1.0 - self._frozen_mask)

        speed = velocity.norm(dim=-1)
        if self._betas[2]: lerp_unbiased(self._nu, speed, self._betas[2], self._step)
        else: self._nu.copy_(speed)

        ramped_lr = self._lr * min(1, self._step * (1-self._betas[0]) / self._ramp)
        self._vertices.add_(velocity * self._ref_len[:,None], alpha=-ramped_lr)

        if self._step % self._remesh_interval == 0:
            if self._local_edgelen:
                len_change = (1 + (self._nu - self._nu_ref) * self._gain)
            else:
                len_change = (1 + (self._nu.mean() - self._nu_ref) * self._gain)
            self._ref_len *= len_change
            self._ref_len.clamp_(*self._edge_len_lims)

    def remesh(self, flip=True, custom_edge_lens=None):
        if custom_edge_lens is not None:
            self._ref_len = custom_edge_lens.clamp(*self._edge_len_lims)
        min_edge_len = self._ref_len * (1 - self._edge_len_tol)
        max_edge_len = self._ref_len * (1 + self._edge_len_tol)

        self._vertices_etc, self._faces = remesh_autoreg(self._vertices_etc, self._faces, min_edge_len, max_edge_len, flip)

        self._split_vertices_etc()
        self._vertices.requires_grad_()
        return self._vertices, self._faces
