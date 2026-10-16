# StructBridge

Official code for **StructBridge: Structure-Grounded 3D Indoor Object Generation via 3D Latent Diffusion Bridge and Normal Refinement**.

## Overview

StructBridge generates detailed 3D indoor objects while preserving a specified structural configuration. The paper introduces **StructMap**, which represents object structure through composable geometric primitives, giving control over component layout, geometric forms, and relative sizes.

Generation proceeds in two stages:

1. **3D latent diffusion bridge.** A diffusion bridge in a frozen shape VAE's latent space transforms the structural condition into a coarse object mesh.
2. **Structure-preserving normal refinement.** An image generation model enriches rendered multi-view normal maps. Frequency-aware normal extraction transfers fine surface details while retaining the coarse mesh's low-frequency structure, and differentiable mesh optimization incorporates these details into the geometry.

```text
Structural mesh → Surface sampling → 3D latent diffusion bridge → Coarse mesh
                                                                    ↓
Final OBJ ← Normal-guided mesh refinement ← Enhanced multi-view normal maps
```


## Installation

The entire pipeline uses **one Python 3.10 environment**, with PyTorch 2.6.0 and CUDA 12.4. Requirements: Linux x86_64 (glibc ≥ 2.28), an NVIDIA GPU, Conda, Git, a C++ compiler, and the CUDA 12.4 toolkit (`nvcc`).

Run from the repository root:

```bash
# Adjust this path if CUDA is installed elsewhere.
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"

# Create the environment, then build/install extensions against its PyTorch.
conda env create -f environments/structbridge.yaml
conda activate structbridge
python -m pip install --no-build-isolation -r environments/extensions.txt

# Check the main dependencies and GPU access.
python -c "import torch, torch_cluster, torch_scatter, cubvh, diso, bpy, fpsample, nvdiffrast.torch; assert torch.cuda.is_available(); print('Environment ready')"
```

Use this same environment for preprocessing, diffusion bridge inference, image API calls, and normal refinement. For optional smoothing (`--smooth_iter > 0`), also run `python -m pip install open3d==0.19.0`.

## Model weights

Download the VAE and diffusion bridge checkpoints from [Google Drive](https://drive.google.com/drive/folders/1fATyK2V32KrubtWgVvKkB8QDJnjXw-qP) and place them under `ckpts/` at the repository root:

```text
StructBridge/
├── ckpts/
│   ├── dora_vae_1_1.ckpt
│   └── diffusion_bridge_300k.pt
├── diffusion_bridge/
├── normal_refinement/
└── scripts/sample.sh
```

## Image generation API

Stage two requires an **enhanced normal atlas for the current mesh**. You can generate it with the included Gemini-compatible adapter or provide an image generated elsewhere.

For the adapter, set these variables in the shell before running the generation command:

```bash
export GEMINI_API_URL="https://YOUR_PROVIDER/v1beta/models/YOUR_IMAGE_MODEL:generateContent"
export GEMINI_API_KEY="YOUR_API_KEY"
```

Use an image-capable Gemini-compatible `generateContent` endpoint that accepts `Authorization: Bearer` authentication. The script reads these environment variables directly and calls the API automatically when a normal atlas is missing.

## Inference

After installing the environment, downloading the weights, and configuring the API, run the entire pipeline from the repository root:

```bash
conda activate structbridge
CUDA_VISIBLE_DEVICES=0 bash scripts/sample.sh /path/to/structural_mesh.glb \
  --cat "<object_category>"
```

Replace `<object_category>` with the object category, such as `chair` or `mug`. The script automatically preprocesses the input, runs the diffusion bridge, renders multi-view normals, calls the image API, and refines the geometry.

Results are saved to **`outputs/<input-stem>/`** by default. For `structural_mesh.glb`, the final mesh is **`outputs/structural_mesh/final.obj`**. Use the optional `--output_dir /path/to/results` to choose a different directory, for example when comparing multiple runs of the same input.

you can run the bundled chair model as example:

```bash
conda activate structbridge
CUDA_VISIBLE_DEVICES=0 bash scripts/sample.sh assets/chair.glb --cat chair
```

The final mesh is saved to **`outputs/chair/final.obj`**. The coarse mesh and normal maps are retained in `outputs/chair/stage1/` and `outputs/chair/stage2/`.
