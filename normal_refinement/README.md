# Normal-guided mesh refinement

Stage two of StructBridge fits explicit mesh geometry to generated multi-view
normal maps. It accepts the GLB from the diffusion bridge and produces `final.obj`
in the **same coordinate frame and scale as its input**.

## Layout

```text
normal_refinement/
  pipeline.py          # Resumable preparation, image handoff, fitting and export
  rendering.py         # Shared 4/6-view cameras and coordinate transforms
  images.py            # Generation requests, atlas validation, alignment and masks
  optimization.py      # Frequency separation, normal loss and fitting loop
  mesh.py              # Conservative welding/cleaning and subdivision
  core/                # Adaptive remeshing and vertex optimizer
  utils/               # Mesh I/O, differentiable rendering and optional viewers
  postprocess.py       # Optional smoothing
  simplify.py          # Optional explicit mesh decimation
  providers/           # Image generation and optional foreground-removal adapters
  experimental/        # Historical autoregressive experiment (not the default)
  tests/               # Pipeline regression tests
```

The default path generates missing normal images through the configured API.
It does not download segmentation weights.
It masks the generated black background and intersects it with the rendered
silhouette after alignment. The original core remeshing algorithm is retained.
Camera construction is shared by rendering and fitting; six-view indexing and
output coordinate restoration no longer depend on duplicated code.

## Environment

Use the shared `structbridge` Python 3.10 environment described in the
[root installation guide](../README.md#installation). The same environment runs
preprocessing, diffusion bridge inference, and normal refinement. Rasterization
uses CUDA and does not require a graphical display.

For optional smoothing or decimation, install `open3d==0.19.0` in that environment.
`requirements-optional.txt` lists additional dependencies for experimental tools.

## End-to-end use

Run from the StructBridge repository root:

```bash
conda activate structbridge
CUDA_VISIBLE_DEVICES=0 bash scripts/sample.sh --cat chair
```

This reads `mesh.glb` by default. You can also pass a mesh/NPZ path:

```bash
bash scripts/sample.sh /path/to/input.npz --cat chair --output_dir outputs/chair_case
bash scripts/sample.sh --stage1_mesh /path/to/stage1.glb --cat chair --output_dir outputs/refine_case
```

The final file is `<output_dir>/final.obj`. The default directory is
`outputs/<input-stem>/`. `--stage1_only` stops after the diffusion bridge.
Existing stage-one sampler flags such as `--steps`, `--eta`, `--seed`, `--config`
and `--set` remain available. `--refine_steps` controls stage two separately
(default 101); `--refine_resolution` defaults to 2048 per view. Stage two uses all
four views by default; `--num_views 6` adds top and bottom. Category names affect
the generation prompt, not hidden view-selection rules.

## Automatic image generation

Set `GEMINI_API_URL` and `GEMINI_API_KEY` as described in the
[root API configuration](../README.md#image-generation-api). Both the full wrapper
and the standalone refinement command automatically generate a missing atlas,
then continue to mesh fitting. Four-view generation uses a 1:1 atlas and six-view
generation uses 3:2. Valid images are saved atomically as `generated_normals.png`
and reused on subsequent runs. HTTP errors, timeouts, and invalid image responses
stop the pipeline with a nonzero exit status.

`--image_provider auto` is the default. `--image_provider gemini` requires API
configuration when a new image is needed; `--image_provider manual` disables API
calls. An explicit `--normal_image` always takes priority and is never replaced
with an API result.

## Image generation without an API

Codex's built-in image generator is available to the conversation, **not to a
standalone Bash/Python process**. Use `--image_provider manual` for this workflow:

1. The pipeline renders `stage2/rendered_normals.png` and writes
   `stage2/image_request.json` plus `image_prompt.txt`.
2. Use the reference atlas as an edit target in Codex's built-in image generator.
   Preserve its layout, silhouettes and camera-normal colors. Add coherent small
   surface details, not photographic shading. Save the PNG to the request's
   `output_image` path, normally `stage2/generated_normals.png`.
3. Rerun the same command. Valid stage-one outputs and rendered references are
   reused; alignment, geometry optimization and OBJ export continue.

Missing images cause exit status **3**, not a successful result or a silent copy
of the original normals. `--wait_for_image 1800` keeps the script waiting for up
to 30 minutes while an agent supplies the PNG; write to a temporary filename and
rename after the image is fully saved. `--normal_image /path/to/atlas.png` accepts
an explicitly supplied atlas. For a previously completed input, cached generated
images and the checked final OBJ allow the same command to finish unattended.
A new input still needs image generation; this repository does not pretend that
the built-in tool is a locally callable API.

The current demo images were generated with the built-in tool. Each verified case
includes `generation_provenance.json` with the exact prompt, reference and saved
image path. The separate autoregressive experiment can
be accessed as `python -m normal_refinement.experimental.legacy_pipeline
--autoregressive ...`; it was retained but not validated end to end in this run.

## Standalone second stage

```bash
conda activate structbridge
python -m normal_refinement.pipeline \
  --mesh_path /path/to/stage1.glb --out_dir outputs/refined_case --cat chair
```

`normal_refinement/run_pipeline.py` remains a compatibility entry point.
The standalone CLI also exposes `--no-frequency_separation`, `--detail_strength`,
`--edge_min`, `--edge_max`, `--laplacian_weight`, `--silhouette_weight`,
`--render_resolution`, `--smooth_iter` and `--re_remesh`. Smoothing is disabled
by default to retain the fitted details. `--re_remesh` reruns optimization using
the existing image; it does not make a new image-generation request.

## Outputs and repeatability

```text
outputs/<case>/
  pipeline.json                 # Stage-one identity and final output location
  preprocessing/                # Only needed for raw mesh input
  stage1/bridge.glb
  stage2/
    manifest.json               # Input mesh hash, view settings and final hash
    input_clean.obj
    rendered_normals.png
    rendered_views/             # RGBA camera normals
    image_request.json
    image_prompt.txt
    generated_normals.png
    target_views/               # Aligned and masked generated normals
    image_alignment.json        # Per-view silhouette overlap
    optimization/
      before_normals/ after_normals/ fitting_targets/
      loss_history.json metrics.json
    refined.obj final.obj
  final.obj
```

Mesh hashes bind the refinement cache to its input. Changes to mesh/view settings
require another output directory. Changed generated image content or optimization
settings rerun fitting. Stage-one cache checks include resolved settings and
checkpoint path/size/modification time; keep checkpoint files unchanged for
reproducibility. Original mesh preprocessing is stochastic on a fresh run, so the
pipeline retains its sampled NPZ/bridge output for resumes. Failed preprocessing
cannot consume a stale output. This is stage-level resume, not optimizer-step
checkpointing inside stage two.

Normal loss reduction measures agreement with generated supervision, not a
reference-based guarantee of reconstruction quality. Inspect the result before
using it in paper figures. Adaptive remeshing can produce millions of triangles;
`python -m normal_refinement.simplify input.obj output.obj --target_faces 500000`
provides optional decimation when file size matters.

## Checks

In the refinement environment:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m unittest discover -s normal_refinement/tests -v
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m unittest discover -s normal_refinement/core/tests -v
```

The pipeline tests cover camera-normal round trips, six-view rendering, coordinate
restoration, equivalent separable filtering, image requests, masking and stale
input rejection. The core suite exercises geometry/remeshing operations.
Execution logs for the local full-resolution runs are in
`run_logs/normal_refinement/` (ignored by Git).

The verified chair cases ran 101 updates at 2048 pixels per view using built-in
generated atlases. The earlier bridge output was refined to
`outputs/refinement_previous/final.obj`; the complete `scripts/sample.sh` run
produced `outputs/mesh/final.obj`. Their final normal-fit MSE reductions were
about 76% and 77%, respectively. Each case includes `comparison.png` showing
surface shading derived from its before/after rendered normals, with the same
camera and light. These are local smoke/quality checks, not a dataset benchmark.
Eight pipeline tests, fifteen remeshing tests and eleven stage-one regression
tests passed. The no-argument shell command was also rerun successfully using
the cached outputs.
