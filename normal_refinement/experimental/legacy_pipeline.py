"""Historical API-based autoregressive experiment; use pipeline.py for the supported path."""

import sys
import subprocess
from pathlib import Path

try:
    from normal_refinement.providers.background import process_single_image as rb_process
    from normal_refinement.postprocess import smooth_taubin
    from normal_refinement.mesh import preprocess_mesh
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

import argparse

def main():
    parser = argparse.ArgumentParser(description="3D Mesh Refinement Pipeline")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to the input mesh file (.obj or .glb)")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save the outputs. Defaults to the directory of the input mesh.")
    parser.add_argument("--cat", type=str, default="object", help="Category of the object (e.g., chair, car) to guide the detail generation.")
    parser.add_argument("--re_gen", action="store_true", help="Force re-generation of normals and segmentation")
    parser.add_argument("--re_remesh", action="store_true", help="Force re-execution of the mesh refinement step")
    parser.add_argument("--use_frequency_separation", action=argparse.BooleanOptionalAction, help="Enable frequency separation to preserve global shape", default=True)
    parser.add_argument("--smooth", action="store_true", help="Apply Taubin smoothing after mesh refinement")
    parser.add_argument("--smooth_iter", type=int, default=10, help="Number of Taubin smoothing iterations")
    parser.add_argument("--autoregressive", action="store_true", help="Enable Autoregressive processing mode instead of parallel Multi-view mode")
    parser.add_argument("--n_azimuth", type=int, default=12, help="Number of azimuth views for Autoregressive refinement")
    parser.add_argument("--style_ref", type=str, default=None, help="Optional path to a style reference image for both Autoregressive and Parallel modes.")
    args = parser.parse_args()
    if not args.autoregressive:
        parser.error("Use normal_refinement.pipeline for parallel 4/6-view refinement; this entry point preserves the historical AR experiment only")

    obj_path = Path(args.mesh_path)
    if not obj_path.exists():
        raise FileNotFoundError(args.mesh_path)

    if args.out_dir is None:
        case_dir = obj_path.parent
    else:
        case_dir = Path(args.out_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

    mesh_name = obj_path.stem
    re_gen = args.re_gen
    re_remesh = args.re_remesh
    apply_smooth = args.smooth
    smooth_iter = args.smooth_iter
    cat_name = args.cat

    print("==========================================")
    print(f"Processing mesh: {obj_path.name} in {case_dir}")
    print("==========================================")

    preprocessed_obj_path = case_dir / f"{mesh_name}_preprocessed.obj"
    if not preprocessed_obj_path.exists() or re_gen or re_remesh:
        print("[Step 0] Preprocessing mesh topology and sampling before rendering/refinement...")
        try:
            _, preprocess_stats = preprocess_mesh(obj_path, preprocessed_obj_path)
        except Exception as e:
            raise RuntimeError(f"Mesh preprocessing failed for {obj_path}") from e
        print(
            "[Step 0] Preprocessed mesh saved to "
            f"{preprocessed_obj_path.name}: "
            f"V {preprocess_stats['input_vertices']} -> {preprocess_stats['output_vertices']}, "
            f"F {preprocess_stats['input_faces']} -> {preprocess_stats['output_faces']}, "
            f"welded {preprocess_stats['welded_vertices']} vertices, "
            f"removed {preprocess_stats['removed_faces']} faces, "
            f"subdivision_added V/F "
            f"{preprocess_stats['added_vertices_by_subdivision']}/"
            f"{preprocess_stats['added_faces_by_subdivision']}, "
            f"target_edge={preprocess_stats['target_edge_len']:.3e}"
        )
    else:
        print(f"[Step 0] Skipped. Preprocessed mesh {preprocessed_obj_path.name} already exists.")
    obj_path = preprocessed_obj_path

    if args.autoregressive:
        # Autoregressive block
        n_views = args.n_azimuth
        print(f"==== Autoregressive optimization for {case_dir.name} ====")

        current_mesh_path = obj_path
        historical_img_paths = []
        historical_views = []

        for v_idx in range(n_views):
            print(f"\n--- [AR Step {v_idx+1}/{n_views}] View angle index: {v_idx} ---")
            view_dir = case_dir / f"view_step_{v_idx}"
            view_dir.mkdir(exist_ok=True)

            rendered_normal_path = view_dir / f"rendered_normal_{v_idx}.png"
            mask_path = view_dir / "visible_mask.png"
            inpainted_normal_path = view_dir / "inpainted_normal.png"
            segmented_inpainted_path = view_dir / "inpainted_normal_segmented.png"
            next_mesh_path = view_dir / f"refined_step_{v_idx}.obj"

            historical_img_paths.append(str(segmented_inpainted_path))
            historical_views.append(str(v_idx))

            if next_mesh_path.exists() and not re_gen and not re_remesh:
                print(f"[Skip] {next_mesh_path.name} already exists.")
                current_mesh_path = next_mesh_path
                continue

            coarse_normal_path = view_dir / f"coarse_normal_{v_idx}.png"
            if not coarse_normal_path.exists() or re_gen:
                print("AR-0. Rendering coarse mesh for view as geometric reference...")
                cmd = [
                    sys.executable, "-m", "normal_refinement.experimental.autoregressive",
                    "--name", case_dir.name,
                    "--mesh_path", str(obj_path),
                    "--autoreg_mode", "render",
                    "--view_idx", str(v_idx),
                    "--n_azimuth", str(n_views),
                    "--out_dir", str(view_dir),
                    "--force_normalize"
                ]
                subprocess.run(cmd, check=True)
                if rendered_normal_path.exists():
                    rendered_normal_path.rename(coarse_normal_path)

            print("AR-1. Rendering current mesh and extracting new-part mask...")
            cmd_curr = [
                sys.executable, "-m", "normal_refinement.experimental.autoregressive",
                "--name", case_dir.name,
                "--mesh_path", str(current_mesh_path),
                "--autoreg_mode", "render",
                "--view_idx", str(v_idx),
                "--n_azimuth", str(n_views),
                "--out_dir", str(view_dir),
                "--out_mask_path", str(mask_path)
            ]
            if current_mesh_path == obj_path:
                cmd_curr.append("--force_normalize")

            subprocess.run(cmd_curr, check=True)

            print("AR-2. Inpainting new regions using Gemini...")
            from normal_refinement.providers.gemini import process_inpainting_image

            success = False
            max_retries = 3
            for attempt in range(max_retries):
                success = process_inpainting_image(
                    coarse_file=str(coarse_normal_path),
                    mask_file=str(mask_path),
                    output_file=str(inpainted_normal_path),
                    cat=cat_name,
                    style_ref=args.style_ref
                )
                if success:
                    break
                print(f"Gemini inpainting failed. Retrying ({attempt + 1}/{max_retries})...")
                import time
                time.sleep(2)

            # Fallback handling in case of API failure
            if not success:
                print(f"Gemini inpainting failed after {max_retries} attempts. Falling back to the original render...")
                import shutil
                if rendered_normal_path.exists():
                    shutil.copy(str(rendered_normal_path), str(inpainted_normal_path))
                else:
                    shutil.copy(str(coarse_normal_path), str(inpainted_normal_path))

            print("AR-3. Removing background from inpainted normal...")
            rb_process(str(inpainted_normal_path))

            print(f"AR-4. Optimizing mesh (joint optimization with {len(historical_views)} views)...")
            cmd_opt = [
                sys.executable, "-m", "normal_refinement.experimental.autoregressive",
                "--name", case_dir.name,
                "--mesh_path", str(current_mesh_path),
                "--autoreg_mode", "optimize",
                "--view_idx", str(v_idx),
                "--n_azimuth", str(n_views),
                "--out_dir", str(view_dir),
                "--output_mesh_path", str(next_mesh_path),
                "--views", *historical_views,
                "--img_paths", *historical_img_paths,
            ]
            if args.use_frequency_separation:
                cmd_opt.append("--use_frequency_separation")
            if current_mesh_path == obj_path:
                cmd_opt.append("--force_normalize")

            subprocess.run(cmd_opt, check=True)
            current_mesh_path = next_mesh_path

        print(f"[Done] Final AR mesh saved at {current_mesh_path}")

        if apply_smooth:
            smoothed_obj_path = case_dir / f"{mesh_name}_autoreg_smoothed.obj"
            if not smoothed_obj_path.exists() or re_remesh:
                print(f"[AR Step 6] Applying Taubin smoothing for {smooth_iter} iterations...")
                smooth_taubin(str(current_mesh_path), str(smoothed_obj_path), smooth_iter)


if __name__ == "__main__":
    main()
