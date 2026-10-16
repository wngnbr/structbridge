import pathlib
import subprocess
import sys

def process_single_image(img_path_str):
    """
    对单张图片去除背景，输出 xxx_segmented.png
    """
    img_path = pathlib.Path(img_path_str)
    if not img_path.exists():
        print(f"Skipping: {img_path} (Not found)")
        return None

    output_name = img_path.stem + '_segmented.png'
    output_path = img_path.parent / output_name


    print(f"[Start] Removing Background: {img_path}")
    try:
        subprocess.run([
            sys.executable, "-m", "normal_refinement.providers.foreground",
            "--input", str(img_path),
            "--output", str(output_path)
        ], check=True)
        print(f"[Done] Background removed: {output_path.name}")
        return str(output_path)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Failed processing {img_path}: {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process_single_image(sys.argv[1])
    else:
        print("Usage: python rb_img.py <path_to_image>")
