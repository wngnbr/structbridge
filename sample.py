"""Generate meshes for a directory of preprocessed shape NPZ files."""

from pathlib import Path
from diffusion_bridge.inference import create_argparser, run_sampling


def main():
    args = create_argparser(batch=True).parse_args()
    paths = sorted(Path(args.data_dir).glob("*.npz"))
    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if args.limit:
        paths = paths[:args.limit]
    run_sampling(paths, args, batch=True)


if __name__ == "__main__":
    main()
