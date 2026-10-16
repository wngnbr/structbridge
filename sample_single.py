"""Generate one mesh from a preprocessed shape NPZ."""

from diffusion_bridge.inference import create_argparser, run_sampling


def main():
    args = create_argparser().parse_args()
    run_sampling([args.npz_path], args)


if __name__ == "__main__":
    main()
