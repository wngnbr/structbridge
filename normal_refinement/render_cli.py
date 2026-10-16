"""Render normal targets in a short-lived GPU process."""

import argparse
from .rendering import render_normals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh_path', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--num_views', type=int, choices=[4, 6], default=4)
    args = parser.parse_args()
    render_normals(args.mesh_path, args.out_dir, args.resolution, args.num_views)


if __name__ == '__main__':
    main()
