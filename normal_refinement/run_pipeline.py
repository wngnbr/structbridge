"""Compatibility entry point; prefer python -m normal_refinement.pipeline."""

if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from normal_refinement.pipeline import main

if __name__ == '__main__':
    main()
