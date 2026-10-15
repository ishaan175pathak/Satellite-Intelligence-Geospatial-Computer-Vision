import os
import sys

# Allow `pytest` (not only `python -m pytest`) to import the `src` package.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
