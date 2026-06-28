"""pytest configuration — path setup and global filter rules."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# rasterio uses a NumPy 2.5-deprecated array-shape pattern internally; suppress
# the noise until rasterio ships a fix upstream.
warnings.filterwarnings(
    "ignore",
    message="Setting the shape on a NumPy array has been deprecated",
    category=DeprecationWarning,
)
