"""Prints matplotlib's inferno colormap as the bytes `iris.downstream.depth.colorize` writes:
`round(lut[i][:3] * 255)` (NumPy's half-to-even), one row per index, as a Rust table."""
import numpy as np
from matplotlib import colormaps

lut = colormaps["inferno"](np.arange(256))[:, :3]
rows = (lut * 255).round().astype(np.uint8)
print("pub(crate) const INFERNO: [[u8; 3]; 256] = [")
for row in rows:
    print(f"    [{row[0]}, {row[1]}, {row[2]}],")
print("];")
