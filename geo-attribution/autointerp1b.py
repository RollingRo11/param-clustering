"""Auto-interp on the Llama-3.2-1B decomposition. Importing geo1b first
patches geo67 (OUT_ROOT, target, loader), so autointerp67's evidence/label
stages run verbatim against the 1B banks:
  python3.12 autointerp1b.py evidence --tag run1 --banks_tag propC2048 --batches 36
  python3.12 autointerp1b.py label    --tag run1 --banks_tag propC2048
"""

import sys

sys.path.insert(0, "/workspace/circuit-decomp/geo-attribution")

import geo1b  # noqa: F401 — applies the 1B patches before autointerp67 binds
import autointerp67

if __name__ == "__main__":
    autointerp67.main()
