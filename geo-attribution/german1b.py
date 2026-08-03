"""German-erasure edit on the Llama-3.2-1B decomposition.

Importing geo1b first patches geo67 (OUT_ROOT -> /dev/shm/geo1b, Llama target),
so german67 runs verbatim against the 1B banks:
  python3.12 german1b.py --tag run1 --banks_tag prop1b --space weight ...
"""

import sys

sys.path.insert(0, "/workspace/circuit-decomp/geo-attribution")

import geo1b  # noqa: F401 — applies the 1B patches to geo67 before german67 binds
import german67

if __name__ == "__main__":
    german67.main()
