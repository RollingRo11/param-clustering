"""test_matryoshka67.py's checks on KISSKI, where the paired train split is not copied: the checks that read training
rows (speed-ups against the reference implementation, the persistent PGD loss, gradient flow) read rows of VPD's
stream (data/vpd_stream/rank0.uint16) instead. Everything else is the same test file."""
import numpy as np

import test_matryoshka67 as tests
from common import ROOT

stream = np.memmap(ROOT/'data/vpd_stream/rank0.uint16', mode='r', dtype=np.uint16, shape=(3_200_000, 513))
tests.tokens = lambda split='train': (stream, None)

if __name__ == '__main__': tests.main()
