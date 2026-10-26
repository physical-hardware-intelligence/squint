"""Task 2 part dimensions (m), nominal. The one source for the simulator (`envs/insertion.py`) and the printed parts
(`experiments/bimanual/make_insert_stl.py`). No imports, so the STL script can load it without the simulator.

- peg: 22 x 22 x 90 mm box. 22 mm is inside the jaw range that grasps task 1's 25 mm bar.
- socket: 40 x 40 x 80 mm box with a 32 x 32 mm square hole, 50 mm deep, open at one end; the other 30 mm is solid,
  where the left arm holds it. 32 - 22 = 10 mm total clearance (5 mm a side): the policy's 16 x 16 images resolve
  0.9-1.8 cm per pixel (GUIDE section 3), so a tighter fit would be below one pixel.
"""

PEG_HALF_WIDTH = 0.011
PEG_HALF_LENGTH = 0.045
SOCKET_HALF_OUTER = 0.020
SOCKET_HALF_HOLE = 0.016
SOCKET_HALF_LENGTH = 0.040
SOCKET_HOLE_DEPTH = 0.050
