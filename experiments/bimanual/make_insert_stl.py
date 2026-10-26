"""Write peg.stl and socket.stl (millimetres) for task 2 from `envs/insertion_parts.py`, the constants the simulator
builds the parts from. Both meshes are built face by face so every edge is shared by exactly two triangles (watertight);
the script checks that and the volume before writing. numpy only; the simulator is not imported.

    python3 experiments/bimanual/make_insert_stl.py [out_dir]      (default: experiments/bimanual/out/insert_stl)
"""
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("insertion_parts", Path(__file__).resolve().parents[2] / "envs" / "insertion_parts.py")
parts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parts)
mm = lambda half: round(2000 * half, 6)  # noqa: E731  half size (m) -> full size (mm)
PEG = (mm(parts.PEG_HALF_LENGTH), mm(parts.PEG_HALF_WIDTH), mm(parts.PEG_HALF_WIDTH))  # length, width, height
SOCKET_LEN, SOCKET_OUT, SOCKET_HOLE = mm(parts.SOCKET_HALF_LENGTH), mm(parts.SOCKET_HALF_OUTER), mm(parts.SOCKET_HALF_HOLE)
HOLE_DEPTH = round(1000 * parts.SOCKET_HOLE_DEPTH, 6)


def quad(a, b, c, d):
    """Two triangles a-b-c, a-c-d (counter-clockwise seen from outside = outward normal)."""
    return [(a, b, c), (a, c, d)]


def box(lx, ly, lz):
    x, y, z = lx / 2, ly / 2, lz / 2
    v = {k: np.array(p, float) for k, p in {
        "000": (-x, -y, -z), "100": (x, -y, -z), "110": (x, y, -z), "010": (-x, y, -z),
        "001": (-x, -y, z), "101": (x, -y, z), "111": (x, y, z), "011": (-x, y, z)}.items()}
    t = []
    t += quad(v["000"], v["010"], v["110"], v["100"])  # bottom (-z)
    t += quad(v["001"], v["101"], v["111"], v["011"])  # top (+z)
    t += quad(v["000"], v["100"], v["101"], v["001"])  # -y
    t += quad(v["010"], v["011"], v["111"], v["110"])  # +y
    t += quad(v["000"], v["001"], v["011"], v["010"])  # -x
    t += quad(v["100"], v["110"], v["111"], v["101"])  # +x
    return t


def socket():
    """Long axis x from 0 (closed end) to SOCKET_LEN (mouth). Outer half o, hole half h, hole from x = L - D to L."""
    L, o, h, x0 = SOCKET_LEN, SOCKET_OUT / 2, SOCKET_HOLE / 2, SOCKET_LEN - HOLE_DEPTH
    P = lambda x, y, z: np.array((x, y, z), float)  # noqa: E731
    sq = [(-1, -1), (1, -1), (1, 1), (-1, 1)]  # counter-clockwise in (y, z) seen from +x
    t = []
    # back face x = 0, normal -x: clockwise seen from +x
    b = [P(0, o * a, o * c) for a, c in sq]
    t += quad(b[0], b[3], b[2], b[1])
    # outer sides, x 0..L
    for i in range(4):
        a, c = sq[i], sq[(i + 1) % 4]
        t += quad(P(0, o * a[0], o * a[1]), P(0, o * c[0], o * c[1]), P(L, o * c[0], o * c[1]), P(L, o * a[0], o * a[1]))
    # mouth ring x = L, normal +x: outer corners O, inner corners I, counter-clockwise from +x
    O = [P(L, o * a, o * c) for a, c in sq]
    I = [P(L, h * a, h * c) for a, c in sq]
    for i in range(4):
        j = (i + 1) % 4
        t += quad(O[i], O[j], I[j], I[i])
    # hole walls x0..L, normals pointing into the hole (toward the axis)
    for i in range(4):
        a, c = sq[i], sq[(i + 1) % 4]
        t += quad(P(x0, h * a[0], h * a[1]), P(L, h * a[0], h * a[1]), P(L, h * c[0], h * c[1]), P(x0, h * c[0], h * c[1]))
    # hole bottom x = x0, normal +x (facing the mouth)
    f = [P(x0, h * a, h * c) for a, c in sq]
    t += quad(f[0], f[1], f[2], f[3])
    return t


def check(tris, name):
    edges = {}
    vol = 0.0
    for a, b, c in tris:
        vol += np.dot(a, np.cross(b, c)) / 6
        for p, q in ((a, b), (b, c), (c, a)):
            k = (tuple(np.round(p, 6)), tuple(np.round(q, 6)))
            edges[k] = edges.get(k, 0) + 1
    bad = [k for k, n in edges.items() if n != 1 or edges.get((k[1], k[0]), 0) != 1]
    print(f"{name}: {len(tris)} triangles, signed volume {vol / 1000:.2f} cm^3, unpaired edges {len(bad)}")
    assert not bad and vol > 0, f"{name} is not a closed outward-facing mesh"
    return vol


def write_stl(tris, path, name):
    with open(path, "wb") as fh:
        fh.write(name.encode().ljust(80, b" "))
        fh.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            n = np.cross(b - a, c - a)
            n = n / (np.linalg.norm(n) or 1)
            fh.write(struct.pack("<12fH", *n, *a, *b, *c, 0))


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "out" / "insert_stl")
    out.mkdir(parents=True, exist_ok=True)
    peg = box(*PEG)
    sock = socket()
    vp = check(peg, "peg")
    vs = check(sock, "socket")
    exp_s = SOCKET_LEN * SOCKET_OUT ** 2 - HOLE_DEPTH * SOCKET_HOLE ** 2
    assert abs(vp - PEG[0] * PEG[1] * PEG[2]) < 1e-3 and abs(vs - exp_s) < 1e-3, "volume does not match the design"
    write_stl(peg, out / "peg.stl", f"phi task2 peg {PEG[1]:g}x{PEG[2]:g}x{PEG[0]:g} mm")
    write_stl(sock, out / "socket.stl", f"phi task2 socket {SOCKET_OUT:g}x{SOCKET_OUT:g}x{SOCKET_LEN:g} mm, {SOCKET_HOLE:g} mm hole {HOLE_DEPTH:g} mm deep")
    print("wrote", out / "peg.stl", out / "socket.stl")
