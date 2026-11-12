"""Build one depth-guide PNG for the head-proxy mesh at a given angle index.
v2: depth is Gaussian-smoothed (radius 2) to remove flat-face steps.
Replicates the Colab v2/v3 notebook guide cell exactly (same mesh, camera,
projection) so guides are reproducible from source on any machine.
Angle index 0..7 -> 0,45,...,315 degrees.
"""
import argparse, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perspective_fixed as fx

def sphere(r, nt=32, np_=24):
    v = []
    for p in np.linspace(0, np.pi, np_):
        for t in np.linspace(0, 2 * np.pi, nt, endpoint=False):
            v.append([r * math.sin(p) * math.cos(t), r * math.sin(p) * math.sin(t), r * math.cos(p)])
    return np.array(v)[:, [0, 2, 1]]

def faces(nt=32, np_=24):
    def idx(p, t): return p * nt + (t % nt)
    f = []
    for p in range(np_ - 1):
        for t in range(nt):
            a, b, c, d = idx(p, t), idx(p, t + 1), idx(p + 1, t + 1), idx(p + 1, t)
            f += [(a, b, d), (b, c, d)]
    return f

def build_mesh():
    parts = [(1.3, (0, 0, 0)), (0.45, (0.88, 1.12, 0)), (0.45, (-0.88, 1.12, 0)), (0.20, (0, -0.05, 1.28))]
    allv, allf, off = [], [], 0
    base = faces()
    for r, o in parts:
        v = sphere(r) + np.array(o)
        allv.append(v)
        allf += [(a + off, b + off, c + off) for (a, b, c) in base]
        off += len(v)
    return np.vstack(allv), np.array(allf)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from PIL import Image, ImageDraw
    verts, faces_arr = build_mesh()
    cam = fx.CameraSystem(position=[0, 0.3, -5.2], target=[0, 0.2, 0], up=[0, 1, 0])
    proj = fx.EfficientPerspective(fov=40, aspect=1.0, near=0.1, far=100.0)
    W = H = 512
    ang = args.index * 45
    th = math.radians(ang)
    rot = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
    vc = cam.transform_points(verts @ rot.T)
    xy = proj.project_points_vectorized(vc)
    px = (xy[:, 0] * 0.5 + 0.5) * (W - 1)
    py = (0.5 - xy[:, 1] * 0.5) * (H - 1)
    fc = vc[faces_arr]
    cents = fc.mean(axis=1)
    order = np.argsort(-cents[:, 2])
    im = Image.new("L", (W, H), 0)  # v2: depth is smoothed before saving (see below)
    dr = ImageDraw.Draw(im)
    dmin, dmax = vc[:, 2].min(), vc[:, 2].max()
    for f_i in order:
        n = np.cross(fc[f_i][1] - fc[f_i][0], fc[f_i][2] - fc[f_i][0])
        if n[2] >= 0:
            continue
        g = int(255 * (1 - (cents[f_i][2] - dmin) / (dmax - dmin)))
        dr.polygon([(px[j], py[j]) for j in faces_arr[f_i]], fill=g)
    # v2 fix: smooth the flat-face depth steps so the generator does not
    # read the faceting as a tile pattern (v1 result: disco-ball surface).
    from PIL import ImageFilter
    im = im.filter(ImageFilter.GaussianBlur(radius=2))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    im.save(args.out)
    print(f"guide index {args.index} (angle {ang}) -> {args.out}")

if __name__ == "__main__":
    main()
