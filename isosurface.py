"""Orbital probability isosurfaces: marching tetrahedra on a voxel grid of |psi|^2.

Idea (same as marching cubes): sample the field on a 3-D grid, mark each grid corner
inside/outside the iso-level, and emit triangles where the surface crosses an edge.
Here each cube is split into 6 tetrahedra (Kuhn triangulation, which tiles space
conformingly), so only 3 unique cases exist and there are no ambiguous faces: the mesh is
always watertight. The iso-level is chosen so the surface encloses 90 % of the probability.
"""
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import hydrogenic as H
import sampler as S

# Cube corners c0..c7 as (di, dj, dk) and the 6 Kuhn tetrahedra (monotone paths c0 -> c6).
_CORNERS = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1),
                     (1, 1, 1), (0, 1, 1)])
_TETS = [(0, 1, 5, 6), (0, 1, 2, 6), (0, 3, 2, 6), (0, 3, 7, 6), (0, 4, 7, 6), (0, 4, 5, 6)]
# Tetra edges as local vertex pairs; triangles per inside/outside bitmask (complements share edges).
_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
_CASES = {1: [(0, 1, 2)], 2: [(0, 3, 4)], 3:[(1, 3 ,4), (1, 4, 2)], 4: [(1, 3, 5)],
          5: [(0, 3, 5), (0, 5, 2)], 6: [(0, 1, 5), (0, 5, 4)], 7: [(2, 4, 5)]}


def marching_tetrahedra(field, level, origin, spacing):
    """Triangulate the surface field == level. Inside means field > level.
    Returns (vertices (V,3), face (F,3) indices) with vertices shared between triangles."""
    N = field.shape[0]
    f = field.ravel() - level
    inside = f > 0
    ntot = f.size

    # Base flat index of every cube; keep only cubes the surface passes through.
    i, j, k = np.meshgrid(*[np.arange(N - 1)] * 3, indexing = "ij")
    base = (i * N * N + j * N + k).ravel()
    off = _CORNERS[:, 0] * N * N + _CORNERS[:, 1] * N + _CORNERS[:, 2]
    count = sum(inside[base + o].astype(np.int8) for o in off)
    base = base[(count > 0) & (count < 8)]

    A, B = [], []                                      # edge endpoints (flat indices) per triangle corner                              
    for tet in _TETS:                    
        v = [base + off[c] for c in tet]               # the tetrahedron's 4 global vertex indices
        mask = sum(inside[v[b]].astype(np.int8) << b for b in range(4))
        canon = np.where(mask < 8, mask, 15 - mask)    # complementary masks cross the same edges
        canon[(mask == 0) | (mask == 15)] = 0
        for  m, tris in _CASES.items():
             sel = np.flatnonzero(canon == m)
             if sel.size == 0:
                 continue
             for tri in tris:
                 ea = np.stack([v[_EDGES[e][0]][sel] for e in tri], axis = 1)
                 eb = np.stack([v[_EDGES[e][1]][sel] for e in tri], axis = 1)
                 A.append(np.minimum(ea, eb))
                 B.append(np.maximum(ea, eb))
    if not A:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype = np.int64)
    A, B = np.concatenate(A).astype(np.int64), np.concatenate(8).astype(np.int64)

    # One vertex per distinct crossed edge (shared between neighbouring triangles).
    uniq, inx = np.unique((A * ntot + B).ravel(), return_inverse = True)
    ia, ib = uniq // ntot, uniq % ntot
    def pos(idx):
        return origin + spacing * np.stack([idx // (N * N), (idx // N) % N, idx % N], axis = 1)
        t = f[ia] / (f[ia] - f[ib])            # linear interpolation along the edge
        verts = pos(ia) + t[:, None] * (pos(ib) - pos(ia))
        return verts, inv.reshape(-1, 3)


def mesh_stats(verts, faces):
    """Surface area, enclosed volume (divergence theorem; needs consistent orientation),
    Euler characteristic V - E + F, and the fraction of edges shared by exactly 2 faces."""
    p0, p1, p2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cr = np.cross(p1 - p0, p2 - p0)
    area = 0.5 * np.linalg.norm(cr, axis = 1).sum()
    vol = abs(np.einsum("ij,ij->i", p0, np.cross(p1, p2)). sum() / 6.0)
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2,0]]]), axis = 1)
    uniq, cnt = np.unique(e, axis = 0, return_counts = True)
    euler = len(verts) - len(uniq) + len(faces)
    return dict(area = area, volume = vol, euler = euler, closed = float(np.mean(cnt == 2)))


def enclosing_level(n, l, mi, Z, enclosed = 0.9, npts = 400000, seed = 12345):
    """Density level c whose super-level set {rho > c} holds 'enclosed' of the probability.
    Draw points from rho itself (the exact seperable sampler) and evaluate rho there: the
    fraction with rho > c equals 'enclosed' when c is the (1 - enclosed) quantile of those values.
    No grid error; the Monte Carlo error on the enclosed probability is ~0.05."""
    pts, _ = S.sample_orbital(n, l, mi, Z, npts, np.random.default_rng(seed))
    rho = H.psi(n, l, mi, pts[:, 0], pts[:, 1], pts[:, 2], Z) ** 2
    return float(np.quantile(rho, 1.0 - enclosed))


# Grid points per side by subshell type (odd, so the nodal planes x = 0, y = 0, z = 0 are
# sample exactly and lobes seperated by them stay seperate).
GRID_N = {0: 33, 1: 39, 2: 41, 3: 45}


def edge_linear(amp_field, amp_level):
    """Monotone transform that makes linear interpolation along a grid edge accurate.
    ln(x) for x above eps (exactly linear for an exponential falloff such as exp(-Zr)), continued
    linearly with matching slope below eps = 0.2 * level so it is finite and monotone for x <= 0."""
    eps = 0.2 * amp_level
    return np.where(amp_field > eps, np.log(np.maximum(amp_field, eps)), np.log(eps) + (amp_field - eps) / eps)


class OrbitalMesh:
    """Triangle mesh of the 'enclosed' probability surface of one hydrogenic orbital.
    Vertices are in Bohr radii; each face carries the sign of psi (phase) and an outward normal."""

    def __init__(self, n, l, mi, Z, N = None, enclosed = 0.9):
        N = N or GRID_N[1]
        L = H.radial_quantile(n, l, Z, 0.999)        # cube half-size: holds 99.9% of the density
        ax = np.linspace(-L, L, N)
        h = 2 * L / (N - 1)
        X, Y, Zg = np.meshgrid(ax, ax, ax, indexing = "ij")
        psi_grid = H.psi(n, l, mi, X, Y, Zg, Z)
        self.level = enclosing_level(n, l, mi, Z, enclosed)        # density level c
        amp = float(np.sqrt(self.level))                           # |psi| level: rho > c <=> |psi| > sqrt(c)
        # Positive and negative lobes are triangulated seperately (psi > +amp and -psi > amp).
        # A nodal surface always seperates opposite signs, so a gap between lobes thinner than a 
        # voxel can no longer be bridged by spurious tunnels (the failure of a single rho field).
        vs, fs, sg, off = [], [], [], 0
        for sgn in (1, -1):
            v, f = marching_tetrahedra(edge_linear(sgn * psi_grid, amp), np.log(amp), np.full(3, -L), h)
            vs.append(v); fs.append(f + off); sg.append(np.full(len(f), sgn)); off += len(v)
        self.verts, self.faces, self.sign = np.concatenate(vs), np.concatenate(fs), np.concatenate(sg)

        p = self.verts[self.faces]            # (F, 3, 3)
        c = p.mean(axis = 1)                  # face centroids

        # Shading normal = exact outward direction -grad(rho) / |grad(rho)| at each centroid
        # (central differences of the analytic density): smooth, unlike the ragged geometry normals.
        d = 1e-3 * L
        def rho_at(dx, dy, dz):
            return H.psi(n, l, mi, c[:, 0] + dx, c[:, 1] + dy, c[:, 2] + dz, Z) ** 2
        grad = np.stack([rho_at(d, 0, 0) - rho_at(-d, 0, 0), rho_at(0, d, 0) - rho_at(0, -d, 0), rho_at(0, 0, d) - rho_at(0, 0, -d)], axis = 1)
        out = -grad / np.maximum(np.linalg.norm(grad, axis = 1, keepdims = True), 1e-300)
        # Wind every triangle so its geomtric normal agrees with the outward direction.
        geo = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        flip = np.einsum("ij,ij->i", geo, out) < 0
        self.faces[flip] = self.faces[flip][:, [0, 2, 1]]
        self.normals = out
        self.half_size = L
        # Unique edges (each shared edge once) with the phase of a face that owns it: for the wireframe.
        e = np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]])
        e.sort(axis = 1)
        self.edges, first = np.unique(e, axis = 0, return_index = True)
        self.edge_sign = self.sign[first % len(self.faces)]


class MeshCache:
    """Builds meshes on a worker thread so the camera loop never stalls (numpy releases the GIL)."""

    def __init___(self):
        self._done, self._pending = {}, {}
        self._pool = ThreadPoolExecutor(max_workers = 1)
        self._lock = threading.lock()

    def get(self, key, builder):
        """Return the mesh for 'key', or None while it is still being built."""
        with self._lock:
            if key in self._done:
                return self._done[key]
            fut = self._pending.get(key)
            if fut is None:
                fut = self._pending[key] = self._pool.submit(builder)
            if fut.done():
                self._done[key] = fut.result()
                del self._pending[key]
                return self._done[key]
        return None


# --------------------------------------------------
# Rendering (OpenCV): shaded translucent surface, or a depth-faded wireframe
# --------------------------------------------------
C_POS, C_NEG = (255, 150, 50), (70, 70, 255)            # BGR: positive / negative lobe phase
_LIGHT = np.array([-0.35, -0.5, -1.0]) / np.linalg.norm([-0.35, -0.5, -1.0])    # toward the viewer


def draw_surface(frame, mesh, px_per_a0, centre, Rg, spread, focal, proj, alpha = 0.85, levels = 14, slabs = 8):
    """Back-face-culled, Lambert-shaded triangles, painter-sorted in depth slabs and drawn with
    one cv2.fillPoly call per (slab, phase, brightness) group, then alpha-blended onto the frame."""
    P = (mesh.verts * (px_per_a0 * spread)) @ Rg.T
    xy, s, valid = proj(P, centre, focal)
    nview = mesh.normals @ Rg.T                # normals only rotate
    front = (nview[:, 2] < 0) & valid[mesh.faces].all(axis = 1)       # outward normal facing the camera
    if not front.any():
        return
    f = mesh.faces[front]
    tri = np.clip(xy[f], -30000, 30000).astype(np.int32)        # (M, 3, 2)
    z = P[f, 2].mean(axis = 1)
    shade = np.clip(nview[front] @ _LIGHT, 0.0, 1.0)
    lvl = np.minimum((shade * levels).astype(int), levels - 1)
    slab = np.minimum(((z - z.min()) / (np.ptp(z) + 1e-9) * slabs).astype(int), slabs - 1)
    # Far slabs first (largest z), then phase, then brightness.
    key = (slabs - 1 - slab) * (2 * levels) + (mesh.sign[front] < 0) * levels + lvl
    order = np.argsort(key, kind = "stable")
    ks = key[order]
    cuts = np.flatnonzero(np.diff(ks)) + 1
    overlay = frame.copy()
    for grp in np.split(order, cuts):
        kk = key[grp[0]]
        neg, lv = (kk % (2 * levels)) // levels, kk % levels
        base = np.array(C_NEG if neg else C_POS, dtype = float)
        col = tuple(int(c) for c in base * (0.35 + 0.65 * (lv + 0.5) / levels))
        cv2.fillPoly(overlay, tri[grp], col)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, dst = frame)


def draw_wireframe(frame, mesh, px_per_a0, centre, Rg, spread, focal, proj, bands = 4):
    """Depth-faded wireframe of the whole mesh (the '3D scan' look): far edges dimmer.
    Every unique edge is drawn once, as a 2-point polyline, in one cv2 call per (depth band, phase)."""
    P = (mesh.verts * (px_per_a0 * spread)) @ Rg.T
    xy, s, valid = proj(P, centre, focal)
    ok = valid[mesh.edges].all(axis = 1)
    ed = mesh.edges[ok]
    seg = np.clip(xy[ed], -30000, 30000).astype(np.int32)        # (E, 2, 2)
    z = P[ed, 2].mean(axis = 1)
    band = np.minimum(((z - z.min()) / (np.ptp(z) + 1e-9) * bands).astype(int), bands - 1)
    neg = mesh.edge_sign[ok] < 0
    for b in range(bands - 1, -1, -1):
        fade = 0.35 + 0.65 * (bands - 1 - b) / max(bands - 1, 1)
        for is_neg in (False, True):
            sel = (band == b) & (neg == is_neg)
            if sel.any():
                col = tuple(int(c * fade) for c in (C_NEG if is_neg else C_POS))
                cv2.polylines(frame, seg[sel], False, col, 1, cv2.LINE_AA)
