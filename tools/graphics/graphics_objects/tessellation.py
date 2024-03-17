import numpy as np


def tessellate(vnxyz, uv_params, flip=False):
    nu, u0, u1, nv, v0, v1 = uv_params
    du = (u1 - u0) / nu
    dv = (v1 - v0) / nv

    nu1, nv1 = nu + 1, nv + 1
    n1 = nu1 * nv1
    vertices_uv = np.zeros((nu1, nv1, 3))
    normals_uv = np.zeros((nu1, nv1, 3))
    indices_uv = np.arange(n1, dtype=np.uint32).reshape((nu1, nv1))
    for iu in range(nu + 1):
        u = u0 + du * iu
        for iv in range(nv + 1):
            v = v0 + dv * iv
            vxyz, nxyz = vnxyz(u, v)
            vertices_uv[iu, iv] = vxyz
            normals_uv[iu, iv] = nxyz

    vertices = vertices_uv.reshape((n1, 3))
    normals = normals_uv.reshape((n1, 3))
    element_indices = np.zeros((2 * n1, 3), dtype=np.uint32)
    i = 0
    for iu in range(nu):
        for iv in range(nv):
            if flip:
                i0, i1, i2 = 2, 1, 0
            else:
                i0, i1, i2 = 0, 1, 2
            element_indices[i, i0] = indices_uv[iu, iv]
            element_indices[i, i1] = indices_uv[iu + 1, iv + 1]
            element_indices[i, i2] = indices_uv[iu, iv + 1]
            i += 1
            element_indices[i, i0] = indices_uv[iu, iv]
            element_indices[i, i1] = indices_uv[iu + 1, iv]
            element_indices[i, i2] = indices_uv[iu + 1, iv + 1]
            i += 1
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    return vertices, normals, element_indices


def tessellate_sphere():
    uv_params = 4, 0, np.pi, 8, 0, 2.0 * np.pi
    return tessellate(sphere_vnxyz, uv_params, flip=False)


def tessellate_cylinder(length_ratio):
    uv_params = 8, 0.0, 2.0 * np.pi, 1, 0.0, length_ratio
    return tessellate(cylinder_vnxyz, uv_params)


def tessellate_torus(radius_ratio):
    uv_params = 8, 0.0, 2.0*np.pi, 8, 0.0, 2.0*np.pi
    return tessellate(get_torus_vnxyz(radius_ratio), uv_params)


def tessellate_cone(radius_ratio):
    uv_params = 8, 0.0, 2.0*np.pi, 1, 0.0, 1.0
    return tessellate(get_cone_vnxyz(radius_ratio), uv_params)


def get_torus_vnxyz(r1):
    def vnxyz(u, v):
        vx = (r1 + np.cos(v)) * np.cos(u)
        vy = (r1 + np.cos(v)) * np.sin(u)
        vz = np.sin(v)
        tx = -np.sin(u)
        ty = np.cos(u)
        sx = -np.cos(u) * np.sin(v)
        sy = -np.sin(u) * np.sin(v)
        sz = np.cos(v)
        nx = ty * sz
        ny = -tx * sz
        nz = tx * sy - ty * sx
        return (vx, vy, vz), (nx, ny, nz)

    return vnxyz


def get_cone_vnxyz(r):
    def cone_vnxyz(u, v):
        vx = r * (1.0 - v) * np.cos(u)
        vy = r * (1.0 - v) * np.sin(u)
        vz = v
        l = np.hypot(1.0, r)
        c = 1.0 / l
        s = r / l
        nx = c * np.cos(u)
        ny = c * np.sin(u)
        nz = s
        return (vx, vy, vz), (nx, ny, nz)

    return cone_vnxyz


# Spring formulae courtesy of Paul Bourke: http://paulbourke.net/geometry/spring/
def get_spring_vnxyz(r1, r2, length, cycles):
    nu = 200
    nv = 20
    du = cycles * np.pi * 2 / nu
    dv = np.pi * 2 / nv

    def normalize(n):
        l = np.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
        if l != 0:
            return n / l
        return [0, 0, 0]

    def calc_normal(p, p1, p2):
        pa = [0, 0, 0]
        pb = [0, 0, 0]
        n = [0, 0, 0]

        pa[0] = p1[0] - p[0]
        pa[1] = p1[1] - p[1]
        pa[2] = p1[2] - p[2]
        pb[0] = p2[0] - p[0]
        pb[1] = p2[1] - p[1]
        pb[2] = p2[2] - p[2]
        pa = normalize(pa)
        pb = normalize(pb)

        n[0] = pa[1] * pb[2] - pa[2] * pb[1]
        n[1] = pa[2] * pb[0] - pa[0] * pb[2]
        n[2] = pa[0] * pb[1] - pa[1] * pb[0]
        return normalize(n)

    def eval_position(u, v):
        vx = ((1 - r1) * np.cos(v)) * np.cos(u)
        vy = ((1 - r1) * np.cos(v)) * np.sin(u)
        vz = r2 * (np.sin(v) + length * u / np.pi)
        return [vx, vy, vz]

    def spring_vnxyz(u, v):
        p = eval_position(u, v)
        n = calc_normal(p, eval_position(u + du / 10, v), eval_position(u, v + dv / 10))
        return (p[0], p[1], p[2]), (n[0], n[1], n[2])

    return spring_vnxyz


def cylinder_vnxyz(u, v):
    vx = np.cos(u)
    vy = np.sin(u)
    vz = v
    nx = np.cos(u)
    ny = np.sin(u)
    nz = 0.0
    return (vx, vy, vz), (nx, ny, nz)


def sphere_vnxyz(u, v):
    vx = np.sin(u) * np.cos(v)
    vy = np.sin(u) * np.sin(v)
    vz = np.cos(u)
    nx = vx
    ny = vy
    nz = vz
    return (vx, vy, vz), (nx, ny, nz)
