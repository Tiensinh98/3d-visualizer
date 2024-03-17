# Copyright (C) 2015 Akselos
import math
import numbers

import numpy as np


# This module holds various types of 3-vectors: Direction, Point and Vector.
# They differ in how they are affected by multiplication by a Transformation:
# directions rotate; vectors rotate and scale; points rotate,
# scale and translate.


class Direction(object):
    __slots__ = 'ds',

    def __init__(self, d=None):
        if d is None:
            self.ds = np.array([1.0, 0.0, 0.0])
            return
        if isinstance(d, Direction):
            self.ds = d.ds.copy()
            return
        assert len(d) == 3
        ds = norm(d)
        assert not np.isnan(ds).any()
        self.ds = ds

    @staticmethod
    def try_create(ds, eps=1e-6):
        if np.linalg.norm(np.array(ds)) < eps:
            return None

        return Direction(ds)

    def perpendicular(self, axis_hint_dir=None):
        if axis_hint_dir is not None:
            hint_v = Vector(axis_hint_dir)
            self_v = Vector(self.ds)
            perp = Direction.try_create(-self_v.cross(self_v.cross(hint_v)))
        else:
            perp = None

        if perp is None:
            ads = [abs(d) for d in self.ds]
            if ads[0] >= ads[1] and ads[0] >= ads[2]:
                other = (0, 1, 0)
            elif ads[1] >= ads[0] and ads[1] >= ads[2]:
                other = (0, 0, 1)
            else:
                other = (1, 0, 0)

            perp = Direction(np.cross(self.ds, other))

        return perp

    def dot(self, other):
        return float(np.dot(self.ds, other.ds))

    def cross_dir(self, other):
        return Direction(np.cross(self.ds, other.ds))

    def cross_vec(self, other):
        return Vector(np.cross(self.ds, other.ds))

    def mul_sign(self, sign):
        assert sign == 1 or sign == -1
        return Direction(sign * self.ds)

    def decompose(self, other):
        n = Direction(other)
        parallel = self.dot(n) * n
        perpendicular = 1.0 * self - parallel
        return parallel, perpendicular

    def __mul__(self, other):
        d = float(other)
        return Vector(d * self.ds)

    def __neg__(self):
        # Use copy-constructor to avoid `norm()`.
        d = Direction(self)
        d.ds *= -1
        return d

    __rmul__ = __mul__

    def __getitem__(self, i):
        if i < 0 or i >= 3:
            raise IndexError(i)
        return self.ds[i]

    def __len__(self):
        return 3

    def __str__(self):
        return f'Direction({self.ds})'

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        assert isinstance(other, Direction)
        return np.all(self.ds == other.ds)

    def __hash__(self):
        return hash((hash(self[0]), hash(self[1]), hash(self[2])))

    def __array__(self):
        return np.array(self.ds)


class Point:
    """
    A 3D point
    """
    __slots__ = 'ps',

    def __init__(self, p=None):
        if p is None:
            self.ps = np.zeros(3)
            return
        if isinstance(p, Point):
            self.ps = p.ps.copy()
            return
        assert len(p) == 3
        self.ps = np.array([float(i) for i in p])

    def mid_point(self, other):
        """

        :param other: a Point
        :return: The Point at the mid point between this Point and `other`
        """
        return 0.5 * Point(self.ps + other.ps)

    def __neg__(self):
        return Point(-self.ps)

    def __add__(self, other):
        if isinstance(other, Vector):
            return Point(self.ps + other.vs)
        # AE: Its useful to add points sometimes, e.g. to calculate a centroid
        # BS: To do this you should subtract the origin from the point to get a vector, then add
        # the other point.  Unfortunately, I can't remove the case below because I don't know
        # everywhere it might now be used.
        # CK: Flag for ValueError when these cases are found
        # BS: If we raise ValueError in this case then existing code may crash so it's not
        # really worth it.
        if isinstance(other, Point):
            return Point(self.ps + other.ps)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Point):
            return Vector(self.ps - other.ps)
        if isinstance(other, Vector):
            return Point(self.ps - other.vs)

        raise NotImplementedError

    def norm(self):
        """

        :return: The euclidean norm of this Point to the origin
        """
        return np.linalg.norm(self.ps)

    def __mul__(self, other):
        p = Point(self)
        p.ps *= float(other)
        return p

    def __rmul__(self, other):
        return self * float(other)

    def __getitem__(self, i):
        if i < 0 or i >= 3:
            raise IndexError

        return self.ps[i]

    def __array__(self):
        return np.array(self.ps)

    def __len__(self):
        return 3

    def __str__(self):
        return f'Point({self.ps})'

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self[0] == other[0] and self[1] == other[1] \
               and self[2] == other[2]

    def __hash__(self):
        return hash((hash(self[0]), hash(self[1]), hash(self[2])))


class Vector:
    __slots__ = 'vs',

    def __init__(self, v=None):
        if v is None:
            self.vs = np.zeros(3)
            return
        if isinstance(v, Vector):
            self.vs = v.vs.copy()
            return
        assert len(v) == 3
        self.vs = np.array([float(i) for i in v])

    def dot(self, other):
        return np.dot(self.vs, other.vs)

    def cross(self, other):
        return Vector(np.cross(self.vs, other.vs))

    def mag(self):
        return np.linalg.norm(self.vs)

    def normalize(self):
        return Direction(self.vs)

    def norm(self):
        return np.linalg.norm(self.vs)

    def __mul__(self, other):
        v = Vector(self)
        v.vs *= float(other)
        return v

    def __rmul__(self, other):
        return self * float(other)

    def __neg__(self):
        v = Vector(self)
        v.vs *= -1
        return v

    def __add__(self, other):
        if isinstance(other, Vector):
            v = Vector(other)
            v.vs += self.vs
            return v
        if isinstance(other, Point):
            p = Point(other)
            p.ps += self.vs
            return p
        return NotImplemented

    def __getitem__(self, i):
        if i < 0 or i >= 3:
            raise IndexError
        return self.vs[i]

    def __sub__(self, other):
        return self + -other

    def __len__(self):
        return 3

    def __str__(self):
        return f'Vector({self.vs})'

    def __array__(self):
        return np.array(self.vs)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        assert isinstance(other, Vector)
        return np.all(self.vs == other.vs)

    def __hash__(self):
        return hash((hash(self[0]), hash(self[1]), hash(self[2])))


def norm(tup):
    mag = np.linalg.norm(tup)
    if mag < 1e-20:
        return np.array([0., 0., 0.])
    return np.array(tup) / mag


class Quaternion(object):
    __slots__ = 'qs',

    def __init__(self, q=None):
        if q is None:
            self.qs = np.array((1.0, 0.0, 0.0, 0.0))
            return
        # Remark that `norm()` is quite an expensive operation. So once the Quaternion
        # is created with norm=1, the copy-constructor can skip normalization.
        if isinstance(q, Quaternion):
            self.qs = q.qs.copy()
            return
        assert len(q) == 4
        qs = norm(q)
        assert not np.isnan(qs).any()
        if qs[0] < 0:
            qs = -qs
        self.qs = qs

    @staticmethod
    def from_axis_angle(axis, angle):
        sin_half_angle = np.sin(angle / 2)
        qx = axis[0] * sin_half_angle
        qy = axis[1] * sin_half_angle
        qz = axis[2] * sin_half_angle
        qw = np.cos(angle / 2)
        return Quaternion((qw, qx, qy, qz))

    @staticmethod
    def from_axis_name(axis_name: str):
        """
        Create instance of Quaternion readily usable for camera when axis name
        is given.
        For X/Y axis, oriented such that Z is vertical.
        """
        if axis_name == '+x':
            return Quaternion((0.5, -0.5, 0.5, 0.5))
        if axis_name == '-x':
            return Quaternion((0.5, -0.5, -0.5, -0.5))
        if axis_name in ('+y', '-y'):
            half_sqrt_two = 2 ** -0.5
            if axis_name == '+y':
                return Quaternion((half_sqrt_two, -half_sqrt_two, 0, 0))
            return Quaternion((0, 0, half_sqrt_two, half_sqrt_two))
        if axis_name == '+z':
            return Quaternion((0, 0, 1, 0))
        if axis_name == '-z':
            return Quaternion((1, 0, 0, 0))
        if axis_name == 'isometric':
            # CK: such that the triad form a equilateral triangle,
            #  with Z at the top, X at lower-left, Y at lower-right.
            return Quaternion([0.327, -0.171, -0.430, -0.824])

        raise ValueError(f'Unknown axis name: {axis_name}')

    @staticmethod
    def dir_to_dir(dir1, dir2, axis_hint_dir=None):
        dir1 = Direction(dir1)
        dir2 = Direction(dir2)
        cos_theta = np.dot(dir1.ds, dir2.ds)
        if cos_theta <= -1.0 + 1e-12 or axis_hint_dir is not None and cos_theta <= -1.0 + 1e-6:
            perp = dir1.perpendicular(axis_hint_dir)
            return Quaternion([0.0] + list(perp.ds))
        elif cos_theta >= 1.0 - 1e-12:
            return Quaternion()

        theta = np.arccos(cos_theta)
        cs, sn = np.cos(theta / 2.0), np.sin(theta / 2.0)
        qs = [cs] + list(sn * norm(np.cross(dir1.ds, dir2.ds)))
        return Quaternion(qs)

    @staticmethod
    def create_random():
        u1, u2, u3 = np.random.rand(), np.random.rand(), np.random.rand()
        s1, s2 = np.sqrt(1 - u1), np.sqrt(u1)
        q = Quaternion((s1 * np.sin(2 * np.pi * u2),
                        s1 * np.cos(2 * np.pi * u2),
                        s2 * np.sin(2 * np.pi * u3),
                        s2 * np.cos(2 * np.pi * u3)))
        return q

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w1, x1, y1, z1 = self.qs
            w2, x2, y2, z2 = other.qs
            tup = (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                   w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
                   w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2)
            q = Quaternion()
            q.qs = np.array(tup)
            return q
        return NotImplemented

    def rotate(self, v):
        qw, qx, qy, qz = self.qs

        # These computations repeat 3 times, re-use them.
        qw2 = qw * qw
        qx2 = qx * qx
        qy2 = qy * qy
        qz2 = qz * qz

        vp0 = (qw2 + qx2 - qy2 - qz2) * v[0] + \
              2 * (qx * (qy * v[1] + qz * v[2]) + qw * (qy * v[2] - qz * v[1]))
        vp1 = (qw2 - qx2 + qy2 - qz2) * v[1] + \
              2 * (qy * (qz * v[2] + qx * v[0]) + qw * (qz * v[0] - qx * v[2]))
        vp2 = (qw2 - qx2 - qy2 + qz2) * v[2] + \
              2 * (qz * (qx * v[0] + qy * v[1]) + qw * (qx * v[1] - qy * v[0]))
        if isinstance(v, Point):
            return Point((vp0, vp1, vp2))
        if isinstance(v, Vector):
            return Vector((vp0, vp1, vp2))
        if isinstance(v, Direction):
            return Direction((vp0, vp1, vp2))
        assert False

    @staticmethod
    def qs_to_matrix(qs):
        qw, qx, qy, qz = qs
        m = np.array([
            [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw, 0.0],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw, 0.0],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2, 0.0],
            [0.0, 0.0, 0.0, 1.0]])
        return m

    def matrix(self):
        return Quaternion.qs_to_matrix(self.qs)

    def inv(self):
        # Avoid calling the `norm()` function by relying on copy-constructor.
        q = Quaternion(self)
        q.qs[0] *= -1
        return q

    def __str__(self):
        return f'Quaternion({self.qs})'

    def __repr__(self):
        return str(self)


class Transformation:
    __slots__ = 'r', 't', 's'

    '''A linear transformation, represented as rotate, translate, scale,
    applied in that order.'''

    def __init__(self, rotation=None, translation=None, scale=1.0):
        self.r = Quaternion(rotation)
        self.t = Vector(translation)
        self.s = float(scale)

    @staticmethod
    def scaling(s):
        return Transformation(None, None, s)

    @staticmethod
    def translation(dv):
        return Transformation(None, Vector(dv))

    @staticmethod
    def rotation(angle, axis, center=None):
        axis = Direction(axis)
        center = Point(center)
        c, s = math.cos(angle / 2.0), math.sin(angle / 2.0)
        t = Transformation.translation(Point() - center)
        q = Transformation(Quaternion((c, s * axis[0], s * axis[1], s * axis[2])))
        tp = Transformation(t.r, -t.t, t.s)  # previous computation reusable.
        return tp * q * t

    @staticmethod
    def reflection(origin, direction):
        t = Transformation.translation(Point() - Point(origin))
        s = Transformation.scaling(-1.0)
        r = Transformation.rotation(np.pi, direction)
        ref = t.inv() * s * r * t
        return ref

    @staticmethod
    def direction_alignment(dir1, dir2, origin=None, axis_hint_dir=None):
        if origin is None:
            origin = (0.0, 0.0, 0.0)
        rotation = Quaternion.dir_to_dir(dir1, dir2, axis_hint_dir)
        to_origin = Transformation.translation(origin).inv()
        final = to_origin.inv() * Transformation(rotation) * to_origin
        return final

    @staticmethod
    def from_matrix(m):
        assert np.linalg.det(m[:3, :3]) > 0.9
        t = m[0, 0] + m[1, 1] + m[2, 2]
        if t > 0:
            r = np.sqrt(1 + t)
            s = 0.5 / r
            q = [0] * 4
            q[0] = 0.5 * r
            q[1] = (m[2, 1] - m[1, 2]) * s
            q[2] = (m[0, 2] - m[2, 0]) * s
            q[3] = (m[1, 0] - m[0, 1]) * s
        else:
            m0, m1, m2 = m[0, 0], m[1, 1], m[2, 2]
            if m0 >= m1 and m0 >= m2:
                i0, i1, i2 = 0, 1, 2
            elif m1 >= m0 and m1 >= m2:
                i0, i1, i2 = 1, 2, 0
            else:
                i0, i1, i2 = 2, 0, 1

            q = [0] * 4
            ss = max(1.0 + m[i0, i0] - m[i1, i1] - m[i2, i2], 0.0)
            r = np.sqrt(ss)
            s = 0.5 / r
            q[0] = (m[i2, i1] - m[i1, i2]) * s
            q[i0 + 1] = 0.5 * r
            q[i1 + 1] = (m[i0, i1] + m[i1, i0]) * s
            q[i2 + 1] = (m[i2, i0] + m[i0, i2]) * s
        return Transformation(q, m[:3, 3], 1 / m[3, 3])

    def __mul__(self, other):
        if isinstance(other, Point):
            return self.s * (self.r.rotate(other) + self.t)
        if isinstance(other, Vector):
            return self.s * self.r.rotate(other)
        if isinstance(other, Direction):
            return self.r.rotate(other)
        if isinstance(other, Transformation):
            return Transformation(self.r * other.r,
                                  self.r.rotate(other.t) + 1.0 / other.s * self.t,
                                  self.s * other.s)
        if isinstance(other, numbers.Number):
            return self.s * other

        return NotImplemented

    def inv(self):
        qp = self.r.inv()
        vp = -self.s * qp.rotate(self.t)
        sp = 1 / self.s
        return Transformation(qp, vp, sp)

    def matrix(self):
        m = np.array(self.r.matrix())
        m[0, 3] = self.t.vs[0]
        m[1, 3] = self.t.vs[1]
        m[2, 3] = self.t.vs[2]
        m[3, 3] = 1.0 / self.s
        return m

    def scale(self):
        return self.s

    def __str__(self):
        return f'Transformation({self.r}, {self.t}, {self.s})'

    def __repr__(self):
        return str(self)


# shorthand:
T = Transformation


def orthogonalize(w):
    u, s, v = np.linalg.svd(w, full_matrices=True)
    new_s = np.identity(len(s))
    m = np.dot(u, np.dot(new_s, v))
    return m


def get_two_most_perpendicular(direction, transformation):
    max_dot, max_i = None, None
    identity = np.identity(3)
    basis = [transformation * Direction(identity[i]) for i in range(3)]
    for i in range(3):
        dot = abs(direction.dot(basis[i]))
        if max_dot is None or dot > max_dot:
            max_dot, max_i = dot, i
    return basis[(max_i + 1) % 3], basis[(max_i + 2) % 3]


def centroid(points):
    c = Point()
    for p in points:
        c = c + p
    return (1.0 / len(points)) * c


# Signed angle between two vectors given a normal vector for direction
def angle(v1, v2, vn):
    v1_n = v1.normalize()
    v2_n = v2.normalize()
    cos_v1v2 = round(v1_n.dot(v2_n), 6)
    angle = math.acos(cos_v1v2)
    if not utils.is_close(abs(cos_v1v2), 1.0):
        cross_v1v2 = v1_n.cross_dir(v2_n)
        if vn.dot(cross_v1v2) < 0:
            angle = -angle
    return angle


# Sort by angle a list of points, using the centroid as origin and a normal direction
def sort_points_by_angle(points, normal):
    if len(points) <= 1:
        return points
    c = centroid(points)
    ref_point = points[0]
    return sorted(points, key=lambda p: angle(ref_point - c, p - c, normal))


def distance_to_line(point, line_point, line_normal):
    p_2_lp = Vector((point - line_point))
    projection_length = p_2_lp.dot(line_normal)
    projection = Vector(projection_length * line_normal)
    return (p_2_lp - projection).mag()
