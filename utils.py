import numpy as np

FLOAT_TOL = 1e-6


class BoundingBox:
    def __init__(self, min_coord, max_coord):
        self.verify_data(min_coord)
        self.verify_data(max_coord)
        #
        self.min_coord = min_coord
        self.max_coord = max_coord

    @staticmethod
    def verify_data(x):
        assert isinstance(x, np.ndarray)

    @staticmethod
    def create(coords: np.ndarray):
        BoundingBox.verify_data(coords)
        min_coord = coords.min(axis=0)
        max_coord = coords.max(axis=0)
        return BoundingBox(min_coord, max_coord)

    def get_center(self):
        return (self.min_coord + self.max_coord) * 0.5

    def get_diag(self):
        # diagonal dimension
        return np.linalg.norm(self.max_coord - self.min_coord)


def verify_type(other, types):
    if not isinstance(other, types):
        raise TypeError(f"Unexpected type: {type(other)}. Use: {types} instead")


class Vector:
    def __init__(self, vs: np.ndarray=None):
        if vs is None:
            self.vs = np.zeros(3)
        else:
            self.vs = vs
            self.verify_data()

    def verify_data(self):
        assert isinstance(self.vs, np.ndarray)
        assert self.vs.shape == (3,)

    def norm(self) -> float:
        return np.linalg.norm(self.vs)

    def unit(self):
        return Vector(self.vs / self.norm())

    def is_unit(self):
        return abs(self.norm() - 1) <= FLOAT_TOL

    def __repr__(self):
        return f"{self.__class__.__name__}{tuple(self.vs)}"

    def __getitem__(self, i):
        return self.vs[i]

    def __mul__(self, other: (float, int)):
        verify_type(other, (float, int))
        return Vector(self.vs * other)

    __rmul__ = __mul__

    def __truediv__ (self, other):
        verify_type(other, (float, int))
        return Vector(self.vs / other)

    def __neg__(self):
        return Vector(self.vs * -1)

    def __add__(self, other):
        verify_type(other, Vector)
        return Vector(self.vs + other.vs)

    def __sub__(self, other):
        verify_type(other, Vector)
        return Vector(self.vs + other.vs * -1)

    def array(self):
        return self.vs.__copy__()

    def dot(self, other):
        verify_type(other, Vector)
        return np.dot(self.vs, other.vs)

    def cross(self, other):
        verify_type(other, Vector)
        return Vector(np.cross(self.vs, other.vs))

    def v4(self):
        return np.array([*self.vs, 1])


class Direction(Vector):
    '''
    Direction is a subset of Vector, where it is an unit vector
    '''
    def __new__(cls, ds: np.ndarray):
        return Vector(ds).unit()


class Quaternion:
    def __init__(self, qs: np.ndarray=None):
        if qs is None:
            self.qs = np.array([1., *np.zeros(3)])
        else:
            self.qs = qs
            self.verify_data()

    def verify_data(self):
        assert isinstance(self.qs, np.ndarray)
        assert self.qs.shape == (4,)

    def norm(self) -> float:
        return np.linalg.norm(self.qs)

    def unit(self):
        return Quaternion(self.qs / self.norm())

    def __repr__(self):
        return f"{self.__class__.__name__}{tuple(self.qs)}"

    @staticmethod
    def create_from_axis_angle(v: Vector, theta: float):
        # theta: radian
        verify_type(v, Vector)
        qs = np.array([np.cos(theta*0.5), *(v.unit() * np.sin(theta*0.5))])
        # which is already an unit quaternion
        return Quaternion(qs)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w1, x1, y1, z1 = self.qs
            w2, x2, y2, z2 = other.qs
            qs = (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
                w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
            )
            return Quaternion(np.array(qs))
        elif isinstance(other, Vector):
            return self.rotate(other)

        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def rotate(self, v):
        verify_type(v, Vector)

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

        return Vector(np.array([vp0, vp1, vp2]))

    @staticmethod
    def convert_to_matrix(q):
        verify_type(q, Quaternion)
        qw, qx, qy, qz = q.qs
        m = np.array([
            [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw, 0.0],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw, 0.0],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2, 0.0],
            [0.0, 0.0, 0.0, 1.0]])
        return m

    def matrix(self):
        return Quaternion.convert_to_matrix(self)

    def conjugate(self):
        qs = np.array([self.qs[0], *(self.qs[1:] * -1)])
        return Quaternion(qs)

    def inv(self):
        return self.conjugate()


class Transformation:
    '''
    Linear transformation:
        Orthogonal matrix of rotation (represented by Quaternion)
        Vector of translation
        Factor of volume scale
    '''
    def __init__(self, r: Quaternion=None, t: Vector=None, s: (float, int)=None):
        if r is None:
            self.r = Quaternion()
        else:
            verify_type(r, Quaternion)
            self.r = r
        if t is None:
            self.t = Vector()
        else:
            self.t = t
            verify_type(t, Vector)
        if s is None:
            self.s = 1.
        else:
            verify_type(s, (float, int))
            self.s = s

    def __repr__(self):
        return f"{self.__class__.__name__}({self.r}, {self.t}, Scale({self.s}))"

    def __mul__(self, other):
        if isinstance(other, Point):
            return self.s * (self.r*other + self.t)

        if isinstance(other, Vector):
            return self.s * (self.r*other)

        if isinstance(other, Direction):
            return self.r*other

        if isinstance(other, Transformation):
            s = self.s * other.s
            r = self.r * other.r
            t = self.r * other.t + self.t / other.s
            return Transformation(r=r, t=t, s=s)

    def inv(self):
        s = 1. / self.s
        r = self.r.inv()
        t = -self.s * (r * self.t)
        return Transformation(r=r, t=t, s=s)

    @staticmethod
    def convert_to_matrix(tf):
        verify_type(tf, Transformation)
        '''
        [[s.r, s.t], [0, 1]]
        '''
        m = np.eye(4)
        m[:3, :3] = tf.r.matrix()[:3, :3] * tf.s
        m[:3, 3] = (tf.t * tf.s).array()
        return m

    def matrix(self):
        return self.convert_to_matrix(self)


# v = Vector(np.array([1, 1, 1]))
# v1 = Vector(np.array([1, 1, 1]))
# q = Quaternion.create_from_axis_angle(v, 30 * np.pi / 180)
# q_null = Quaternion.create_default()
#
# print(q)
# print(q_null)
# print(q_null.rotate(v))
# print(q*q_null)
# print(q_null * q)

# axis = Vector(np.array([0, 1, 0]))
# angle_1 = 30 * np.pi/180
# angle_2 = 60 * np.pi/180
# q1 = Quaternion.create_from_axis_angle(axis, angle_1)
# q2 = Quaternion.create_from_axis_angle(axis, angle_2)
#
# v = Vector(np.array([0, 0, 1]))
# q3 = q2 * q1
# u = q3 * v
# print(u)
#
# print(q3.inv().rotate(u))
#
# print(Quaternion().matrix())



# v = Vector(np.array([0, 0, 1]))
# T1 = Transformation()
# T2 = Transformation()
# print(T1 * v)

# axis = Vector(np.array([0, 1, 0]))
# angle_1 = 30 * np.pi/180
# q1 = Quaternion.create_from_axis_angle(axis, angle_1)
# t = Vector(np.array([1, 1, 1]))
# tf = Transformation(r=q1, t=t)
# print(tf)
# v = Vector(np.array([0, 0, 1]))
# u = tf * v
# print(u)
# print(tf.matrix())
# print(tf.inv().matrix())
# print(tf.inv().matrix().dot(tf.matrix()))
