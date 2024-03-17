# Copyright (C) 2015 Akselos
import sys

import numpy as np

import point as pt


# A class that represents the bounding box of an object.

class BoundingBox:
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max
        assert (min is None) == (max is None)
        assert min is None or isinstance(min, pt.Point)
        assert max is None or isinstance(max, pt.Point)

    @staticmethod
    def create(points):
        '''Create a bounding box that contains a list of points.'''
        if len(points) == 0:
            return BoundingBox()

        if not isinstance(points, np.ndarray):
            pts = np.array([np.array(point) for point in points])
        else:
            pts = points
        return BoundingBox(pt.Point(pts.min(0)), pt.Point(pts.max(0)))

    def __add__(self, other):
        if self.is_empty():
            return other
        if other.is_empty():
            return self

        min = pt.Point(np.minimum(self.min.ps, other.min.ps))
        max = pt.Point(np.maximum(self.max.ps, other.max.ps))
        return BoundingBox(min, max)

    def __rmul__(self, transformation):
        assert isinstance(transformation, pt.Transformation)

        if self.corners() is None:
            return self

        return BoundingBox.create([transformation * p for p in self.corners()])

    def get_scaled(self, factor):
        if self.is_empty():
            return self

        center = self.center()
        v = (factor / 2.0) * (self.max - self.min)
        new_min = center - v
        new_max = center + v
        new_bounding_box = BoundingBox(new_min, new_max)
        return new_bounding_box

    def get_center(self):
        return (self.min + self.max) * 0.5

    def get_diag(self):
        # diagonal dimension
        return np.linalg.norm(self.max - self.min)

    def radius(self):
        if self.min is None:
            return 0.0

        dm = 0.5 * (self.max - self.min)
        return float(np.sqrt(dm[0] ** 2 + dm[1] ** 2 + dm[2] ** 2))

    def is_empty(self):
        return self.min is None or self.max is None

    def center(self):
        if self.is_empty():
            return pt.Point([0.0, 0.0, 0.0])

        return pt.Point((self.min.ps + self.max.ps) / 2.0)

    def corners(self):
        if self.is_empty():
            return None

        a, b = self.min, self.max
        points = [(a[0], a[1], a[2]), (a[0], a[1], b[2]), (a[0], b[1], a[2]),
                  (a[0], b[1], b[2]), (b[0], a[1], a[2]), (b[0], a[1], b[2]),
                  (b[0], b[1], a[2]), (b[0], b[1], b[2])]
        return [pt.Point(point) for point in points]

    # Returns the plane original and normal for the bounding planes of this box
    def planes(self):
        corners = self.corners()
        # Hack to avoid issues with 2D or 1D bounding boxes
        if all(c[0] == corners[0][0] for c in corners):
            corners_to_keep = [corners[i] for i in [0, 1, 4, 5]]
            corners_to_push = [corners[i] for i in [2, 3, 6, 7]]
            corners = corners_to_keep + [c + pt.Vector((1e5, 0, 0)) for c in corners_to_push]
        if all(c[1] == corners[0][1] for c in corners):
            corners_to_keep = [corners[i] for i in [0, 1, 2, 3]]
            corners_to_push = [corners[i] for i in [4, 5, 6, 7]]
            corners = corners_to_keep + [c + pt.Vector((0, 1e5, 0)) for c in corners_to_push]
        if all(c[2] == corners[0][2] for c in corners):
            corners_to_keep = [corners[i] for i in [0, 2, 4, 6]]
            corners_to_push = [corners[i] for i in [1, 3, 5, 7]]
            corners = corners_to_keep + [c + pt.Vector((0, 0, 1e5)) for c in corners_to_push]
        dir_a = pt.Direction()
        if (corners[0] - corners[2]).mag() != 0:
            dir_a = (corners[0] - corners[2]).normalize()
        dir_b = pt.Direction()
        if (corners[0] - corners[4]).mag() != 0:
            dir_b = (corners[0] - corners[4]).normalize()
        dir_c = pt.Direction()
        if (corners[0] - corners[1]).mag() != 0:
            dir_c = (corners[0] - corners[1]).normalize()
        planes_origin_normal = []
        planes_origin_normal.append((
            0.25 * (corners[0] + corners[1] + corners[4] + corners[5]), dir_a))
        planes_origin_normal.append((
            0.25 * (corners[2] + corners[3] + corners[6] + corners[7]), dir_a))
        planes_origin_normal.append((
            0.25 * (corners[0] + corners[1] + corners[2] + corners[3]), dir_b))
        planes_origin_normal.append((
            0.25 * (corners[4] + corners[5] + corners[6] + corners[7]), dir_b))
        planes_origin_normal.append((
            0.25 * (corners[0] + corners[2] + corners[4] + corners[6]), dir_c))
        planes_origin_normal.append((
            0.25 * (corners[1] + corners[3] + corners[5] + corners[7]), dir_c))
        return planes_origin_normal

    def intersects(self, other):
        # This function isn't used anywhere and hasn't been tested.
        if self.is_empty() or other.is_empty():
            return False

        return not (np.any(self.max.ps < other.min.ps) or np.any(other.max.ps < self.min.ps))

    # Returns the plane equation coefficients for the bounding planes of this box
    def plane_coefficients(self):
        planes_origin_normal = self.planes()
        planes_coeffs = []
        for origin, normal in planes_origin_normal:
            plane_coeff = [normal[0], normal[1], normal[2], -pt.Vector(normal).dot(pt.Vector(origin))]
            planes_coeffs.append(plane_coeff)
        return planes_coeffs

    # Returns the vertices of the intersection polygon formed by intersecting this box with the
    # given plane
    def get_plane_intersection_polygon(self, origin, normal):
        bounding_plane_coeffs = self.plane_coefficients()
        normal = pt.Direction(normal)
        plane_coeff = [normal[0], normal[1], normal[2], -pt.Vector(normal).dot(pt.Vector(origin))]
        plane_pairs = [(0, 4), (4, 1), (1, 5), (5, 0), (0, 3), (3, 1), (1, 2), (2, 0), (5, 3), (3, 4), (4, 2), (2, 5)]
        intersection_points = []
        for index_0, index_1 in plane_pairs:
            matrix_a = [bounding_plane_coeffs[index_0][0:3], bounding_plane_coeffs[index_1][0:3],
                        plane_coeff[0:3]]
            vector_c = [bounding_plane_coeffs[index_0][3], bounding_plane_coeffs[index_1][3],
                        plane_coeff[3]]
            a = np.array(matrix_a)
            if np.linalg.cond(a) > 1 / sys.float_info.epsilon:
                continue
            b = np.array(vector_c)
            x = np.linalg.solve(a, b)
            # why do we need this -1, why!
            p = -1 * pt.Point(x)
            if self.is_inside(p):
                intersection_points.append(p)
        return pt.sort_points_by_angle(intersection_points, pt.Direction(normal))

    # Returns true if this box intersects with the given plane, false otherwise
    def test_plane_intersection(self, origin, normal):
        corners = self.corners()
        positive_count = 0
        negative_count = 0
        normal = pt.Direction(normal)
        for c in corners:
            d = pt.Vector(c).dot(pt.Vector(normal)) - pt.Vector(origin).dot(pt.Vector(normal))
            positive_count = positive_count + 1 if d >= 0 else positive_count
            negative_count = negative_count + 1 if d < 0 else negative_count
        return positive_count * negative_count != 0

    def is_inside(self, point):
        return (point[0] >= self.min[0] and point[0] <= self.max[0] and
                point[1] >= self.min[1] and point[1] <= self.max[1] and
                point[2] >= self.min[2] and point[2] <= self.max[2])

    def correct_very_small_spans(self, tol=1e-2):
        if self.is_empty():
            return
        spans = np.abs(self.max - self.min)
        largest_span = np.max(spans)
        for span_idx in range(len(spans)):
            if spans[span_idx] >= tol * largest_span:
                continue
            if self.min.ps[span_idx] <= self.max.ps[span_idx]:
                self.min.ps[span_idx] -= 0.5 * (tol * largest_span - spans[span_idx])
                self.max.ps[span_idx] += 0.5 * (tol * largest_span - spans[span_idx])
            else:  # can this case happen?
                self.min.ps[span_idx] += 0.5 * (tol * largest_span - spans[span_idx])
                self.max.ps[span_idx] -= 0.5 * (tol * largest_span - spans[span_idx])

    def __repr__(self):
        return "BoundingBox(" + ",".join([repr(self.min), repr(self.max)]) + ")"

    __str__ = __repr__
