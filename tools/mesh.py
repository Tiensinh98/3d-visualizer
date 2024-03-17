import numpy as np

import lite_mesh_tools as lmt
import bounding_box as bb
import exo_data as ed


class Mesh:
    def __init__(self, mesh_data, vertices, indices, sedges_indices, field_names_scalar_values):
        self.mesh_data = mesh_data
        self.vertices = vertices
        self.indices = indices
        self.sedges_indices = sedges_indices
        self.field_names_scalar_values = field_names_scalar_values
        self.field_names = list(field_names_scalar_values.keys())
        self.is_solution_mesh = len(field_names_scalar_values)

    @staticmethod
    def create(exo_filepath):
        exo_data = ed.ExoData.read(exo_filepath)
        surface_exo_data, _ = lmt.get_surface_mesh(exo_data)
        vertices = np.ascontiguousarray(surface_exo_data.coords, dtype=np.float32)
        field_names_scalar_values = {f: v[0] for f, v in surface_exo_data.node_field_values.items()}
        indices = np.ascontiguousarray(surface_exo_data.exo_block_datas[0].elems, dtype=np.uint32)
        boundary_sedges = lmt.get_boundary_sedges(surface_exo_data)
        sedges_indices = np.ascontiguousarray(boundary_sedges, dtype=np.uint32)
        return Mesh(surface_exo_data, vertices, indices, sedges_indices, field_names_scalar_values)

    def get_vertices(self):
        return self.vertices

    def get_indices(self):
        return self.indices

    def get_sedges_indices(self):
        return self.sedges_indices

    def get_scalar_values_from_field_name(self, field_name):
        assert field_name in self.field_names_scalar_values
        return self.field_names_scalar_values[field_name]

    def get_bounding_box(self):
        bounding_box = bb.BoundingBox.create(self.vertices)
        return bounding_box

    def __repr__(self):
        print(f"Mesh({len(self.vertices)}, {len(self.indices)}, {self.field_names_scalar_values.keys()})")
