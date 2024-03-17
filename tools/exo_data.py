# Copyright (C) 2021 Akselos
import os
import time
import json
import collections
import tempfile
import numpy as np

import utils
import my_netcdf4 as m4
import block_attribute_tools as bt


# Code to read/write Exodus II mesh files to/from Python lists, dicts, and numpy arrays.
# See Exodus II reference at http://prod.sandia.gov/techlib/access-control.cgi/1992/922137.pdf
# Also contains some basic mesh modification tools, such as extrude.

class ElemType:
    def __init__(self, name, dimension, fside_tuples, second_order_fside_tuples=None,
                 extrude_type_name=None, tet_tuples=None, bottom_face_idx=None, top_face_idx=None,
                 is_shell=False, is_surface=False, tri_tuples=None):
        # The is_shell flag indicates the the element is two-sided.
        # The is_surface flag indicates the the element is part of the surface of a 3D object
        # (used for coarse meshes).
        self.name = name.upper()
        self.dimension = dimension
        self.fside_tuples = fside_tuples
        self.second_order_fside_tuples = second_order_fside_tuples
        self.extrude_type_name = extrude_type_name
        self.tet_tuples = tet_tuples
        self.tri_tuples = tri_tuples
        self.bottom_face_idx = bottom_face_idx
        self.top_face_idx = top_face_idx
        self.is_shell = is_shell
        self.is_surface = is_surface

    def get_faces(self, use_second_order=False):
        if not use_second_order:
            return [(idx, tup) for (idx, tup) in enumerate(self.fside_tuples) if len(tup) > 2]
        else:
            if self.second_order_fside_tuples is not None:
                return [(idx, tup) for (idx, tup) in enumerate(self.second_order_fside_tuples) if len(tup) > 2]
            else:
                return [(idx, tup) for (idx, tup) in enumerate(self.fside_tuples) if len(tup) > 2]

    def get_fedges(self):
        return [(idx, tup) for (idx, tup) in enumerate(self.fside_tuples) if len(tup) == 2]

    def get_fnodes(self):
        return [(idx, tup) for (idx, tup) in enumerate(self.fside_tuples) if len(tup) == 1]

    def get_n_nodes(self, high_order_node=False):
        mx = max(max(side_tuple) for side_tuple in self.fside_tuples)
        if high_order_node and self.second_order_fside_tuples is not None:
            mx = max(max(side_tuple) for side_tuple in self.second_order_fside_tuples)
        return mx + 1

    def get_fside_tuples(self):
        if self.second_order_fside_tuples is not None:
            return self.second_order_fside_tuples
        return self.fside_tuples

    def get_order(self):
        if self.second_order_fside_tuples is None or \
                np.array(self.second_order_fside_tuples).shape == np.array(self.fside_tuples).shape:
            return 1
        return 2

    def get_second_to_first_order_node_map(self):
        # This method returns a mapping from second order node to its neighbor first-order nodes.
        if self.second_order_fside_tuples is None:
            return None
        first_order_nodes = []
        for fside_tuple in self.fside_tuples:
            first_order_nodes.extend(list(fside_tuple))
        first_order_nodes = np.unique(first_order_nodes)
        len_1 = len(self.fside_tuples[0])
        second_order_node_infos = {}
        for _tuple in self.second_order_fside_tuples:
            for node_idx in range(len_1, len(_tuple)):
                second_node = _tuple[node_idx]
                assert second_node not in first_order_nodes
                if second_node in second_order_node_infos:
                    continue
                first_order_node_0 = _tuple[node_idx-len_1]
                first_order_node_1 = _tuple[(node_idx-len_1+1) % len_1]
                assert first_order_node_0 in first_order_nodes \
                    and first_order_node_1 in first_order_nodes, self.name
                second_order_node_infos[second_node] = \
                    (first_order_node_0, first_order_node_1)
        return second_order_node_infos

    def is_equal(self, other):
        # Two element types can be the same even if their names are different
        equal = True
        equal &= self.dimension == other.dimension
        equal &= self.fside_tuples == other.fside_tuples
        equal &= self.second_order_fside_tuples == other.second_order_fside_tuples
        equal &= self.extrude_type_name == other.extrude_type_name
        equal &= self.tet_tuples == other.tet_tuples
        equal &= self.bottom_face_idx == other.bottom_face_idx
        equal &= self.top_face_idx == other.top_face_idx
        equal &= self.is_shell == other.is_shell
        equal &= self.is_surface == other.is_surface
        return equal

# Note that Cubit/Paraview treat sides 0 and 1 of TRI elements as 2D, and sides 2,3,4 as 1D.
# This is different than the Exodus II documentation, which shows TRI elements only having 1D sides.
# When using coarse meshes, we use TRI elements to define the surface, and we copy
# Cubit/Paraview and treat side 0 as a 2D side.  We don't use sides 1 or higher for TRI.
tri3_fside_tuples = [(0,1,2)]
tri6_fside_tuples = [(0,1,2,3,4,5)]
quad4_fside_tuples = [(0,1,2,3)]
quad8_fside_tuples = [(0,1,2,3,4,5,6,7)]
quad9_fside_tuples = [(0,1,2,3,4,5,6,7,8)]
shell4_fside_tuples = [(0,1,2,3),(3,2,1,0),(0,1),(1,2),(2,3),(3,0)]
shell8_fside_tuples = [(0,1,2,3),(3,2,1,0),(0,1,4),(1,2,5),(2,3,6),(3,0,7)]
tetra4_fside_tuples = [(0, 1, 3), (1, 2, 3), (0, 3, 2), (0, 2, 1)]
tetra10_fside_tuples = [(0,1,3,4,8,7), (1,2,3,5,9,8), (0,3,2,7,9,6), (0,2,1,6,5,4)]
hex8_fside_tuples = [(0,1,5,4), (1,2,6,5), (2,3,7,6), (0,4,7,3), (0,3,2,1), (4,5,6,7)]
hex16_fside_tuples = [(0,1,5,4,8,12), (1,2,6,5,9,13), (2,3,7,6,10,14),
                      (0,4,7,3,11,15), (0,3,2,1,11,10,9,8), (4,5,6,7,12,13,14,15)]
hex20_fside_tuples = [(0,1,5,4,8,13,16,12), (1,2,6,5,9,14,17,13), (2,3,7,6,10,15,18,14),
                      (0,4,7,3,12,19,15,11), (0,3,2,1,11,10,9,8), (4,5,6,7,16,17,18,19)]
# HEX27 is not verified yet. Node 26 is the centroid.
# hex27_fside_tuples = [(0,1,5,4,8,13,16,12,22), (1,2,6,5,9,14,17,13,25), (2,3,7,6,10,15,18,14,23),
#                       (0,4,7,3,12,19,15,11,24), (0,3,2,1,11,10,9,8,20), (4,5,6,7,16,17,18,19,21)]
# corrected HEX27. Node 20 should be the centroid. Ref.: https://gsjaardema.github.io/seacas/html/element_types.html
hex27_fside_tuples = [(0,1,5,4, 8,13,16,12, 25), (1,2,6,5, 9,14,17,13, 24), (2,3,7,6, 10,15,18,14, 26),
                      (0,4,7,3, 12,19,15,11, 23), (0,3,2,1, 11,10,9,8, 21), (4,5,6,7, 16,17,18,19, 22)]

pyr5_fside_tuples = [(0,1,4), (1,2,4), (2,3,4), (3,0,4), (0,3,2,1)]
# PYR13 is not verified yet
pyr13_fside_tuples = [(0,1,4,5,10,9), (1,2,4,6,11,10),
                      (2,3,4,7,12,11), (3,0,4,8,9,12), (0,3,2,1,8,7,6,5)]
wedge6_fside_tuples = [(0,1,4,3), (1,2,5,4), (0,3,5,2), (0,2,1), (3,4,5)]
wedge15_fside_tuples = [(0,1,4,3,6,10,12,9), (1,2,5,4,7,11,13,10), (0,3,5,2,9,14,11,8),
                        (0,2,1,8,7,6), (3,4,5,12,13,14)]
ELEM_TYPES = [
    ElemType('SPHERE', 0, [(0,)]),
    # BAR order 1 and 2
    ElemType('BAR2', 1, [(0,1)]),
    ElemType('BAR3', 1, [(0,1)], second_order_fside_tuples=[(0,1,2)]),
    # TRIANGLE order 1 and 2
    ElemType('TRI3', 2, tri3_fside_tuples,
             extrude_type_name='WEDGE6', is_surface=True, tri_tuples=[(0,1,2)]),
    ElemType('TRI6', 2, tri3_fside_tuples, second_order_fside_tuples=tri6_fside_tuples,
             extrude_type_name='WEDGE15', is_surface=True, tri_tuples=[(0,1,2)]),
    ElemType('TRISHELL', 2, [(0,1,2),(2,1,0),(0,1),(1,2),(2,0)],
             extrude_type_name='WEDGE6', is_shell=True, tri_tuples=[(0,1,2)]),
    # QUAD order 1 and 2
    ElemType('QUAD4', 2, quad4_fside_tuples,
             extrude_type_name='HEX8', tri_tuples=[(0,1,2),(0,2,3)]),
    ElemType('QUAD8', 2, quad4_fside_tuples, second_order_fside_tuples=quad8_fside_tuples,
             extrude_type_name='HEX20', tri_tuples=[(0,1,2),(0,2,3)]),
    ElemType('QUAD9', 2, quad4_fside_tuples, second_order_fside_tuples=quad9_fside_tuples,
             extrude_type_name='HEX20', tri_tuples=[(0,1,2),(0,2,3)]),
    ElemType('QUADSHELL', 2, shell4_fside_tuples,
             extrude_type_name='HEX8', is_shell=True, tri_tuples=[(0,1,2),(0,2,3)]),
    ElemType('SHELL4', 2, shell4_fside_tuples,
             extrude_type_name='HEX8', is_shell=True, tri_tuples=[(0,1,2),(0,2,3)]),
    ElemType('SHELL8', 2, shell4_fside_tuples, second_order_fside_tuples=shell8_fside_tuples,
             extrude_type_name='HEX20', is_shell=True, tri_tuples=[(0,1,2), (0,2,3)]),
    # Note that tet tuples are of mixed handedness below.  We take absolute value when computing
    # volume.
    # PYRAMID order 1 and 2
    ElemType('PYRAMID5', 3, pyr5_fside_tuples,
             tet_tuples=[(0,1,3,4),(1,2,3,4)]),
    ElemType('PYRAMID13', 3, pyr5_fside_tuples, second_order_fside_tuples=pyr13_fside_tuples,
             tet_tuples=[(0,1,3,4),(1,2,3,4)]),
    # WEDGE order 1 and 2
    ElemType('WEDGE6', 3, wedge6_fside_tuples,
             tet_tuples=[(0,1,2,3),(2,3,4,5),(2,4,1,3)],
             top_face_idx=5, bottom_face_idx=4),
    ElemType('WEDGE15', 3, wedge6_fside_tuples, second_order_fside_tuples=wedge15_fside_tuples,
             tet_tuples=[(0,1,2,3), (2,3,4,5), (2,4,1,3)],
             top_face_idx=5, bottom_face_idx=4),
    # TETRA order 1 and 2
    ElemType('TETRA4', 3, tetra4_fside_tuples,
             tet_tuples=[(0, 1, 2, 3)]),
    ElemType('TETRA10', 3, tetra4_fside_tuples, second_order_fside_tuples=tetra10_fside_tuples,
             tet_tuples=[(0,1,2,3)]),
    # HEX order 1 and 2
    ElemType('HEX8', 3, hex8_fside_tuples,
             #tet_tuples=[(0,2,7,6), (0,2,6,3), (0,4,5,6), (0,4,6,7), (0,1,4,7), (0,1,7,2)],
             tet_tuples=[(0,2,3,7), (0,2,6,7), (0,4,6,7), (0,6,1,2), (0,6,1,4), (5,6,1,4)],
             top_face_idx=6, bottom_face_idx=5),
    ElemType('HEX9', 3, hex8_fside_tuples,
             tet_tuples=[(0,2,3,7), (0,2,6,7), (0,4,6,7), (0,6,1,2), (0,6,1,4), (5,6,1,4)],
             top_face_idx=6, bottom_face_idx=5),
    ElemType('HEX16', 3, hex8_fside_tuples, second_order_fside_tuples=hex16_fside_tuples,
             tet_tuples=[(0,2,3,7), (0,2,6,7), (0,4,6,7), (0,6,1,2), (0,6,1,4), (5,6,1,4)],
             top_face_idx=6, bottom_face_idx=5),
    ElemType('HEX20', 3, hex8_fside_tuples, second_order_fside_tuples=hex20_fside_tuples,
             tet_tuples=[(0,2,3,7), (0,2,6,7), (0,4,6,7), (0,6,1,2), (0,6,1,4), (5,6,1,4)],
             top_face_idx=6, bottom_face_idx=5),
    ElemType('HEX27', 3, hex8_fside_tuples, second_order_fside_tuples=hex27_fside_tuples,
             tet_tuples=[(0,2,3,7), (0,2,6,7), (0,4,6,7), (0,6,1,2), (0,6,1,4), (5,6,1,4)],
             top_face_idx=6, bottom_face_idx=5)]

ELEM_TYPE_MAP = dict((e.name, e) for e in ELEM_TYPES)
ELEM_TYPE_MAP['HEX'] = ELEM_TYPE_MAP['HEX8']
ELEM_TYPE_MAP["BAR"] = ELEM_TYPE_MAP["BAR2"]
ELEM_TYPE_MAP['EDGE'] = ELEM_TYPE_MAP['BAR2']
ELEM_TYPE_MAP['EDGE2'] = ELEM_TYPE_MAP['BAR2']
ELEM_TYPE_MAP['EDGE3'] = ELEM_TYPE_MAP['BAR3']
ELEM_TYPE_MAP['TRIANGLE'] = ELEM_TYPE_MAP['TRI3']
ELEM_TYPE_MAP['TRI'] = ELEM_TYPE_MAP['TRI3']
ELEM_TYPE_MAP['QUAD'] = ELEM_TYPE_MAP['QUAD4']
ELEM_TYPE_MAP['SHELL'] = ELEM_TYPE_MAP['SHELL4']
ELEM_TYPE_MAP['PYRAMID'] = ELEM_TYPE_MAP['PYRAMID5']
ELEM_TYPE_MAP['PYR'] = ELEM_TYPE_MAP['PYRAMID5']
ELEM_TYPE_MAP['WEDGE'] = ELEM_TYPE_MAP['WEDGE6']
ELEM_TYPE_MAP['TETRA'] = ELEM_TYPE_MAP['TETRA4']
ELEM_TYPE_MAP['TRISHELL3'] = ELEM_TYPE_MAP['TRISHELL']


REFLECTION_TRANSFORMATION_MAP = {
    'TRISHELL': [0, 2, 1],
    'QUAD4': [0, 3, 2, 1],
    'QUAD8': [0, 3, 2, 1, 7, 6, 5, 4],
    'QUAD9': [0, 3, 2, 1, 7, 6, 5, 4, 8],
    'SHELL4': [0, 3, 2, 1],
    'SHELL8': [0, 3, 2, 1, 7, 6, 5, 4],
    'TETRA4': [2, 1, 0, 3],
    'TETRA10': [2, 1, 0, 3, 5, 4, 6, 9, 8, 7],
    'PYRAMID5': [0, 3, 2, 1, 4],
    'PYRAMID13': [0,  3,  2,  1,  4,  8,  7,  6,  5,  9, 12, 11, 10],
    'WEDGE6': [2, 1, 0, 5, 4, 3],
    'WEDGE15': [2, 1, 0, 5, 4, 3, 7, 6, 8, 11, 10, 9, 13, 12, 14],
    'HEX8':  [0, 4, 7, 3, 1, 5, 6, 2],
    'HEX9':  [0, 4, 7, 3, 1, 5, 6, 2, 8],
    'HEX16': [3, 2, 1, 0, 7, 6, 5, 4, 10, 9, 8, 11, 14, 13, 12, 15],
    'HEX20': [0, 4, 7, 3, 1, 5, 6, 2, 12, 19, 15, 11, 8, 16, 18, 10, 13, 17, 14, 9],
    'HEX27': [0, 4, 7, 3, 1, 5, 6, 2, 12, 19, 15, 11, 8, 16, 18, 10, 13, 17, 14, 9, 20, 23, 24, 21, 22, 25, 26]
}
# Note: currently reordering nodes in 'WEDGE' 'HEX16' does not work perfectly for solution reflection


class ExoBlockData:
    def __init__(self, block_idx, elem_type_name, elems, elem_start_idx, elem_field_values=None):
        self.block_idx = block_idx
        self.elem_type_name = elem_type_name
        self.elems = elems
        self.elem_start_idx = elem_start_idx
        self.elem_type = self.get_elem_type(elem_type_name)
        if elem_field_values is None:
            elem_field_values = {}
        self.elem_field_values = elem_field_values

    @staticmethod
    def get_elem_type(elem_type_name):
        '''
        # Remove numerical suffix on the elem type name and change to uppercase.
        elem_type_prefix = elem_type_name
        for i in (str(x) for x in range(10)):
            elem_type_prefix = elem_type_prefix.replace(i, '')
        return ELEM_TYPE_MAP[elem_type_prefix.upper()]
        '''
        return ELEM_TYPE_MAP[elem_type_name]

    def is_equal(self, other):
        # For testing purposes.
        equal = True
        equal &= self.block_idx == other.block_idx
        if not equal:
            print("block idx not equal", self.block_idx, other.block_idx)
        equal &= self.elem_type.is_equal(other.elem_type)
        if not equal:
            print("elem type idx not equal")
        equal &= np.all(self.elems == other.elems)
        if not equal:
            print("elems not equal", self.elems[0], other.elems[0])
            assert False
        equal &= self.elem_start_idx == other.elem_start_idx
        if not equal:
            print("elem_start_idx idx not equal", self.elem_start_idx, other.elem_start_idx)

        return equal

    def write_dict(self):
        j = collections.OrderedDict()
        j["elem_field_values"] = {str(k): v.tolist() for k, v in self.elem_field_values.items()}
        j["elems"] = self.elems.tolist()
        j["elem_type_name"] = self.elem_type_name
        return j

    def write_json_str(self):
        j = self.write_dict()
        s = json.dumps(j, indent=1)
        return s

    @staticmethod
    def read_dict(j, block_idx, elem_start_idx):
        elems = np.array(j["elems"], dtype=np.int32)
        elem_field_values = {str(k): np.array(v, dtype=np.float32)
                             for k, v in j["elem_field_values"].items()}
        return ExoBlockData(
            block_idx, j["elem_type_name"], elems, elem_start_idx, elem_field_values)


class ExoData:
    def __init__(self, coords, exo_block_datas, sideset_elem_idxs, sideset_fside_idxs, nodesets,
                 block_idx_to_subdomain_id, node_field_values=None, extra_data_map=None,  time_values=None,
                 block_id_to_names=None, sideset_names=None, sideset_field_values=None, nodeset_field_values=None,
                 exo_edge_block_datas=None, edge_block_idx_to_subdomain_id=None,
                 edge_block_id_to_names=None):
        self.coords = coords
        if node_field_values is None:
            node_field_values = {}
        self.node_field_values = node_field_values

        self.exo_block_datas = exo_block_datas
        self.block_idx_to_subdomain_id = block_idx_to_subdomain_id
        self.block_id_to_names = block_id_to_names
        assert len(exo_block_datas) == len(block_idx_to_subdomain_id)
        for block_idx in range(len(exo_block_datas)):
            assert exo_block_datas[block_idx].block_idx == block_idx

        # Edge blocks
        self.exo_edge_block_datas = exo_edge_block_datas
        self.edge_block_idx_to_subdomain_id = edge_block_idx_to_subdomain_id
        self.edge_block_id_to_names = edge_block_id_to_names

        self.sideset_elem_idxs = sideset_elem_idxs
        self.sideset_fside_idxs = sideset_fside_idxs
        if sideset_names is None:
            sideset_names = {}
        self.sideset_names = sideset_names
        if sideset_field_values is None:
            sideset_field_values = {}
        self.sideset_field_values = sideset_field_values
        self.nodesets = nodesets
        if nodeset_field_values is None:
            nodeset_field_values = {}
        self.nodeset_field_values = nodeset_field_values

        if time_values is None:
            time_values = np.zeros(1)
        self.time_values = time_values

        # Require that we have exactly one set of field values for each time value (nodes).
        for field_name, field_values in self.node_field_values.items():
            assert len(field_values.shape) == 2 and field_values.shape[0] == len(self.time_values)
            assert field_values.shape[1] == len(coords)

        # Require that we have exactly one set of field values for each time value (elems).
        for exo_block_data in self.exo_block_datas:
            for field_name, field_values in exo_block_data.elem_field_values.items():
                assert len(field_values.shape) == 2 and field_values.shape[0] == len(
                    self.time_values), f"{field_name}: {field_values.shape} vs." \
                                       f" {len(self.time_values)}"

        # This is a generic map to store extra data that may be used for postprocessing the exo
        # It is "semi write only" because it is written into file when the exo is saved, but when
        # the exo is read it is not retrieved from the internal netcdf container, so it has to be
        # accessed by reading directly attributes and arrays from that container (this way this
        # extra data can be generic)
        self.extra_data_map = extra_data_map

    def write_dict(self):
        j = collections.OrderedDict()
        j_exo_block_datas = []
        for exo_block_data in self.exo_block_datas:
            j_exo_block_datas.append(exo_block_data.write_dict())
        j["sideset_elem_idxs"] = {int(k): v.tolist() for k, v in self.sideset_elem_idxs.items()}
        j["sideset_fside_idxs"] = {int(k): v.tolist() for k, v in self.sideset_fside_idxs.items()}
        j["nodesets"] = {int(k): v.tolist() for k, v in self.nodesets.items()}
        j["block_idx_to_subdomain_id"] = self.block_idx_to_subdomain_id.tolist()
        j["node_field_values"] = {str(k): v.tolist() for k, v in self.node_field_values.items()}
        j["exo_block_datas"] = j_exo_block_datas
        j["coords"] = self.coords.tolist()
        return j

    def write_json_str(self):
        j = self.write_dict()
        s = json.dumps(j, indent=1)
        # Some ridiculous junk to get pretty printing for our specific data:
        s = s.replace("[\n   ", "[")
        s = s.replace(", \n   ", ", ")
        s = s.replace("\n    ]",  "]\n")
        s = s.replace("\n  ]", "]")
        s = s.replace("]\n,  [", "],  \n    [")
        s = s.replace(' [ [  ', ' [\n    [')
        return s

    def write_json_file(self, filepath):
        s = self.write_json_str()
        with utils.atomic_write(filepath) as f:
            f.write(s)

    @staticmethod
    def read_dict(j):
        coords = np.array(j["coords"], dtype=np.float64)
        sideset_elem_idxs = {int(k): np.array(v, dtype=np.int32)
                             for k, v in j["sideset_elem_idxs"].items()}
        sideset_fside_idxs = {int(k): np.array(v, dtype=np.int32)
                              for k, v in j["sideset_fside_idxs"].items()}
        nodesets = {int(k): np.array(v, dtype=np.int32)
                    for k, v in j["nodesets"].items()}
        block_idx_to_subdomain_id = np.array(j["block_idx_to_subdomain_id"])
        node_field_values = {str(k): np.array(v, dtype=np.float32)
                             for k, v in j["node_field_values"].items()}
        exo_block_datas = []
        elem_start_idx = 0
        for block_idx, exo_block_data in enumerate(j["exo_block_datas"]):
            exo_block_data = ExoBlockData.read_dict(exo_block_data, block_idx, elem_start_idx)
            elem_start_idx += len(exo_block_data.elems)
            exo_block_datas.append(exo_block_data)

        return ExoData(
            coords, exo_block_datas, sideset_elem_idxs, sideset_fside_idxs, nodesets,
            block_idx_to_subdomain_id, node_field_values)

    @staticmethod
    def read_json_str(s):
        return ExoData.read_dict(json.loads(s))

    @staticmethod
    def read_json_file(filepath):
        return ExoData.read_json_str(open(filepath).read())

    def set_sideset(self, sideset_id, mesh_elem_idxs, face_idxs):
        self.sideset_elem_idxs[sideset_id] = mesh_elem_idxs
        self.sideset_fside_idxs[sideset_id] = face_idxs

    def remove_sideset(self, sideset_id):
        if sideset_id in self.sideset_elem_idxs:
            del self.sideset_elem_idxs[sideset_id]
            del self.sideset_fside_idxs[sideset_id]
            return True
        else:
            return False

    def set_nodeset(self, nodeset_id, node_idxs):
        self.nodesets[nodeset_id] = node_idxs

    def remove_nodeset(self, nodeset_id):
        if nodeset_id in self.nodesets:
            del self.nodesets[nodeset_id]
            return True
        else:
            return False

    @staticmethod
    def read(filename, force_3d_coords=False):
        assert os.path.exists(filename), f'Mesh not found: {filename}'
        # Read an .exo file and create an ExoData instance
        my_netcdf = m4.create_netcdf4_or_netcdf3(filename)
        exo_data = ExoData._read(my_netcdf, force_3d_coords=force_3d_coords)
        my_netcdf.close()
        return exo_data

    def copy(self):
        with tempfile.NamedTemporaryFile() as f:
            self.write(f.name)
            return ExoData.read(f.name)

    @staticmethod
    def read_from_data(data):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
        f.close()
        exo_data = ExoData.read(f.name)
        try:
            os.remove(f.name)
        except IOError or FileExistsError or FileNotFoundError or BlockingIOError:
            pass
        return exo_data

    @staticmethod
    def _read(f, force_3d_coords=False):
        # Read coordinates (floating point values).
        if f.has_array('coord'):
            coords = f.get_array('coord').transpose()
        elif not f.has_array('coordy'):
            x_coords = f.get_array('coordx')
            coords = np.reshape(x_coords, (len(x_coords), 1))
        elif not f.has_array('coordz'):
            x_coords = f.get_array('coordx')
            y_coords = f.get_array('coordy')
            coords = np.vstack((x_coords, y_coords)).T
        else:
            x_coords = f.get_array('coordx')
            y_coords = f.get_array('coordy')
            z_coords = f.get_array('coordz')
            coords = np.vstack((x_coords, y_coords, z_coords)).T

        if force_3d_coords and coords.shape[-1] != 3:
            coords_3d = np.zeros((len(coords), 3), dtype=coords.dtype)
            for c in range(coords.shape[-1]):
                coords_3d[:, c] = coords[:, c]
            coords = coords_3d

        # coor_names aren't implemented yet.  Need to save these in ExoData and write them when
        # writing ExoData.  When converting 2D mesh to 3D maybe can use these names to know
        # how to map 2D coords to 3D.
        # coor_names = f.get_strings("coor_names")

        # Ugh, previously we first checked 'time_step' value from f.get_dimension('time_step'),
        # but it looks like there is a bug with h5py when reading a dataset that contains a
        # single scalar(?), because it returns an empty array.
        # So here, alternately, I get directly the 'time_whole' variable.
        #time_step = f.get_dimension('time_step')
        time_values = None
        if f.has_array('time_whole'):
            _time_values = f.get_array('time_whole')
            # f.get_array('time_whole') can be an empty array. In that case, just leave
            # time_values as None.
            if len(_time_values) > 0:
                time_values = _time_values

        # Read element blocks.
        n_blocks = f.get_dimension('num_el_blk')
        subdomain_ids = f.get_array('eb_prop1', dtype=np.int)[:n_blocks]
        block_id_to_names = None
        if f.has_array("eb_names"):
            block_names = f.get_strings("eb_names")
            block_id_to_names = {}
            for block_idx in range(n_blocks):
                block_id_to_names[subdomain_ids[block_idx]] = block_names[block_idx]
        exo_block_datas = [None]*n_blocks
        elem_start_idx = 0

        if 'name_elem_var' in f.variables:
            elem_field_names = f.get_strings('name_elem_var')
        else:
            elem_field_names = []

        if 'elem_var_tab' in f.variables:
            elem_var_tab = f.get_array('elem_var_tab')
        else:
            elem_var_tab = np.ones((n_blocks, len(elem_field_names)), dtype=np.int16)

        for block_idx in range(n_blocks):
            connect_name = 'connect' + str(block_idx + 1)
            elem_type = f.get_attribute(connect_name, 'elem_type')
            elems = f.get_idx_array(connect_name)
            # element field values for each ExoBlockData
            elem_field_values = {}
            for elem_name_idx, elem_name in enumerate(elem_field_names):
                if not elem_var_tab[block_idx, elem_name_idx]:
                    continue
                array_name = 'vals_elem_var' + str(elem_name_idx + 1) + 'eb' + str(block_idx + 1)
                if not f.has_array(array_name):
                    continue
                elem_field_values[elem_name] = f.get_array(array_name)

            exo_block_data = ExoBlockData(
                block_idx, elem_type, elems, elem_start_idx, elem_field_values)
            exo_block_datas[block_idx] = exo_block_data
            elem_start_idx += len(elems)

        # Read edge blocks.
        exo_edge_block_datas = None
        edge_subdomain_ids = None
        edge_block_id_to_names = None
        if f.has_dimension('num_ed_blk'):
            n_edge_blocks = f.get_dimension('num_ed_blk')
            edge_subdomain_ids = f.get_array('ed_prop1', dtype=np.int)[:n_edge_blocks]
            if f.has_array("ed_names"):
                edge_block_names = f.get_strings("ed_names")
                edge_block_id_to_names = {}
                for edge_block_idx in range(n_edge_blocks):
                    edge_block_id_to_names[edge_subdomain_ids[edge_block_idx]] = \
                        edge_block_names[edge_block_idx]
            exo_edge_block_datas = [None]*n_edge_blocks
            edge_start_idx = 0

            for edge_block_idx in range(n_edge_blocks):
                connect_name = 'ebconn' + str(edge_block_idx + 1)
                elem_type = f.get_attribute(connect_name, 'elem_type')
                edge_elems = f.get_idx_array(connect_name)

                # We don't support elem_field_names for edge block at the moment
                exo_edge_block_data = ExoBlockData(
                    edge_block_idx, elem_type, edge_elems, edge_start_idx)
                exo_edge_block_datas[edge_block_idx] = exo_edge_block_data
                edge_start_idx += len(edge_elems)

        # Read sidsets.
        if f.has_array('ss_prop1'):
            sideset_ids = f.get_array('ss_prop1')
        else:
            sideset_ids = np.array([], dtype=np.int32)

        sideset_elem_idxs = {}
        sideset_fside_idxs = {}
        for sideset_idx, sideset_id in enumerate(sideset_ids):
            suffix = str(sideset_idx + 1)
            # Sometimes a numpy int escapes into Python world and we get a crash trying to
            # convert it to json.  Make sure dict id keys are Python ints.
            sideset_elem_idxs[int(sideset_id)] = f.get_idx_array('elem_ss' + suffix)
            sideset_fside_idxs[int(sideset_id)] = f.get_idx_array('side_ss' + suffix)

        # Read node set data.
        if f.has_dimension('num_node_sets') and f.has_array('ns_prop1'):
            n_nodesets = f.get_dimension('num_node_sets')
            nodeset_ids = f.get_array('ns_prop1')[:n_nodesets]
            n_nodesets = len(nodeset_ids)
        else:
            n_nodesets = 0
            nodeset_ids = np.array([], dtype=np.int32)

        nodesets = {}
        for nodeset_idx in range(n_nodesets):
            nodeset_id = nodeset_ids[nodeset_idx]
            suffix = str(nodeset_idx + 1)
            coord_idxs = f.get_idx_array('node_ns' + suffix)
            nodesets[int(nodeset_id)] = coord_idxs

        node_field_values = {}
        if 'name_nod_var' in f.variables:
            node_field_names = f.get_strings('name_nod_var')
            for i in range(len(node_field_names)):
                node_field_values[node_field_names[i]] = f.get_array('vals_nod_var' + str(i + 1))

        if 'name_sset_var' in f.variables:
            sset_field_names = f.get_strings('name_sset_var')
            sset_var_tab = f.get_array('sset_var_tab')
            sideset_field_values = {}
            for sset_idx, sset_id in enumerate(sideset_ids):
                for sset_field_name_idx, sset_field_name in enumerate(sset_field_names):
                    if not sset_var_tab[sset_idx, sset_field_name_idx]:
                        continue
                    array_name = 'vals_sset_var' + str(sset_field_name_idx + 1) \
                                 + 'ss' + str(sset_idx + 1)
                    if not f.has_array(array_name):
                        print("Could not find ncdf variable {}".format(array_name))
                        continue
                    sideset_field_values.setdefault(sset_id, {})
                    sideset_field_values[sset_id][sset_field_name] = f.get_array(array_name)
        else:
            sideset_field_values = None

        if 'name_nset_var' in f.variables:
            nset_field_names = f.get_strings('name_nset_var')
            nset_var_tab = f.get_array('nset_var_tab')
            nodeset_field_values = {}
            for nset_idx, nset_id in enumerate(nodeset_ids):
                for nset_field_name_idx, nset_field_name in enumerate(nset_field_names):
                    if not nset_var_tab[nset_idx, nset_field_name_idx]:
                        continue
                    array_name = 'vals_nset_var' + str(nset_field_name_idx + 1) \
                                 + 'ns' + str(nset_idx + 1)
                    if not f.has_array(array_name):
                        print("Could not find ncdf variable {}".format(array_name))
                        continue
                    nodeset_field_values.setdefault(nset_id, {})
                    nodeset_field_values[nset_id][nset_field_name] = f.get_array(array_name)
        else:
            nodeset_field_values = None

        return ExoData(
            coords, exo_block_datas, sideset_elem_idxs, sideset_fside_idxs, nodesets,
            subdomain_ids, node_field_values, time_values=time_values,
            block_id_to_names=block_id_to_names, sideset_field_values=sideset_field_values,
            nodeset_field_values=nodeset_field_values,
            exo_edge_block_datas=exo_edge_block_datas,
            edge_block_idx_to_subdomain_id=edge_subdomain_ids,  edge_block_id_to_names=edge_block_id_to_names)

    def write(self, filepath, use_netcdf_python=False, *args, **kwargs):
        '''
        # For some reason, in order to make ParaView read the output mesh, the variable list
        # must be in a certain order.
        # Don't need to reorder the dimension list, but doing so would make it easier to
        # debug (to compare with reference mesh).
        variable_order = [
             ['time_whole', 'qa_records', 'coor_names', 'eb_names'],
             ['ns_status', 'ns_prop', 'ns_names'],
             ['node_ns', 'dist_fact_ns'],
             ['ss_status', 'ss_prop', 'ss_names'],
             ['elem_ss', 'side_ss', 'dist_fact_ss'],
             ['elem_map', 'eb_status', 'eb_prop'],
             ['connect', 'coordx', 'coordy', 'coordz'],
             ['tag', 'part', 'prop_ss']]

        dimension_order = [
             ['len_string', 'len_line', 'four', 'time_step', 'len_name'],
             ['num_dim', 'num_nodes', 'num_elem'],
             ['num_el_blk', 'num_qa_rec'],
             ['num_node_sets', 'num_nod_ns'],
             ['num_side_sets'],
             ['num_side_ss', 'num_df_ss'],
             ['num_el_in_blk', 'num_nod_per_el'],
             ['num_nod_var'],
             ['num_tag', 'num_part']]
        '''

        with utils.atomic_write(filepath, yield_filepath=True, mode='wb') as f0:
            my_netcdf = m4.create_netcdf4_or_netcdf3(f0, use_netcdf_python, mode='w', *args, **kwargs)
            self._write(my_netcdf, os.path.split(filepath)[-1])
            my_netcdf.close()

    def _write_extra_data(self, f):
        if self.extra_data_map is None:
            return

        float_type = self.coords.dtype.type
        for key, value in self.extra_data_map.items():
            if isinstance(value, (np.ndarray, np.generic)):
                dim_names = []
                for index, dim in enumerate(value.shape):
                    dim_names.append(key + "_size_" + str(index))
                f.set_array(value, key, tuple(dim_names))
            else:
                f.set_attribute(None, key, float_type(value))

    def _write(self, f, title):
        len_string = 66
        n_dim = self.coords.shape[-1]
        float_type = self.coords.dtype.type
        f.set_attribute(None, "api_version", float_type(5.22))
        f.set_attribute(None, "version", float_type(5.22))
        f.set_attribute(None, "floating_point_word_size", np.int32(self.coords.dtype.itemsize))
        f.set_attribute(None, "file_size", 1)
        # Uhg, looks like h5py cannot work width fixed-width string. The trick here is to convert
        #  it to np.string_.
        f.set_attribute(None, "title", np.string_(title))
        f.set_dimension("len_line", 81)

        # time_whole
        f.set_array(self.time_values, "time_whole", "time_step")

        # qa_records
        qa_record_strs = ["Akselos", "1.0", time.strftime("%Y-%m-%d %H:%M")]
        qa_records = np.zeros((1, 4, len_string), dtype='|S1')
        for i, qa_record_str in enumerate(qa_record_strs):
            qa_records[0, i, :len(qa_record_str)] = list(qa_record_str)
        f.set_array(qa_records, "qa_records", ("num_qa_rec", "four", "len_string"))

        # coor_names
        coor_names = np.zeros((n_dim, len_string), dtype='|S1')
        for i, coor_name in enumerate(['x', 'y', 'z'][:n_dim]):
            coor_names[i, :len(coor_name)] = list(coor_name)
        f.set_array(coor_names, "coor_names", ("num_dim", "len_string"))

        # eb_names
        if self.block_id_to_names is not None:
            eb_names = np.chararray((len(list(self.block_id_to_names.values())), len_string))
            for i, name in enumerate(self.block_id_to_names.values()):
                for j in range(len_string):
                    if j < len(name):
                        eb_names[i, j] = name[j]
                    else:
                        eb_names[i, j] = ''
        else:
            eb_names = np.zeros((len(self.exo_block_datas), len_string), dtype='|S1')
        f.set_array(eb_names, "eb_names", ("num_el_blk", "len_string"))

        # Node sets
        nodeset_ids = np.array(sorted(self.nodesets.keys()))
        n_nodesets = len(nodeset_ids)
        if n_nodesets > 0:
            # ns_status
            ns_status = np.ones(n_nodesets, dtype=np.int32)
            f.set_array(ns_status, "ns_status", ("num_node_sets",))

            # ns_prop
            ns_prop1 = f.set_array(nodeset_ids, "ns_prop1", ("num_node_sets",))
            f.set_attribute(ns_prop1, 'name', np.string_('ID'))

            # ns_names
            # Note that for some reason, when opening in Cubit, if a nodeset name is == ''
            # then it will it will take the name of the next nodeset.
            ns_names = np.zeros((n_nodesets, len_string), dtype='|S1')
            ns_names[:] = ' '
            f.set_array(ns_names, "ns_names", ("num_node_sets", "len_string"))

            for nodeset_idx, nodeset_id in enumerate(nodeset_ids):
                suffix = str(nodeset_idx + 1)
                # node_ns
                f.set_idx_array(
                    self.nodesets[nodeset_id], "node_ns"+suffix, "num_nod_ns"+suffix)

                # dist_fact_ns
                # Not currently used:
                #f.set_array(
                #   np.zeros(0, dtype=float_type), "dist_fact_ns"+suffix, "num_df_ns"+suffix)

        # Side sets
        # NetCDF4 doesn't seem to allow zero-length arrays.  "NC_UNLIMITED size already in use" ?
        # Skip writing these sidesets.  Be careful because this is a special case in which an
        # ExoData instance does not map exactly to its Exodus II file.
        sideset_ids = np.array(sorted(
            [i for i in self.sideset_elem_idxs if len(self.sideset_elem_idxs[i]) > 0]))
        n_sidesets = len(sideset_ids)
        if n_sidesets > 0:
            # ss_status
            ss_status = np.ones(n_sidesets, dtype=np.int32)
            f.set_array(ss_status, "ss_status", ("num_side_sets",))

            # ss_prop1
            ss_prop1 = f.set_array(sideset_ids, "ss_prop1", ("num_side_sets",))
            f.set_attribute(ss_prop1, 'name', np.string_('ID'))

            # ss_names
            # Note that for some reason, when opening in Cubit, if a sideset name is == ''
            # then it will it will take the name of the next sideset.
            ss_names = np.zeros((n_sidesets, len_string), dtype='|S1')
            ss_names[:] = ' '
            f.set_array(ss_names, "ss_names", ("num_side_sets", "len_string"))

            for sideset_idx, sideset_id in enumerate(sideset_ids):
                elem_idxs = self.sideset_elem_idxs[sideset_id]
                fside_idxs = self.sideset_fside_idxs[sideset_id]
                assert len(elem_idxs) == len(fside_idxs)

                suffix = str(sideset_idx + 1)
                # elem_ss
                f.set_idx_array(elem_idxs, 'elem_ss'+suffix, "num_side_ss"+suffix)

                # side_ss
                f.set_idx_array(fside_idxs, 'side_ss'+suffix, "num_side_ss"+suffix)

                # dist_fact_ss
                # Not currently used:
                #f.set_array(
                #    np.zeros(0, dtype=float_type), "dist_fact_ss"+suffix, "num_df_ss"+suffix)

        # Element blocks
        # elem_map
        n_elems = sum(len(e.elems) for e in self.exo_block_datas)
        f.set_idx_array(np.arange(n_elems, dtype=np.int32), "elem_map", ("num_elem",))

        # eb_status
        n_blocks = len(self.exo_block_datas)
        f.set_array(np.ones(n_blocks, dtype=np.int32), "eb_status", ("num_el_blk",))

        # eb_prop1
        eb_prop1 = f.set_array(self.block_idx_to_subdomain_id, "eb_prop1", ("num_el_blk",))
        f.set_attribute(eb_prop1, 'name', np.string_('ID'))

        # Blocks
        name_elem_var = np.array([]) # this variable should be used for all blocks
        for exo_block_data in self.exo_block_datas:
            suffix = str(exo_block_data.block_idx + 1)
            connect_name = 'connect' + suffix
            nr_name = 'num_el_in_blk' + suffix
            nc_name = 'num_nod_per_el' + suffix
            # connect
            connect = f.set_idx_array(exo_block_data.elems, connect_name, (nr_name, nc_name))

            # Elem field values
            elem_field_names = list(exo_block_data.elem_field_values.keys())
            n_elem_fields = len(elem_field_names)
            if n_elem_fields > 0:
                # write "name_elem_var"
                _name_elem_var = np.zeros((n_elem_fields, len_string), dtype='|S1')
                for i, field_name in enumerate(elem_field_names):
                    _name_elem_var[i, :len(field_name)] = list(field_name)
                # add _name_elem_var of this block into name_elem_var
                if len(name_elem_var) == 0:
                    name_elem_var = _name_elem_var
                else:
                    name_elem_var = np.concatenate((name_elem_var, _name_elem_var))

            # elem_type attribute
            f.set_attribute(connect, 'elem_type', np.string_(exo_block_data.elem_type_name))

        # Find unique field names in name_elem_var and write those field values to ncdf
        if len(name_elem_var) > 0:
            b = np.ascontiguousarray(name_elem_var).view(
                np.dtype((np.void, name_elem_var.dtype.itemsize * name_elem_var.shape[1])))
            _, unique_idxs = np.unique(b, return_index=True)
            name_elem_var = name_elem_var[np.sort(unique_idxs)]
            f.set_array(name_elem_var, "name_elem_var", ("num_elem_var", "len_string"))

            elem_var_tab = np.zeros((n_blocks, len(name_elem_var)), dtype=np.int16)
            for exo_block_data in self.exo_block_datas:
                suffix = str(exo_block_data.block_idx + 1)
                nr_name = 'num_el_in_blk' + suffix
                elem_field_names = list(exo_block_data.elem_field_values.keys())
                n_elem_fields = len(elem_field_names)
                if n_elem_fields > 0:
                    for name_idx, elem_name in enumerate(name_elem_var):
                        elem_name = elem_name.tostring().split(b'\x00')[0].decode('utf-8')
                        if elem_name in elem_field_names:
                            elem_var_tab[exo_block_data.block_idx, name_idx] = 1
                            nectcdf_variable_name = \
                                'vals_elem_var' + str(name_idx + 1) + 'eb' + suffix
                            f.set_array(
                                exo_block_data.elem_field_values[elem_name],
                                nectcdf_variable_name, ("time_step", nr_name))

            f.set_array(elem_var_tab, "elem_var_tab", ("num_el_blk", "num_elem_var"))

        # Edge blocks
        if self.exo_edge_block_datas is not None:
            n_edges = sum(len(e.elems) for e in self.exo_edge_block_datas)
            f.set_dimension("num_edge", n_edges)
            # ed_status
            n_edge_blocks = len(self.exo_edge_block_datas)
            f.set_array(np.ones(n_edge_blocks, dtype=np.int32), "ed_status", ("num_ed_blk",))
            # eb_prop1
            ed_prop1 = f.set_array(self.edge_block_idx_to_subdomain_id, "ed_prop1", ("num_ed_blk",))
            f.set_attribute(ed_prop1, 'name', np.string_('ID'))
            for edge_block_data in self.exo_edge_block_datas:
                suffix = str(edge_block_data.block_idx + 1)
                connect_name = 'ebconn' + suffix
                nr_name = 'num_ed_in_blk' + suffix
                nc_name = 'num_nod_per_ed' + suffix
                # connect
                ebconn = f.set_idx_array(edge_block_data.elems, connect_name, (nr_name, nc_name))
                f.set_attribute(ebconn, 'elem_type', np.string_(edge_block_data.elem_type_name))

        # coord
        if n_dim == 1:
            f.set_array(self.coords[:, 0], "coordx", "num_nodes")
        elif n_dim == 2:
            f.set_array(self.coords[:, 0], "coordx", "num_nodes")
            f.set_array(self.coords[:, 1], "coordy", "num_nodes")
        elif n_dim == 3:
            f.set_array(self.coords[:, 0], "coordx", "num_nodes")
            f.set_array(self.coords[:, 1], "coordy", "num_nodes")
            f.set_array(self.coords[:, 2], "coordz", "num_nodes")
        else:
            assert False

        # Node field values
        node_field_names = list(self.node_field_values.keys())
        n_node_fields = len(node_field_names)
        if n_node_fields > 0:
            # write "name_nod_var"
            name_nod_var = np.zeros((n_node_fields, len_string), dtype='|S1')
            for i, field_name in enumerate(node_field_names[:n_node_fields]):
                name_nod_var[i, :len(field_name)] = list(field_name)
            f.set_array(name_nod_var, "name_nod_var", ("num_nod_var", "len_string"))
            # write 'vals_nod_vari'
            for i in range(n_node_fields):
                uname = node_field_names[i]
                f.set_array(
                    np.array(self.node_field_values[uname]), 'vals_nod_var' + str(i + 1),
                    ("time_step", "num_nodes"))

        # Sideset field values
        name_sset_var = np.array([]) # this variable should be used for all sidesets
        for sideset_id in sideset_ids:
            if sideset_id in self.sideset_field_values:
                _sset_field_names = list(self.sideset_field_values[sideset_id].keys())
                if len(_sset_field_names) > 0:
                    _name_sset_var = np.zeros((len(_sset_field_names), len_string), dtype='|S1')
                    for i, field_name in enumerate(_sset_field_names):
                        _name_sset_var[i, :len(field_name)] = list(field_name)
                    if len(name_sset_var) == 0:
                        name_sset_var = _name_sset_var
                    else:
                        name_sset_var = np.concatenate((name_sset_var, _name_sset_var))

        # Find unique field names in name_sset_var and write those fields to ncdf
        if len(name_sset_var) > 0:
            b = np.ascontiguousarray(name_sset_var).view(
                np.dtype((np.void, name_sset_var.dtype.itemsize * name_sset_var.shape[1])))
            _, unique_idxs = np.unique(b, return_index=True)
            name_sset_var = name_sset_var[np.sort(unique_idxs)]
            f.set_array(name_sset_var, "name_sset_var", ("num_sset_var", "len_string"))

            sset_var_tab = np.zeros((n_sidesets, len(name_sset_var)), dtype=np.int16)
            for sset_idx, sset_id in enumerate(sideset_ids):
                if sset_id not in self.sideset_field_values:
                    continue
                suffix = str(sset_idx + 1)
                n_side_dimension_name = 'num_side_ss' + suffix
                for field_name_idx, field_name in enumerate(name_sset_var):
                    field_name = field_name.tostring().split(b'\x00')[0].decode('utf-8')
                    if field_name in self.sideset_field_values[sset_id]:
                        sset_var_tab[sset_idx, field_name_idx] = 1
                        nectcdf_variable_name = \
                            'vals_sset_var' + str(field_name_idx + 1) + 'ss' + suffix
                        f.set_array(
                            self.sideset_field_values[sset_id][field_name],
                            nectcdf_variable_name, ("time_step", n_side_dimension_name))

            f.set_array(sset_var_tab, "sset_var_tab", ("num_side_sets", "num_sset_var"))

        # Nodeset field values
        if hasattr(self, "nodeset_field_values"):
            # Hmm, exo_data loaded from old pickled mesh might not have "nodeset_field_values" yet.
            # It'd be cleaner to increase the RefMesh version but I don't want to do that because
            # that will affect current users when they load their existing models
            name_nset_var = np.array([])  # this variable should be used for all nodesets
            for nodeset_id in nodeset_ids:
                if nodeset_id in self.nodeset_field_values:
                    _nset_field_names = list(self.nodeset_field_values[nodeset_id].keys())
                    if len(_nset_field_names) > 0:
                        _name_nset_var = np.zeros((len(_nset_field_names), len_string), dtype='|S1')
                        for i, field_name in enumerate(_nset_field_names):
                            _name_nset_var[i, :len(field_name)] = list(field_name)
                        if len(name_nset_var) == 0:
                            name_nset_var = _name_nset_var
                        else:
                            name_nset_var = np.concatenate((name_nset_var, _name_nset_var))

            # Find unique field names in name_nset_var and write those fields to ncdf
            if len(name_nset_var) > 0:
                b = np.ascontiguousarray(name_nset_var).view(
                    np.dtype((np.void, name_nset_var.dtype.itemsize * name_nset_var.shape[1])))
                _, unique_idxs = np.unique(b, return_index=True)
                name_nset_var = name_nset_var[np.sort(unique_idxs)]
                f.set_array(name_nset_var, "name_nset_var", ("num_nset_var", "len_string"))

                nset_var_tab = np.zeros((n_nodesets, len(name_nset_var)), dtype=np.int16)
                for nset_idx, nset_id in enumerate(nodeset_ids):
                    if nset_id not in self.nodeset_field_values:
                        continue
                    suffix = str(nset_idx + 1)
                    n_node_dimension_name = 'num_nod_ns' + suffix
                    for field_name_idx, field_name in enumerate(name_nset_var):
                        field_name = field_name.tostring().split(b'\x00')[0].decode('utf-8')
                        if field_name in self.nodeset_field_values[nset_id]:
                            nset_var_tab[nset_idx, field_name_idx] = 1
                            nectcdf_variable_name = \
                                'vals_nset_var' + str(field_name_idx + 1) + 'ns' + suffix
                            f.set_array(
                                self.nodeset_field_values[nset_id][field_name],
                                nectcdf_variable_name, ("time_step", n_node_dimension_name))

                f.set_array(nset_var_tab, "nset_var_tab", ("num_node_sets", "num_nset_var"))

        # Optional extra data to be stored in file for postprocessing tasks
        self._write_extra_data(f)

    @staticmethod
    def get_block_attributes(filepath):
        my_netcdf = m4.create_netcdf4_or_netcdf3(filepath)
        n_blocks = my_netcdf.get_dimension('num_el_blk')
        block_ids = my_netcdf.get_array('eb_prop1', dtype=np.int)[:n_blocks]
        block_attributes = {}
        for i in range(n_blocks):
            block_id = block_ids[i]
            # Get thickness info
            dimesion_name = 'num_att_in_blk'+str(i+1)
            if my_netcdf.has_dimension(dimesion_name):
                # print "Block", i, "num attribute:", my_netcdf4.get_dimension('num_att_in_blk'+str(i+1))
                _block_attributes = my_netcdf.get_array("attrib"+str(i+1))
                # By Exodus documentation, thickness is the first attribute if SHELL element type
                _block_thickness = _block_attributes[:, 0]
                # I don't know why Exodus store block thickness by an array (maybe it is used to
                # indicate thickness for each element)
                # because all the values are uniform (is that always true?), just pick the first one
                block_attributes[block_id] = bt.BlockAttribute.create(None, _block_thickness[0])

        '''
        # This part can be internally used by the production teams to quickly define component material
        if 'block_material_prop' in my_netcdf4.variables:
            block_material_map = my_netcdf4.get_array('block_material_prop', dtype=np.int)
            if len(block_material_map) == n_blocks:
                for i in range(n_blocks):
                    block_id = block_ids[i]
                    # Get material info
                    material_idx = block_material_map[i]
                    material_var_name = 'material' + str(material_idx+1)
                    if material_var_name in my_netcdf4.variables:
                        try:
                            material_name = my_netcdf4.get_attribute(material_var_name, 'name')
                            young_modulus = my_netcdf4.get_attribute(material_var_name, 'young_modulus')
                            poisson_ratio = my_netcdf4.get_attribute(material_var_name, 'poisson_ratio')
                            mass_density = my_netcdf4.get_attribute(material_var_name, 'mass_density')
                            block_attributes[block_id].update_material(material_name, young_modulus, poisson_ratio, mass_density)
                        except AttributeError:
                            pass
        '''

        my_netcdf.close()
        return block_attributes

    @staticmethod
    def get_elem_start_idxs_static(exo_block_datas):
        # Returns an array giving the mesh-wide idx of the first elem in each block
        # (and with an extra item on the end giving the number of elements in the mesh).
        idxs = [0]
        for exo_block_data in exo_block_datas:
            idxs.append(idxs[-1] + len(exo_block_data.elems))
        return np.array(idxs)

    def get_elem_start_idxs(self):
        return self.get_elem_start_idxs_static(self.exo_block_datas)

    def get_mesh_elem_idxs_from_block_elem_ids(self, block_elem_ids):
        # Returns the mesh-wide elem idxs elem given the block_elem_ids.
        # The block_elem_ids are pairs giving the block idx and the block-specific elem idx
        # (block_idx, block elem idx).
        elem_start_idxs = self.get_elem_start_idxs()
        mesh_elem_idxs = elem_start_idxs[block_elem_ids[:, 0]] + block_elem_ids[:, 1]
        return mesh_elem_idxs

    def get_block_elem_ids_from_mesh_elem_idxs(self, mesh_elem_idxs, *attach_arrays):
        # Returns block idx and block-specific elem idxs given mesh-wide elem idxs.
        # First row of the result is the block idx, next is the block-specific elem idx.
        # Additional arrays passed in as attach_arrays are stuck onto columns 2 and on.
        n = 2 + len(attach_arrays)
        block_elem_ids = np.zeros((len(mesh_elem_idxs), n), dtype=np.int32)
        elem_start_idxs = self.get_elem_start_idxs()
        for i in range(len(elem_start_idxs)-1):
            s = (mesh_elem_idxs >= elem_start_idxs[i]) & (mesh_elem_idxs < elem_start_idxs[i+1])
            block_elem_ids[s, 0] = i
            block_elem_ids[s, 1] = mesh_elem_idxs[s] - elem_start_idxs[i]

        for c, attach_array in enumerate(attach_arrays):
            block_elem_ids[:, c+2] = attach_array

        return block_elem_ids

    def get_minimum_order(self, dimension=None):
        exo_block_datas = [b for b in self.exo_block_datas
                           if dimension is None or b.elem_type.dimension == dimension]
        if len(exo_block_datas) == 0:
            return 0

        return min(b.elem_type.get_order() for b in exo_block_datas)

    def get_block_dimensions(self):
        return np.array([b.elem_type.dimension for b in self.exo_block_datas])

    def get_block_is_shell(self):
        return np.array([b.elem_type.is_shell for b in self.exo_block_datas], dtype=np.bool)

    def get_block_is_surface(self):
        return np.array([b.elem_type.is_surface for b in self.exo_block_datas], dtype=np.bool)

    def mutate_add_labeled_nodes(self, node_coords, node_ids, subdomain_id):
        # Add labeled nodes at positions specified by node_coords and with ids given by node_ids.
        # This function modifies the exo_data.
        node_coords = np.array(node_coords)
        n_dim = self.coords.shape[-1]
        if len(node_coords.shape) == 1:
            node_coords = node_coords.reshape(1, n_dim)
        assert len(node_coords) == len(node_ids)
        coords = np.concatenate((self.coords, node_coords))
        block_idx = len(self.exo_block_datas)
        elem_start_idx = self.get_elem_start_idxs()[-1]
        dtype = np.min_scalar_type(len(coords))
        elems = np.array([[i] for i in range(len(self.coords), len(coords))], dtype=dtype)
        exo_block_data = ExoBlockData(block_idx, "SPHERE", elems, elem_start_idx)
        self.coords = coords
        self.exo_block_datas.append(exo_block_data)
        mesh_node_idxs = reversed(range(len(coords)-1, len(coords)-len(node_ids)-1, -1))
        for mesh_node_idx, node_id in zip(mesh_node_idxs, node_ids):
            self.nodesets[node_id] = np.array([mesh_node_idx])
        self.block_idx_to_subdomain_id = np.append(self.block_idx_to_subdomain_id, subdomain_id)

    # Below functions are used for solution mesh. When we completely switch to use the HUI,
    # we should remove similar functions in VisualizationData.
    def get_field_names(self):
        return self.get_nodal_field_names() | self.get_elem_field_names() | \
               self.get_sideset_field_names()

    def get_nodal_field_names(self):
        nodal_field_names = set()
        if self.node_field_values is not None:
            nodal_field_names = set(self.node_field_values.keys())
        return nodal_field_names

    def get_elem_field_names(self):
        elem_field_names = set()
        for exo_block_data in self.exo_block_datas:
            elem_field_names.update(list(exo_block_data.elem_field_values.keys()))
            # We also want to include the "nodal field name" from "_elem_corner" field names
            # so that gmv.has_field() can work.
            for field_name in exo_block_data.elem_field_values.keys():
                if "_elem_corner" in field_name:
                    part_name = field_name.split("_elem_corner")[0]
                    elem_field_names.add(part_name)

        return elem_field_names

    def get_sideset_field_names(self):
        ss_field_names = set()
        for ss_id, ss_field_dict in self.sideset_field_values.items():
            ss_field_names.update(list(ss_field_dict.keys()))
        return ss_field_names

    def get_nodeset_field_names(self):
        ns_field_names = set()
        for ns_id, ns_field_dict in self.nodeset_field_values.items():
            ns_field_names.update(list(ns_field_dict.keys()))
        return ns_field_names

    def is_elem_field_integer_valued(self, field_name):
        found = False
        for exo_block_data in self.exo_block_datas:
            if field_name not in exo_block_data.elem_field_values:
                continue

            found = True
            if not np.all(exo_block_data.elem_field_values[field_name] ==
                   np.array(exo_block_data.elem_field_values[field_name], dtype=np.int64)):
                return False

        return found

    def get_all_elem_field_values(self, field_name, time_idx=None):
        if time_idx is None:
            time_idx = 0

        elem_field_values_list = []
        for exo_block_data in self.exo_block_datas:
            field_values = exo_block_data.elem_field_values.get(field_name, None)
            if field_values is not None:
                elem_field_values_list.append(field_values[time_idx])
        if len(elem_field_values_list) > 0:
            all_elem_field_values = np.concatenate(elem_field_values_list)
        else:
            all_elem_field_values = np.zeros(0)

        return all_elem_field_values

    def is_equal(self, other, check_node_field_values=False):
        # For testing purposes.
        equal = True
        equal &= np.allclose(self.coords, other.coords)
        if not equal:
            print("nodes not equal")
            print(self.coords)
            print(other.coords)
        equal &= len(self.exo_block_datas) == len(other.exo_block_datas)
        if not equal:
            print("num block not equal")
        for b1, b2 in zip(self.exo_block_datas, other.exo_block_datas):
            equal &= b1.is_equal(b2)
        equal &= is_sidesets_equal(self.sideset_elem_idxs, self.sideset_fside_idxs,
                                   other.sideset_elem_idxs, other.sideset_fside_idxs)
        equal &= is_map_array_equal(self.nodesets, other.nodesets)
        equal &= np.all(self.block_idx_to_subdomain_id == other.block_idx_to_subdomain_id)

        if check_node_field_values:
            node_field_values_keys = self.node_field_values.keys()
            equal &= set(node_field_values_keys) == set(other.node_field_values.keys())
            if not equal:
                print("node_field_values_keys not equal", node_field_values_keys, other.node_field_values.keys())

            if len(node_field_values_keys) > 0:
                for field_name in node_field_values_keys:
                    equal &= np.allclose(self.node_field_values[field_name], other.node_field_values[field_name])
                if not equal:
                    print("node_field {} not equal".format(field_name))

        return equal

    def is_surface_mesh(self):
        return max(self.get_block_dimensions()) < 3 and not np.any(self.get_block_is_shell())

    def is_pure_1d_mesh(self):
        block_dimensions = self.get_block_dimensions()
        if len(np.unique(block_dimensions)) == 1 and max(block_dimensions) == 1:
            return True
        return False

    def has_1d_blocks(self):
        block_dimensions = self.get_block_dimensions()
        return 1 in block_dimensions

    def get_min_max_field_value_and_index(self, field_option, is_get_max, time_idx=0):
        extreme_node_index = None
        extreme_node_value = None
        extreme_node_position = None
        extreme_elem_info = None

        for part_name in field_option.get_names_for_combine_bound():
            if part_name in self.node_field_values:
                field_values = self.node_field_values[part_name]
                field_values_at_time_step = field_values[time_idx]
                if is_get_max:
                    extreme_node_index_i = np.nanargmax(field_values_at_time_step)
                else:
                    extreme_node_index_i = np.nanargmin(field_values_at_time_step)

                extreme_node_value_i = field_values_at_time_step[extreme_node_index_i]
                if is_get_max:
                    is_new = extreme_node_value is None or extreme_node_value_i > extreme_node_value
                else:
                    is_new = extreme_node_value is None or extreme_node_value_i < extreme_node_value

                if is_new:
                    extreme_node_index = extreme_node_index_i
                    extreme_node_value = extreme_node_value_i
                    extreme_node_position = self.coords[extreme_node_index_i]

            test_elem_corner_field = f'{part_name}_elem_corner_0'
            for block_data in self.exo_block_datas:
                if test_elem_corner_field not in block_data.elem_field_values:
                    continue
                # for now, elem_corner fields are only on first order nodes
                for corner_idx in range(block_data.elem_type.get_n_nodes(high_order_node=False)):
                    field_values = \
                        block_data.elem_field_values[f'{part_name}_elem_corner_{corner_idx}']
                    field_values_at_time_step = field_values[time_idx]

                    if is_get_max:
                        extreme_elem_index_i = np.nanargmax(field_values_at_time_step)
                    else:
                        extreme_elem_index_i = np.nanargmin(field_values_at_time_step)

                    extreme_node_value_i = field_values_at_time_step[extreme_elem_index_i]
                    if is_get_max:
                        is_new = extreme_node_value is None or \
                                 extreme_node_value_i > extreme_node_value
                    else:
                        is_new = extreme_node_value is None or \
                                 extreme_node_value_i < extreme_node_value

                    if is_new:
                        extreme_node_index_i = block_data.elems[extreme_elem_index_i, corner_idx]
                        extreme_node_index = extreme_node_index_i
                        extreme_node_value = extreme_node_value_i
                        extreme_node_position = self.coords[extreme_node_index_i]
                        block_id = self.block_idx_to_subdomain_id[block_data.block_idx]
                        extreme_elem_info = (block_id, extreme_elem_index_i)

        return extreme_node_index, extreme_node_value, extreme_node_position, extreme_elem_info

    def __repr__(self):
        repr_str = "ExoData(n_nodes: {}, n_blocks: {}, n_nodal_fields: {})".format(
            self.coords.shape[0], len(self.exo_block_datas), len(self.node_field_values))
        repr_str += "\n Element blocks:"
        for b in self.exo_block_datas:
            block_id = self.block_idx_to_subdomain_id[b.block_idx]
            repr_str += "\n"
            repr_str += "   + block idx {}: ID: {}, elem_type: {}, n_elems: {}, n_elem_fields: {" \
                        "}".format(
                b.block_idx, block_id, b.elem_type.name, len(b.elems), len(b.elem_field_values))
        if self.exo_edge_block_datas is not None:
            repr_str += "\n Edge blocks:"
            for e in self.exo_edge_block_datas:
                edge_block_id = self.edge_block_idx_to_subdomain_id[e.block_idx]
                repr_str += "\n"
                repr_str += "   + edge block idx {}: ID: {}, elem_type: {}, n_elems: {}".format(
                    e.block_idx, edge_block_id, e.elem_type.name, len(e.elems))
        return repr_str


def is_sidesets_equal(elem_idxs_a, fside_idxs_a, elem_idxs_b, fside_idxs_b):
    equal = True
    if not set(elem_idxs_a.keys()) == set(elem_idxs_b.keys()):
        return False
    if not set(fside_idxs_a.keys()) == set(fside_idxs_b.keys()):
        return False
    for sideset_id in elem_idxs_a.keys():
        equal &= is_sideset_equal(elem_idxs_a[sideset_id], fside_idxs_a[sideset_id],
                                  elem_idxs_b[sideset_id], fside_idxs_b[sideset_id])
    return equal


def is_sideset_equal(elem_idxs_a, fside_idxs_a, elem_idxs_b, fside_idxs_b):
    equal = True

    idxs_a = np.lexsort((elem_idxs_a, fside_idxs_a))
    a = [(fside_idxs_a[i], elem_idxs_a[i]) for i in idxs_a]
    idxs_b = np.lexsort((elem_idxs_b, fside_idxs_b))
    b = [(fside_idxs_b[i], elem_idxs_b[i]) for i in idxs_b]
    equal &= np.all(a == b)

    return equal


def is_map_array_equal(a, b):
    if not set(a.keys()) == set(b.keys()):
        return False
    for key in a.keys():
        if not np.all(a[key] == b[key]):
            return False

    return True
