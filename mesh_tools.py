import collections
import numpy as np

import set_akselos_path
import akselos.mesh.exo_data as ed
# Tools to transform ExoData into a RefMesh.

# Words used in variable names here:
# "elem": the FEA element (which in 3D typically has "type" cube or tetrahedron)
# "block": one group of elements, all of the same elem type and same attributes (e.g. subdomain id)
# "face": a face of the element type (independent of the mesh)
# "side": a face on a particular element in the mesh
# "fedge": same as a face, but for an edge
# "sedge": same as a side, but for an edge
# "boundary": the surface of a block
# "idx": 0-based index into an array
# "id": integer identifier, often used as dictionary key
# "n_": number of
# "_range": a pair of integers giving start and stop (+1) idxs.

# Use this to get numpy to print more elements: np.set_printoptions(threshold=50000)

# A 2D face, which is the side of an element in the case of a 3D element,
# or a full element in the case of 2D elements.  For two-sided (shell elements), we have two
# Sides per element.
# The elem_idx is an index into the block's elems (not a mesh-wide elem idx).
# If adjacent_block_idx != -1, the side is inside the mesh between two blocks.
# Each (Side, node) pair, with node belonging to the side, is given a unique index, the "split"
# idx; the split idxs for this Side start at split_start_idx.  The splits idxs can be mapped
# to draw coord idxs.
Side = [('elem_idx', np.int32),
        ('block_idx', np.int32),
        ('fside_idx', np.int32),
        ('adjacent_block_idx', np.int32),
        ('split_start_idx', np.int32)]


# A Sedge is an edge segment in the mesh.  The face_sub_idx0 gives the index into the fside_tuple
# for the start node of the edge.  The end node is the next cyclic index in the
# fside_tuple.  If side_idx0 and side_idx1 is -1 it means this edge is not touching any sides;
# otherwise, if side_idx1 is -1 it means this edge is touching only side with side_idx0.
# If the mesh has edges touching more than two sides (e.g. a 2D mesh with two surfaces
# touching as a "T") then there will be a Sedges created for each pair of sides; in the "T"
# example, there will be three Sedges created at the intersection.
# Thuc: previously, 'block_idx' was only set for "free" sedges. But now I implement it for all
# edges to be able to show mesh/edge on certain subdomains (see #2799).
DefaultSedge = [('mesh_node_idxs', np.uint32, 2),
                ('block_idx', np.int32),
                ('side_idx0', np.int32),
                ('side_idx1', np.int32),
                ('packed_fields', np.uint32)]

SEDGE_SMALL_FIELDS = {
    'face_sub_idx0': (0, 5),
    'face_sub_idx1': (5, 5),
    'is_sharp': (10, 1),
    'is_shell_boundary': (11, 1)
}

# A node in the mesh, corresponding to the mesh's coords array.  If draw_coord_idx is not -1,
# then the node is visible (i.e. on the mesh surface).
MeshNode = [('draw_coord_idx', np.int32)]


# Info about mesh node that is on the mesh boundary and will be drawn to the screen.  The
# coordinates are stored in a separate array.
DrawCoordInfo = [('block_idx', np.int32), ('mesh_node_idx', np.int32)]


# Used in get_side_sedges
PairInfo = [('side_idx', np.int32),
            ('face_sub_idx', np.int32),
            ('nodes', np.int32, 2),
            ('block_idx', np.int32),
            ('is_shell', np.bool)]


# The Sedge array is the largest single user of memory for some models.  To decrease its memory
# footprint, we pack small integers bitwise into a single large integer.  Unfortunately, numpy
# does not seem to have functionality to do this packing/unpacking, so we do it ourselves here.
def pack(target, new_values, idxs, position, count):
    assert np.all((new_values < 2**count) & (new_values >= 0))
    shifted_values = (new_values << position)
    if isinstance(shifted_values, np.ndarray):
        shifted_values = shifted_values.astype(target.dtype)

    mask = (2**count - 1) << position

    if idxs is not None:
        target[idxs] &= target.dtype.type(~mask)
        target[idxs] |= shifted_values
    else:
        target &= target.dtype.type(~mask)
        target |= shifted_values


def unpack(array, position, count):
    mask = (2**count - 1) << position
    values = (array & mask) >> position
    if count == 1:
        result = values.astype(np.bool)
    else:
        result = values.astype(np.uint32)

    return result


def get_sedge_field(sedges, field_name):
    return unpack(sedges['packed_fields'], *SEDGE_SMALL_FIELDS[field_name])


def set_sedge_field(sedges, field_name, new_values, idxs=None):
    pack(sedges['packed_fields'], new_values, idxs, *SEDGE_SMALL_FIELDS[field_name])


def invert_argsort(sort_idxs):
    # Inverts an argsort in linear time (can also be done using a second argsort,
    # but we want to avoid this second sort).
    n = len(sort_idxs)
    r = np.zeros(n, dtype=sort_idxs.dtype)
    r[sort_idxs] = np.arange(n)
    return r


def get_is_leader(m):
    # Returns an array of bools indicating if the corresponding element of the input array
    # m (m is typically sorted) is the first element in a run of equal elements.
    if len(m) > 0:
        return np.concatenate(([True], m[1:] != m[:-1]))
    else:
        return np.zeros(0, dtype=np.bool)


def get_repetition_count(m, get_pairs=False, get_first_idxs=False):
    # Returns an array that gives a count of occurrences of each element in the input array.
    # If get_pairs is True, also finds pairs of elements in the input array that are equal,
    # and returns their indices in an n-by-2 array (for each set of k>2 equal elements,
    # gives k-1 "chained" pairs).
    # If get_first_idxs is True, also returns the indexes of the first appearance of each item
    # in the input array (I think it's the first-- may want to check this).
    sort_idxs = m.argsort(kind='mergesort')
    sorted_m = m[sort_idxs]
    is_leader = get_is_leader(sorted_m)
    counts = np.diff(np.concatenate(np.nonzero(is_leader) + ([len(sorted_m)],)))
    unsort_idxs = invert_argsort(sort_idxs)
    result = counts[np.cumsum(is_leader)-1][unsort_idxs],
    if get_pairs:
        not_leader_idxs = np.where(~is_leader)[0]
        second_idxs = sort_idxs[not_leader_idxs]
        first_idxs = sort_idxs[not_leader_idxs - 1]
        result = result + (np.vstack((first_idxs, second_idxs)).T,)
    if get_first_idxs:
        result = result + (sort_idxs[is_leader],)
    return result


def get_row_repetition_count(m, ordered=False):
    # Returns an array that gives the number of occurrences of each row in the input array.
    if not ordered:
        m = np.sort(m)

    m = hash_rows(m)

    return get_repetition_count(m)[0]


def hash_rows(m):
    # Hashes each row of m into a single 64-bit integer.  This allows for a fast sort,
    # but has a tiny chance of collision.
    if m.dtype.names is not None:
        # Note: structured row means each element in a row is defined by dtype (like Side type above)
        return _hash_structured_rows(m)

    old_settings = np.seterr(over='ignore')  # Don't care about overflow when hashing.
    # Polynomial String Hashing; https://cstheory.stackexchange.com/questions/3390/is-there-a-hash-function-for-a-collection-i-e-multi-set-of-integers-that-has
    try:
        p = 8614742970397400329
        nc = m.shape[1]
        xs = np.ones(nc, dtype=np.int64)
        for i in range(1, nc):
            xs[i] = xs[i-1]*p
        # (p^0, p^1, p^2, ..., p^(n-1))
        # Use Horner's method to get more efficient
    finally:
        np.seterr(**old_settings)

    return np.sum(m*xs, axis=1)


def _hash_structured_rows(m):
    old_settings = np.seterr(over='ignore')  # Don't care about overflow when hashing.
    try:
        p = 8614742970397400329
        nc = len(m.dtype.names)
        xs = np.ones(nc, dtype=np.int64)
        for i in range(1, nc):
            xs[i] = xs[i-1] * p
        s = np.zeros(m.shape[0], dtype=np.int64)
        for i, name in enumerate(m.dtype.names):
            s += xs[i]*m[name]
        return s
    finally:
        np.seterr(**old_settings)


def get_np_dtype(item_or_items):
    # Get the smallest dtype that fits the item or items.  Useful for ints, not so much floats.
    # Result is always signed integer type.
    if len(item_or_items) == 0:
        return np.int8

    mx = np.max(np.abs(item_or_items))
    if mx == 0:
        return np.int8

    return np.min_scalar_type(-mx-1)


def reduce_structured_array_memory_footprint(array):
    new_types = []
    for name in array.dtype.names:
        shape = array.dtype.fields[name][0].shape
        if name == 'packed_fields':
            new_dtype = array[name].dtype
        else:
            new_dtype = get_np_dtype(array[name])
        new_types.append((name, new_dtype, shape))

    reduced_array = np.zeros(len(array), new_types)

    for name in array.dtype.names:
        reduced_array[name] = array[name]

    return reduced_array


def get_sedges(exo_data, sides):
    # Get side and free sedges.  Remove any free sedges if it is the same edge as a side sedge.
    # <-- I forget why we do this, but since we now have block_idx stored in the sedge, it seems
    # like this could mess something up.  The block_idx isn't currently stored for a sedge attached
    # to a side, and even if it were it wouldn't be the correct block_idx.  I guess an easy fix
    # is to assign the block_idx from any free sedge being removed to its counterpart.
    # Note that we also remove draw coords that are part of edge labeling-- see
    # combine_draw_coords().
    # As this method usually takes a lot of time, we may need a progress bar.
    side_sedges = get_side_sedges(exo_data, sides)
    # For component mesh, we used to remove edges that have duplicate 'mesh_node_idxs',
    # but that will also remove sedges from 1d (stiffener) blocks that touch shell blocks.
    # Now, because we also want to plot scalar field (e.g. property_ids) on those 1d
    # (stiffener) blocks, I include the 'block_idx' to the repetition check to take
    # in account sedges from 1d (stiffener) blocks.
    reduced_sedges = reduce_structured_array_memory_footprint(side_sedges)
    return reduced_sedges


def get_side_centroids_and_areas(exo_data, sides, mapped_coords=None, exterior_only=True):
    # Get the centroids for the given sides using coordinates directly from the mesh.
    if mapped_coords is not None:
        coords = mapped_coords
    else:
        coords = exo_data.coords
    d = coords.shape[-1]
    if d == 1:
        return np.array([])
    exo_block_data = exo_data.exo_block_datas[0]
    side_elems = exo_block_data.elems
    sides_slice = np.arange(len(side_elems))
    side_areas = np.zeros(len(side_elems))
    centroid_sums = np.zeros((len(side_elems), 3))
    one_vertex = np.zeros((len(side_elems), 3))
    _, face_tuple = exo_block_data.elem_type.get_faces()[0]
    n_face_nodes = len(face_tuple)
    elem_idxs = sides_slice
    for i in range(1, n_face_nodes-1):
        j = i+1
        f0, fi, fj = face_tuple[0], face_tuple[i], face_tuple[j]
        n0, ni, nj = side_elems[elem_idxs, f0], side_elems[elem_idxs, fi], side_elems[elem_idxs, fj]
        c0, ci, cj = coords[n0], coords[ni], coords[nj]
        triangle_centroids = (1.0/3.0) * (c0 + ci + cj)
        triangle_vectors = np.cross(ci-c0, cj-c0)
        if coords.shape[-1] == 3:
            triangle_areas = 0.5 * np.linalg.norm(triangle_vectors, axis=1)
        else:
            triangle_areas = 0.5 * np.abs(triangle_vectors)
        centroid_sums[sides_slice] += triangle_centroids * \
                                      triangle_areas.reshape((len(triangle_areas),1))
        one_vertex[sides_slice] = c0
        side_areas[sides_slice] += triangle_areas

    is_ok = side_areas != 0.0
    centroids = np.zeros(centroid_sums.shape)
    centroids[is_ok] = centroid_sums[is_ok]/side_areas.reshape((len(side_areas), 1))[is_ok]
    centroids[~is_ok] = one_vertex[~is_ok]
    return centroids, side_areas


def get_ordered_pair_infos(exo_data, in_pair_infos, sides, side_normals, exterior_only=True):
    # Helper for get_side_sedges.
    # If an edge is touching more than two elements (e.g. shell mesh in a "T" shape), we
    # need to do some geometric calculations to determine connectedness.  In this case there are
    # six Sides, and three Sedges at the "T intersection; we need to figure out which two Sides
    # each Sedge is connected to.  We do this by ordering each Side by angle around the axis,
    # and pairing each Side with the next side in theta and with normals pointing opposite
    # directions in theta.
    if len(in_pair_infos) == 0:
        return [], []

    side_centroids, side_areas = get_side_centroids_and_areas(
        exo_data, sides, exterior_only=exterior_only)

    # We ignore any cases where we have a side of tiny area or edge of tiny length.
    is_good_sides = side_areas > 1e-9*np.max(side_areas)
    coords = exo_data.coords
    test_node_idxs = in_pair_infos['nodes']
    test_z_dirs = (coords[test_node_idxs[:, 1]] - coords[test_node_idxs[:, 0]]) / 2.0
    test_lengths = np.linalg.norm(test_z_dirs, axis=1)
    max_test_length = np.max(test_lengths)
    is_good_edges = test_lengths > max_test_length*1e-6
    unsorted_pair_infos = in_pair_infos[is_good_edges & is_good_sides[in_pair_infos['side_idx']]]

    # Sort the pair_infos into groups that have that share an edge/axis.  We consider each
    # group separately, but the operations below are performed on entire arrays for performance.
    unsorted_node_idxs = np.sort(unsorted_pair_infos['nodes'])
    unsorted_hashes = hash_rows(unsorted_node_idxs)
    sort_idxs = unsorted_hashes.argsort(kind='mergesort')
    node_idxs = unsorted_node_idxs[sort_idxs]
    hashes = unsorted_hashes[sort_idxs]
    pair_infos = unsorted_pair_infos[sort_idxs]
    is_leader = get_is_leader(hashes)

    # Find outward direction of side around the axis:
    # cross axis with centroid and dot result with side normal, take sign.
    origins = (coords[node_idxs[:, 1]] + coords[node_idxs[:, 0]])/2.0
    z_dirs = normalize_vectors(coords[node_idxs[:, 1]] - coords[node_idxs[:, 0]])
    centroids = side_centroids[pair_infos['side_idx']] - origins
    normals = side_normals[pair_infos['side_idx']]
    crosses = np.cross(z_dirs, centroids)
    signs = np.sign(np.sum(crosses*normals, axis=1))
    # Pick a centroid from each group to act as the x direction, and copy to all elements of the
    # group.
    leader_centroids = centroids[is_leader][np.cumsum(is_leader)-1]

    # Find angle about first centroid and sedge axis.
    y_dirs = normalize_vectors(np.cross(z_dirs, leader_centroids))
    x_dirs = normalize_vectors(np.cross(y_dirs, z_dirs))
    y_vals = np.sum(y_dirs*centroids, axis=1)
    x_vals = np.sum(x_dirs*centroids, axis=1)

    # Do a little hack to avoid the common case in which thetas are exactly at -pi (= pi).
    # In this case, because of floating point error, one may wind up at -pi and the other at pi,
    # which will mess up the sorting below.
    thetas = np.arctan2(y_vals, x_vals) - 0.4709830508038494
    thetas[thetas < -np.pi] += 2*np.pi

    # Sort arrays by angle (after sorting by sedge hash to maintain grouping based on shared axis)
    th_sort_idxs = np.lexsort((thetas, hashes))
    th_pair_infos = pair_infos[th_sort_idxs]
    th_signs = signs[th_sort_idxs]

    # Split into two arrays, one pointing positive theta, one negative
    infos_p = th_pair_infos[th_signs > 0]
    infos_m = th_pair_infos[th_signs < 0]
    hash_p = hashes[th_signs > 0]
    hash_m = hashes[th_signs < 0]
    if len(hash_m) != len(hash_p) or not np.all(hash_m == hash_p):
        # Shouldn't get here if we have all shell elements.
        # If the mesh has bad elements (due to very small surfaces), we can get here
        print("Warning: component mesh might contain very small elements. Please run check on it")
        return [], []

    # Match up sides that share an edge by moving one array forward within its axis group.
    #    Roll one array forward.  Array to roll forward has sign in direction of increasing angle.
    #    Wrap within groups by moving elements that are at a leader back to the previous leader.
    is_leader_p = get_is_leader(hash_p)
    infos_p1 = np.roll(infos_p, 1)
    infos_p1[is_leader_p] = np.roll(infos_p1[is_leader_p], -1)

    return [infos_p1, infos_m]


def get_side_sedges(exo_data, sides, exterior_only=True):
    # Returns an array of Sedge for the given sides.  Determines sharpness of each sedge.
    # The direction of the sedge is determined by the direction of the edge on side_idx0.
    # As this method usually takes a lot of time, we may need a progress bar.

    # Now we have shell-solid component that solid block is wrapped by shell block.
    # In that case, we have to iter sides in interior as well.
    exo_block_data = exo_data.exo_block_datas[0]
    side_elems = exo_block_data.elems
    _, face_tuple = exo_block_data.elem_type.get_faces()[0]
    n_face_nodes = 3
    shell_pair_infos = np.zeros(n_face_nodes * len(side_elems), dtype=PairInfo)
    shell_pairs_counter = 0
    block_idx = exo_block_data.block_idx
    face_side_idxs = np.arange(len(side_elems))
    face_sides = sides[face_side_idxs]
    n_face_sides = len(face_sides)
    for fedge_idx in range(n_face_nodes):
        j = (fedge_idx+1) % n_face_nodes
        pair_info_slice = shell_pair_infos[
            slice(shell_pairs_counter, shell_pairs_counter + n_face_sides)]
        shell_pairs_counter += n_face_sides
        # Note the 0->1 direction in the edge node pair is the same as the
        # direction of that edge on the side.
        pair_info_slice['nodes'][:, 0] = side_elems[:, face_tuple[fedge_idx]]
        pair_info_slice['nodes'][:, 1] = side_elems[:, face_tuple[j]]
        pair_info_slice['face_sub_idx'] = fedge_idx
        pair_info_slice['side_idx'] = face_side_idxs
        pair_info_slice['block_idx'] = block_idx

    side_normals = get_side_normals(exo_data, sides)
    if len(shell_pair_infos) > 0:
        shell_infos1, shell_infos2 = get_ordered_pair_infos(
            exo_data, shell_pair_infos, sides, side_normals, exterior_only=exterior_only)
    else:
        shell_infos1, shell_infos2 = [], []
    n_shell_sedges = len(shell_infos1)
    n_sedges = n_shell_sedges
    sedges = np.zeros(n_sedges, dtype=DefaultSedge)
    start_idx = 0

    sharp_tolerance = 0.45

    if n_shell_sedges > 0:
        infos0, infos1 = shell_infos1, shell_infos2
        s = slice(start_idx, start_idx+n_shell_sedges)
        sedges_slice = sedges[s]
        normals_0 = side_normals[infos0['side_idx']]
        normals_1 = side_normals[infos1['side_idx']]
        dot = np.zeros(len(normals_0))
        for xi in range(0, normals_0.shape[-1]):
            dot += normals_0[:, xi]*normals_1[:, xi]
        sedges_slice['block_idx'] = infos0['block_idx']
        sedges_slice['mesh_node_idxs'] = infos0['nodes']
        sedges_slice['side_idx0'] = infos0['side_idx']
        set_sedge_field(sedges_slice, 'face_sub_idx0', infos0['face_sub_idx'])
        sedges_slice['side_idx1'] = infos1['side_idx']
        set_sedge_field(sedges_slice, 'face_sub_idx1', infos1['face_sub_idx'])
        set_sedge_field(sedges_slice, 'is_sharp', (dot < sharp_tolerance) | (
            infos0['block_idx'] != infos1['block_idx']))
        set_sedge_field(sedges_slice, 'is_shell_boundary', (
            sides['block_idx'][infos0['side_idx']] ==
            sides['block_idx'][infos1['side_idx']]) & (
            sides['elem_idx'][infos0['side_idx']] ==
            sides['elem_idx'][infos1['side_idx']]))

    return sedges


def get_sides(exo_data):
    # Find the sides that compose the boundary of a block (including sides that are internal
    # to the mesh, but at the boundary between two blocks).

    block_infos = get_sides_per_block(exo_data)

    interfaces_idxs = {}
    n_blocks = len(exo_data.exo_block_datas)
    back_counts = {}
    for block_idx_1 in range(n_blocks):
        if block_idx_1 not in block_infos:
            continue

        info_1 = block_infos[block_idx_1]
        boundary_side_idxs_1 = info_1.boundary_side_idxs
        boundary_side_node_idxs_1 = info_1.side_node_idxs[boundary_side_idxs_1]
        # Typically we expect idxs_1 not have repetitions, but that may not be the case in some
        # circumstances.
        counts_1 = get_row_repetition_count(boundary_side_node_idxs_1)

        n_1 = len(boundary_side_node_idxs_1)
        all_counts_1 = np.zeros(n_1, dtype=np.int32)  # count for non-adjacent sides
        block_1_min, block_1_max = info_1.block_bounds
        for block_idx_2 in range(block_idx_1+1, n_blocks):
            if block_idx_2 not in block_infos:
                continue
            info_2 = block_infos[block_idx_2]
            block_2_min, block_2_max = info_2.block_bounds
            if np.any(block_1_max < block_2_min) or np.any(block_2_max < block_1_min):
                # This algorithm is O(N^2/2) in subdomains and so is very slow for models with
                # many subdomains.  We speed things up a little by checking bounding boxes
                # before calling the slow get_row_repetition_count.
                continue

            boundary_side_idxs_2 = info_2.boundary_side_idxs
            boundary_side_node_idxs_2 = info_2.side_node_idxs[boundary_side_idxs_2]
            n_2 = len(boundary_side_node_idxs_2)
            counts_2 = get_row_repetition_count(boundary_side_node_idxs_2)

            counts_1_2 = get_row_repetition_count(
                np.vstack((boundary_side_node_idxs_1, boundary_side_node_idxs_2)))

            counts_repeat_1 = counts_1_2[:n_1] - counts_1
            all_counts_1 += counts_repeat_1

            counts_repeat_2 = counts_1_2[n_1:] - counts_2
            back_counts.setdefault(block_idx_2, np.zeros(n_2, dtype=np.int32))
            back_counts[block_idx_2] += counts_repeat_2

            # Sometimes, we want to show duplicate blocks (see #3975),
            # that's why I also include boundary sides that appear twice.
            interfaces_idxs[block_idx_1, block_idx_2] = \
                boundary_side_idxs_1[(counts_repeat_1 == 1) | (counts_repeat_1 == 2)]
            interfaces_idxs[block_idx_2, block_idx_1] = boundary_side_idxs_2[counts_repeat_2 == 1]

            # # When a solid block is wrapped by a shell block, the repetition is 1 or 2
            # interfaces_idxs[block_idx_1, block_idx_2] = boundary_side_idxs_1[counts_repeat_1 >= 1]
            # interfaces_idxs[block_idx_2, block_idx_1] = boundary_side_idxs_2[counts_repeat_2 >= 1]

        if block_idx_1 in back_counts.keys():
            all_counts_1 += back_counts[block_idx_1]
        interfaces_idxs[block_idx_1, -1] = boundary_side_idxs_1[all_counts_1 == 0]

    n_sides = sum(len(idxs) for idxs in interfaces_idxs.values())
    boundaries_sides = np.zeros(n_sides, dtype=Side)
    # boundary_sides_ranges = {}
    start_idx = 0
    for block_idxs, idxs in interfaces_idxs.items():
        if block_idxs[0] not in block_infos:
            continue

        info = block_infos[block_idxs[0]]
        n = len(idxs)
        if n > 0:
            side_range = start_idx, start_idx+n
            # if block_idxs[1] == -1:
            #     boundary_sides_ranges[block_idxs[0]] = side_range
            # else:
            #     boundary_sides_ranges[block_idxs] = side_range

            b = boundaries_sides[slice(*side_range)]
            b['elem_idx'] = info.side_elem_idxs[idxs]
            b['fside_idx'] = info.side_face_tuple_idxs[idxs]
            b['block_idx'] = block_idxs[0]
            b['adjacent_block_idx'] = block_idxs[1]
            start_idx += n

    return boundaries_sides


def get_sides_per_block(exo_data):
    max_n_face_nodes = 0
    for exo_block_data in exo_data.exo_block_datas:
        elem_type = exo_block_data.elem_type
        if elem_type.dimension < 2:
            continue
        this_max_n_face_nodes = max(len(fside_tuple) for fside_tuple in elem_type.fside_tuples)
        max_n_face_nodes = max(max_n_face_nodes, this_max_n_face_nodes)
    block_infos = {}
    if max_n_face_nodes == 0:
        # 1d component, skip searching for sides
        return block_infos

    SideBlockInfo = collections.namedtuple(
        'SideBlockInfo', ['elems', 'side_node_idxs', 'boundary_side_idxs',
                          'side_elem_idxs', 'side_face_tuple_idxs', 'block_bounds'])
    # Note:
    # Find all sides of all elements in blocks
    # elems: elems in a block
    # n_sides = n_elems * n_faces,
    #     where: n_elems is number of elems in a block, n_faces is number of faces in an element
    # side_elem_idxs: elem indexes in all sides
    # side_node_idxs: node indexes in all sides
    # side_face_tuple_idxs: face indexes in all sides
    # boundary_side_idxs: indexes of boundary sides
    # block_bounds: bounding box
    # max_n_face_nodes: in cases, such as, pyramid or wedge, sides have different number of nodes

    for exo_block_data in exo_data.exo_block_datas:
        elem_type = exo_block_data.elem_type
        if elem_type.dimension < 2:
            continue

        elems = exo_block_data.elems
        block_vertices = exo_data.coords[elems.flatten()]
        block_bounds = block_vertices.min(0), block_vertices.max(0)

        faces = elem_type.get_faces()
        n_elems = len(elems)
        n_faces = len(faces)
        n_sides = n_faces * n_elems
        block_idx = exo_block_data.block_idx
        side_node_idxs = -np.ones((n_sides, max_n_face_nodes), dtype=np.int32)
        for i, face_tuple in faces:
            side_node_idxs[i * n_elems:(i + 1) * n_elems, :len(face_tuple)] = elems[:, face_tuple]

        if elem_type.dimension == 3:
            # For 3D elements, we remove any sides that are internal to the mesh.
            # We can't see them so we don't need to render them or pick them.
            boundary_side_idxs = np.where(get_row_repetition_count(side_node_idxs) == 1)[0]
        elif elem_type.dimension == 2:
            # Assume we can render/pick all 2D elements.
            boundary_side_idxs = np.arange(n_sides)

        side_elem_idxs = np.tile(np.arange(n_elems), n_faces)
        side_face_tuple_idxs = np.repeat(np.arange(n_faces), n_elems)
        block_infos[block_idx] = SideBlockInfo(
            elems, side_node_idxs, boundary_side_idxs, side_elem_idxs,
            side_face_tuple_idxs, block_bounds)
    return block_infos


def get_side_crosses(exo_data, sides):
    # Get the cross product for the given sides using coordinates directly from the mesh.
    coords = exo_data.coords
    d = coords.shape[-1]
    if d == 1:
        return np.array([])
    exo_block_data = exo_data.exo_block_datas[0]
    elems = exo_block_data.elems
    elem_idxs = np.arange(len(elems))
    crosses = np.zeros((len(sides[elem_idxs]), 3), dtype=coords.dtype)
    _, face_tuple = exo_block_data.elem_type.get_faces()[0]
    n_face_nodes = 3
    origin = coords[elems[elem_idxs, face_tuple[0]]]
    for i in range(n_face_nodes):
        j = (i+1) % n_face_nodes
        fi, fj = face_tuple[i], face_tuple[j]
        node_idxs = elems[elem_idxs][:, [fi, fj]]
        ci = coords[node_idxs]
        s = np.cross(ci[:, 0, :] - origin, ci[:, 1, :] - origin)
        if d == 3:
            crosses[elem_idxs] += s
        elif d == 2:
            crosses[elem_idxs, 2] += s
        else:
            assert False

    return crosses


def get_side_normals(exo_data, sides):
    # Get normals for the given sides using coordinates directly from the mesh.
    crosses = get_side_crosses(exo_data, sides)
    if len(crosses) == 0:
        return crosses

    return normalize_vectors(crosses)


def normalize_vectors(vectors, return_float32=False, return_norms=False):
    # Normalize vectors, and give a default value to any vectors that have a tiny norm.
    if len(vectors) == 0:
        return np.array(vectors)

    is_one_vector = len(vectors.shape) == 1
    if is_one_vector:
        vectors = vectors.reshape((1, len(vectors)))

    norm = np.linalg.norm(vectors, axis=1)
    is_tiny = np.where(norm < 1e-12*np.max(norm) + 1e-36)[0]
    norm[is_tiny] = 1.0
    normals = (vectors.T/norm).T
    if len(is_tiny) > 0:
        default = np.zeros(normals.shape[1])
        default[0] = 1.0
        default = default/np.linalg.norm(default)
        normals[is_tiny] = default

    if is_one_vector:
        normals = normals.reshape(vectors.shape[-1])

    if return_float32 and not normals.dtype == np.float32:
        normals = np.array(normals, dtype=np.float32)

    if return_norms:
        return normals, norm
    else:
        return normals


def get_compress_node_ids_map(node_ids):
    map_node_ids = {}  # old id to new id
    for idx, id in enumerate(node_ids):
        map_node_ids[id] = idx
    return map_node_ids


def get_surface_mesh(exo_data=None, sides=None, external_boundary_only=True, merge_blocks=True):
    # # When a solid block is wrapped by a shell block, the repetition is 1 or 2
    # interfaces_idxs[block_idx_1, block_idx_2] = boundary_side_idxs_1[counts_repeat_1 >= 1]
    # interfaces_idxs[block_idx_2, block_idx_1] = boundary_side_idxs_2[counts_repeat_2 >= 1]

    assert not (exo_data is None and sides is None)
    if sides is None:
        sides = get_sides(exo_data)
    boundary_sides = None
    if external_boundary_only:
        # skip 2D boundaries between two blocks
        boundary_sides = sides[sides['adjacent_block_idx'] == -1]
    else:
        boundary_sides = sides

    shell_elems_blocks = {}
    map_elems_idxs = []
    map_fside_idxs = []

    count_new_blocks = 0
    for block_idx, exo_block_data in enumerate(exo_data.exo_block_datas):
        if exo_block_data.elem_type.dimension < 2:
            continue

        block_sides = boundary_sides[boundary_sides['block_idx'] == block_idx]
        if exo_block_data.elem_type.dimension == len(exo_block_data.elem_type.get_faces()) == 2:
            assert not block_sides.shape[0]%2  # a shell block has top and bottom sides
            block_sides = block_sides[:block_sides.shape[0]//2]

        new_elems_dict = {}
        for fside_idx, fside_tuple in enumerate(exo_block_data.elem_type.fside_tuples):
            # block_sides['fside_idx']: external face indexes only
            slices = block_sides['fside_idx'] == fside_idx
            if not np.any(slices):
                continue

            elem_global_idxs = block_sides['elem_idx'][slices] + exo_block_data.elem_start_idx
            map_elems_idxs += elem_global_idxs.tolist()
            map_fside_idxs += [fside_idx for _ in block_sides['elem_idx'][slices].tolist()]
            #
            elems = exo_block_data.elems[block_sides['elem_idx'][slices]][:, list(fside_tuple)]
            n_nodes_per_elem = len(elems[0])
            if n_nodes_per_elem in new_elems_dict.keys():
                new_elems_dict[n_nodes_per_elem] = np.vstack((new_elems_dict[n_nodes_per_elem], elems))
            else:
                new_elems_dict[n_nodes_per_elem] = elems
        #
        if not len(new_elems_dict.keys()):
            continue

        if merge_blocks:
            for n_nodes_per_elem, elems in new_elems_dict.items():
                if n_nodes_per_elem in shell_elems_blocks.keys():
                    shell_elems_blocks[n_nodes_per_elem] = np.vstack((shell_elems_blocks[n_nodes_per_elem], elems))
                else:
                    shell_elems_blocks[n_nodes_per_elem] = elems
        else:
            for elems in new_elems_dict.values():
                shell_elems_blocks[count_new_blocks] = elems
                count_new_blocks += 1
    shell4_elems = shell_elems_blocks.get(4, [])
    trishell_elems = shell_elems_blocks.get(3, [])
    if list(shell_elems_blocks.keys())[0] == 3:
        if len(shell4_elems):
            map_elems_idxs = np.concatenate((np.array(map_elems_idxs), np.array(
                map_elems_idxs)[np.arange(len(shell4_elems)) + len(trishell_elems)]))
            map_fside_idxs = np.concatenate((np.array(map_fside_idxs), np.array(
                map_fside_idxs)[np.arange(len(shell4_elems)) + len(trishell_elems)]))
    else:
        map_elems_idxs = np.insert(np.array(map_elems_idxs), len(shell4_elems), np.array(
            map_elems_idxs)[np.arange(len(shell4_elems))])
        map_fside_idxs = np.insert(np.array(map_fside_idxs), len(shell4_elems), np.array(
            map_fside_idxs)[np.arange(len(shell4_elems))])
    if len(shell4_elems):
        shell_elems_blocks[4] = np.vstack((shell4_elems[:, [0, 1, 2]], shell4_elems[:, [0, 2, 3]]))
    sideset_elem_idxs = {}
    sideset_fside_idxs = {}
    map_tuples = {}
    for i, t in enumerate(list(zip(map_elems_idxs, map_fside_idxs))):
        map_tuples.setdefault(t, []).append(i)
    for sideset_id, elem_idxs in exo_data.sideset_elem_idxs.items():
        sideset_elem_idxs[sideset_id] = []
        sideset_fside_idxs[sideset_id] = []
        fside_idxs = exo_data.sideset_fside_idxs[sideset_id]
        for e, f in zip(elem_idxs, fside_idxs):
            e_idxs = map_tuples[(e, f)]
            sideset_elem_idxs[sideset_id] += e_idxs
            sideset_fside_idxs[sideset_id] += [0 for _ in range(len(e_idxs))]
        #
        sideset_elem_idxs[sideset_id] = np.array(sideset_elem_idxs[sideset_id])
        sideset_fside_idxs[sideset_id] = np.array(sideset_fside_idxs[sideset_id])

    all_node_ids = None
    all_elems = None
    for i, elems in enumerate(shell_elems_blocks.values()):
        if not i:
            all_node_ids = elems.flatten()
            all_elems = elems
        else:
            all_node_ids = np.hstack((all_node_ids, elems.flatten()))
            all_elems = np.vstack((all_elems, elems))

    all_node_ids = np.sort(np.unique(all_node_ids))
    map_node_ids_dict = get_compress_node_ids_map(all_node_ids)
    coords = exo_data.coords[all_node_ids]
    elem_start_idx = 0
    block_idx = 0
    elem_type_name = 'TRISHELL'
    blk_datas = []
    new_elems = []
    for n_ids in all_elems:
        new_elems.append([map_node_ids_dict[i] for i in n_ids])
    blk_datas.append(
        ed.ExoBlockData(block_idx, elem_type_name, np.array(new_elems), elem_start_idx))
    node_field_values = {}
    for field_name, field_values in exo_data.node_field_values.items():
        node_field_values[field_name] = exo_data.node_field_values[field_name][:, all_node_ids]
    new_exo_data = ed.ExoData(coords, blk_datas, sideset_elem_idxs, sideset_fside_idxs, {},
                              np.arange(len(blk_datas)) + 1, node_field_values=node_field_values)

    # returns a shell exo data and a map of new node ids to old node ids
    return new_exo_data, {v: k for k, v in map_node_ids_dict.items()}


def get_boundary_sedges(exo_data):
    surface_exo_data, _ = get_surface_mesh(exo_data)
    sides = get_sides(surface_exo_data)
    sedges = get_sedges(surface_exo_data, sides)
    boundary_sedges = sedges[get_sedge_field(sedges, 'is_sharp')]['mesh_node_idxs']
    return boundary_sedges


if __name__ == '__main__':
    import set_akselos_path
    import copy
    import akselos.mesh.exo_data as ed
    exo_file = 'F:/Sinh/Study/learn_qt/learn_OpenGL/exo_files/rect_solution.exo'
    exo_data = ed.ExoData.read(exo_file)
    exo_data.sideset_fside_idxs[300] = np.zeros(len(exo_data.sideset_fside_idxs[300]), dtype=np.int32)
    new_exo_data, _ = get_surface_mesh(exo_data)
    sides = get_sides(new_exo_data)
    sedges = get_sedges(new_exo_data, sides)
    boundary_sedges = sedges[get_sedge_field(sedges, 'is_sharp')]['mesh_node_idxs']
    # print(boundary_sedges)
