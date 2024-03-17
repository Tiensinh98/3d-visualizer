# Copyright (C) 2021 Akselos
import warnings

try:
    import netCDF4
except ImportError:
    class NetCDF4Dummy:
        def __init__(self, filepath, *args, **kwargs):
            pass
        def __getattr__(self, item):
            raise ValueError('netCDF4 module not installed')
    Dataset = NetCDF4Dummy
else:
    Dataset = netCDF4.Dataset

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        import h5py
except ImportError:
    class Hdf5Dummy:
        def __init__(self, filepath, *args, **kwargs):
            pass
        def __getattr__(self, item):
            raise ValueError('h5py module not installed. '
                             'Please install it by: (sudo) apt-get install python3-h5py '
                             'or pip3 install h5py'
                             )
    Hdf5File = Hdf5Dummy
else:
    Hdf5File = h5py.File

import numpy as np


# A wrapper around the python netCDF4 library that has a more convenient interface for
# our purposes.
class MyNetcdf4(Dataset):
    def __init__(self, filepath, *args, **kwargs):
        try:
            Dataset.__init__(self, filepath, *args, **kwargs)
        except:
            # For some reason, netcdf4 doesn't include the filename in its IOError, annoying.
            # Make our own error message instead.
            import user_error_message
            raise user_error_message.UserErrorMessage("Error reading file " + str(filepath))

    @staticmethod
    def fix_array_dtype(array):
        if array.dtype in [np.int64, np.uint64, np.int16, np.uint16, np.uint8, np.int8]:
            # netCDF format doesn't support any of these integer types?  Figure this out.
            dtype = np.int32
            result = np.array(array, dtype=dtype)
        # Get the smallest dtype that fits the array.
        #    mx = np.max(array)
        #    mn = np.min(array)
        #    dtype = np.min_scalar_type(mx if mn > 0 else -mx)

        else:
            result = array
        return result

    def set_attribute(self, array, attribute_name, value):
        # Sets an attribute for array, or a global attribute if array is None.
        # Note that netCDF4-python allows you to just treat the attribute as a member variable.
        # This is weird because there may already be a member variable with
        # the same name as the attribute you are trying to set.  The 'setncattr' function can be
        # used to avoid this unfortunate scenario.
        if array is None:
            self.setncattr(attribute_name, value)
        else:
            assert isinstance(array, netCDF4.Variable)
            array.setncattr(attribute_name, value)

    def set_idx_array(self, array, name, dimension_names):
        # Exo indexes start at 0, so we add 1 before writing.
        return self.set_array(array+1, name, dimension_names)

    def set_array(self, array, name, dimension_names):
        # Sets an array.  Also creates the necessary dimensions, or, if the dimensions
        # exist already, asserts that they have the correct values.
        array = self.fix_array_dtype(array)

        if isinstance(dimension_names, str):
            dimension_names = (dimension_names,)

        assert len(dimension_names) == len(array.shape)
        for dimension_name, dimension_value in zip(dimension_names, array.shape):
            if not self.has_dimension(dimension_name):
                self.createDimension(dimension_name, dimension_value)
            else:
                assert self.get_dimension(dimension_name) == dimension_value

        variable = self.createVariable(name, array.dtype, dimension_names)
        variable[:] = array
        return variable

    def set_dimension(self, dimension_name, dimension_value):
        self.createDimension(dimension_name, dimension_value)

    def has_array(self, name):
        return name in self.variables

    def get_array(self, name, dtype=None):
        # Make sure to use get_idx_array if the array contains indexes that need to be 0-based.
        # Need to use [:] to get a numpy array from the netcdf4-python array.  Calling np.array
        # on it is extremely slow.
        if self.variables[name].size == 0:
            return np.array([])
        array = self.variables[name][:]
        if dtype is not None:
            array = np.array(array, dtype=dtype)
        return array

    def get_idx_array(self, name):
        # Note that exo indexes start from 1, so we subtract 1 to get indexes
        # starting from 0.
        # Need to use [:] to get a numpy array from the netcdf4-python array.  Calling np.array
        # on it is extremely slow.
        return self.get_array(name) - 1

    def get_strings(self, name):
        return [x.tostring().split(b'\x00')[0].decode('utf-8')for x in self.get_array(name)]

    def get_attribute(self, variable_name, attribute_name):
        if variable_name is not None:
            result = self.variables[variable_name].getncattr(attribute_name)
        else:
            result = self.getncattr(attribute_name)
        return result

    def get_dimension(self, name):
        return len(self.dimensions[name])

    def has_dimension(self, name):
        return name in self.dimensions


class MyHdf5(Hdf5File):
    def __init__(self, filepath, *args, **kwargs):
        Hdf5File.__init__(self, filepath, *args, **kwargs)
        self.variables = self.keys()

    def has_array(self, name):
        has_array = self.get(name)
        return has_array is not None

    def get_array(self, name, dtype=None):
        array = self[name][:]
        if dtype is not None:
            array = np.array(array, dtype=dtype)
        return array

    def get_strings(self, name, turn_byte_to_str=True):
        strings_array = self.get_array(name)
        strings = [x.tostring() for x in strings_array]
        strings = [x if (x.find(b'\x00') == -1) else x[:x.index(b'\x00')] for x in strings]
        if turn_byte_to_str:
            strings = bytes_to_str(strings)
        return strings

    def get_dimension(self, name):
        dataset = self.get(name)
        assert dataset is not None
        if dataset.shape == (): # data is scalar
            return dataset[...]
        else:
            # Not sure why there is the case where dimension is not scalar.
            # For example, when trying to read "time_step" from the 13_component_13_NETCDF4_NDA.exo,
            # it has value as an array [0.], while we expect a scalar (i.e. 1).
            # Maybe because that mesh was converted from NETCDF3 format by netcdf-python (the
            # 13_component_13_NETCDF4_NDA.exo).
            # So here I guess the dimension's value should be the len of the array (it seems OK so
            # far but need to figure out why it is that).
            return len(dataset[:])

    def get_attribute(self, variable_name, attribute_name):
        if variable_name is not None:
            attribute_value = self[variable_name].attrs.get(attribute_name)
            assert attribute_value is not None
        else:
            attribute_value = self.attrs.get(attribute_name)
            assert attribute_value is not None

        if not isinstance(attribute_value, str):
            attribute_value = attribute_value.decode('utf-8')

        return attribute_value

    def get_idx_array(self, name):
        return self.get_array(name) - 1

    def set_idx_array(self, array, name, dimension_names):
        # Exo indexes start at 0, so we add 1 before writing.
        return self.set_array(array+1, name, dimension_names)

    def has_dimension(self, name):
        has_dimension = self.get(name)
        return has_dimension is not None

    def set_attribute(self, array, attribute_name, value):
        # Sets an attribute for array, or a global attribute if array is None.
        if array is None:
            self.attrs.create(attribute_name, value)
        else:
            assert isinstance(array, h5py.Dataset)
            array.attrs.create(attribute_name, value)

    def set_dimension(self, dimension_name, dimension_value):
        self.create_dataset(dimension_name, data=dimension_value)

    def set_array(self, array, name, dimension_names):
        # Sets an array.  Also creates the necessary dimensions, or, if the dimensions
        # exist already, asserts that they have the correct values.
        array = MyNetcdf4.fix_array_dtype(array)

        if isinstance(dimension_names, str):
            dimension_names = (dimension_names,)

        assert len(dimension_names) == len(array.shape)
        for dimension_name, dimension_value in zip(dimension_names, array.shape):
            if not self.has_dimension(dimension_name):
                self.set_dimension(dimension_name, dimension_value)
            else:
                assert self[dimension_name][...] == dimension_value

        variable = self.create_dataset(name, data=array)
        return variable


def bytes_to_str(b):
    if isinstance(b, list):
        return [_b.decode('utf-8') if isinstance(_b, bytes) else _b for _b in b]
    else:
        return b.decode('utf-8') if isinstance(b, bytes) else b


def create_netcdf4_or_netcdf3(exo_filepath, use_netcdf_python=False, mode='r', *args, **kwargs):
    # The idea to use h5py to read/write NETCDF4/HDF5 format and use netcdf-python for NETCDF3 format.

    check_details = False

    if use_netcdf_python or mode=='w':
        # Sometimes, we want to force to use netcdf-python,
        # e.g. update mesh in NETCDF3 to NETCDF4

        # For now, when writing a mesh, force to use netcdf-python because mesh created by h5py
        # seems not to be read by the current scrbe. Should test this again after scrbe is
        # updated to use netCDF 4.6.2.

        my_netcdf = MyNetcdf4(exo_filepath, mode=mode, *args, **kwargs)
        return my_netcdf

    my_netcdf4 = None
    try:
        my_netcdf4 = MyHdf5(exo_filepath, mode=mode, *args, **kwargs)
        if check_details:
            print("NETCDF4/HDF5 detected")
    except IOError:
        # invalid HDF5 file, it should be in NETCDF3 format
        pass

    # Update: Ugh, not all meshes created in NETCDF4 format by netcdf-python can be read by h5py
    # (some netcdf-python/netcdf-c versions are compatible with h5py, see #2728). Here we try to
    # catch that case and use netcdf-python to read instead of h5py.
    compatible_with_h5py = my_netcdf4 is not None
    if my_netcdf4 is not None:
        try:
            # test if the file can be read by h5py
            if mode == 'r':
                # Only test this when reading, for writing we will use h5py
                my_netcdf4.get_strings("coor_names", turn_byte_to_str=False)
            return my_netcdf4
        except IOError:
            # if cannot get_strings("coor_names"), probably the file was written by "
            # a version of netcdf-python which incompatible with h5py
            my_netcdf4.close()
            compatible_with_h5py = False

    if my_netcdf4 is None or not compatible_with_h5py:
        if check_details:
            if my_netcdf4 is None:
                print("NETCDF3 detected")
            elif not compatible_with_h5py:
                print("NETCDF4 format detected but cannot read because looks like it was written by "
                      "a version of netcdf-python which incompatible with h5py. Use netcdf-python to "
                      "read it.")
        my_netcdf = MyNetcdf4(exo_filepath, mode=mode, *args, **kwargs)
        return my_netcdf
