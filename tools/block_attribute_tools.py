# Copyright (C) 2021 Akselos
'''
Classes used to store mesh block attributes such as material, thickness, etc.
and then will be read by the Editor code to generate appropriate fields in JSON.
'''

import numpy as np

try:
    # If users run this within a script not with the GUI, dimensional_value cannot be imported, so
    # just ignore that case for now.
    import dimensional_value as dv
except ImportError:
    pass


# Class that stores thickness, material and element type info gotten from mesh
class BlockAttribute:
    def __init__(self, material, thickness, original_element_type, beam_properties,
                 property_id=None):
        self.material = material
        self.thickness = thickness
        self.original_element_type = original_element_type
        self.beam_properties = beam_properties
        self.property_id = property_id

    @staticmethod
    def create(material, thickness, original_element_type=None, beam_properties=None,
               property_id=None):
        return BlockAttribute(
            material, thickness, original_element_type, beam_properties, property_id)

    def set_material(self, material):
        self.material = material

    def set_thickness(self, thickness):
        self.thickness = thickness

    def set_beam_properties(self, beam_properties):
        self.beam_properties = beam_properties

    def set_property_id(self, property_id):
        self.property_id = property_id

    def get_non_dim_values(self, collection_type):
        new_material = self.material.get_non_dim_values(collection_type)
        new_thickness = None
        if self.thickness is not None:
            new_thickness = dv.decode_param(
                collection_type, "", self.thickness, type="mesh_length")
        new_beam_properties = None
        if self.beam_properties is not None:
            new_beam_properties = {}
            for scalar_name, value in self.beam_properties.items():
                non_dim_value = dv.decode_param(
                    collection_type, "", value, type=scalar_name)
                new_beam_properties[scalar_name] = non_dim_value

        return BlockAttribute(
            new_material, new_thickness, self.original_element_type, new_beam_properties,
            self.property_id)

    @staticmethod
    def write_material_data_to_mesh_block(material_data, n_elem, block_thickness=None,
                                          write_mass_density=False):
        elem_field_values = {}

        if isinstance(material_data, ThermoElasticityMaterial):
            BlockAttribute.assign_value_to_key(
                n_elem, material_data.young_modulus, "young_modulus", elem_field_values)
            BlockAttribute.assign_value_to_key(
                n_elem, material_data.poisson_ratio, "poisson_ratio", elem_field_values)
            BlockAttribute.assign_value_to_key(
                n_elem, material_data.thermal_expansion, "thermal_expansion", elem_field_values)
            BlockAttribute.assign_value_to_key(
                n_elem, 1.0, "thermal_conductivity", elem_field_values)

        elif isinstance(material_data, ElasticityMaterial):
            BlockAttribute.assign_value_to_key(
                n_elem, material_data.young_modulus, "young_modulus", elem_field_values)
            BlockAttribute.assign_value_to_key(
                n_elem, material_data.poisson_ratio, "poisson_ratio", elem_field_values)

        elif isinstance(material_data, AnisotropicThermoElasticityMaterial):
            for idx in range(21):
                coefficient = material_data.stiffness_coefficients[idx]
                BlockAttribute.assign_value_to_key(
                    n_elem, coefficient, "stiffness_value_"+str(idx), elem_field_values)

            name_map = {0: "x", 1: "y", 2: "z"}
            for idx in range(3):
                coefficient = material_data.thermal_coefficients[idx]
                BlockAttribute.assign_value_to_key(
                    n_elem, coefficient, "thermal_expansion_"+name_map[idx], elem_field_values)

            BlockAttribute.assign_value_to_key(
                    n_elem, 1.0, "thermal_conductivity", elem_field_values)

        if block_thickness is not None:
            BlockAttribute.assign_value_to_key(
                n_elem, block_thickness, "thickness", elem_field_values)

        if write_mass_density:
            if hasattr(material_data, "mass_density"):
                BlockAttribute.assign_value_to_key(
                    n_elem, material_data.mass_density, "mass_density", elem_field_values)

        return elem_field_values

    @staticmethod
    def write_beam_properties_to_mesh_block(beam_properties, n_elem, prop_list):
        elem_field_values = {}

        for property_name in beam_properties.keys():
            if property_name in prop_list:
                BlockAttribute.assign_value_to_key(
                    n_elem, beam_properties[property_name], property_name, elem_field_values)

        return elem_field_values

    @staticmethod
    def assign_value_to_key(n_elem, value, key, elem_field_values):
        if isinstance(value, float):
            elem_values = value*np.ones(n_elem)
            elem_field_values[key] = np.array([elem_values.astype(np.float32)])
        elif isinstance(value, int) or isinstance(value, np.integer):
            elem_values = value*np.ones(n_elem)
            elem_field_values[key] = np.array([elem_values.astype(np.int32)])
        else:
            print(value, type(value))
            assert False, key

    @staticmethod
    def create_block_attribute_from_main_operator(exo_data, main_operators, subdomain_name_to_id,
                                                  mu_coeff, is_beam=False):
        block_idx_to_subdomain_id = exo_data.block_idx_to_subdomain_id
        max_subdomain_id = np.max(block_idx_to_subdomain_id)
        subdomain_id_to_block_idx = -1*np.ones(max_subdomain_id+1, dtype=np.int)
        subdomain_id_to_block_idx[block_idx_to_subdomain_id] = \
            np.arange(len(block_idx_to_subdomain_id))

        for main_operator in main_operators:
            subdomain_names = main_operator.subdomains
            scalars = main_operator.scalars

            for scalar_name, scalar_value in scalars.items():
                if scalar_value.type == "literal":
                    value = float(scalar_value.value)
                elif scalar_value.type == "parameter":
                    parameter_name = scalar_value.value
                    value = float(mu_coeff[parameter_name].value)
                else:
                    assert False, scalar_value.type
                for subdomain_name in subdomain_names:
                    if not is_beam:
                        block_id = subdomain_name_to_id[subdomain_name]
                        block_idx = subdomain_id_to_block_idx[block_id]
                    else:
                        # For beam, the mesh viz has only one block (see
                        # exo_data_tools.create_1d_exo_data).
                        # So no matter the subdomain ID is in the JSON (which refers to
                        # cross-section mesh and can have multiple subdomains), we should write
                        # the attribute value to the single 1D beam block.
                        block_idx = 0

                    exo_block_data = exo_data.exo_block_datas[block_idx]
                    elem_field_values = exo_block_data.elem_field_values
                    n_elem = len(exo_block_data.elems)
                    BlockAttribute.assign_value_to_key(n_elem, value, scalar_name, elem_field_values)
        return exo_data

    @staticmethod
    def scale_existing_elem_block_field_value(exo_data, block_idxs, field_names, scaling_value):
        for block_idx in block_idxs:
            exo_block_data = exo_data.exo_block_datas[block_idx]
            elem_field_values = exo_block_data.elem_field_values
            for field_name in field_names:
                if field_name not in elem_field_values:
                    continue
                elem_field_values[field_name] = scaling_value * elem_field_values[field_name]
        return exo_data


# Classic material (usually got from normal .inp mesh)
class Material:
    __slots__ = 'name'

    def __init__(self, name):
        self.name = name

    @staticmethod
    def create(*args, **kwargs):
        pass

    def get_properties(*args, **kwargs):
        pass

    def get_property_dict(self):
        property_dict = {}
        for att in self.__class__.__slots__:
            property_dict[att] = getattr(self, att)
        return property_dict

    def get_non_dim_values(self, collection_type):
        pass


class ElasticityMaterial(Material):
    __slots__ = 'name', 'young_modulus', 'poisson_ratio', 'mass_density'

    def __init__(self, name, young_modulus, poisson_ratio, mass_density):
        Material.__init__(self, name)
        self.young_modulus = young_modulus # supposed to be Pa
        self.poisson_ratio = poisson_ratio
        self.mass_density = mass_density

    @staticmethod
    def create(name, young_modulus, poisson_ratio, mass_density):
        return ElasticityMaterial(name, young_modulus, poisson_ratio, mass_density)

    def get_properties(self):
        return self.name, self.young_modulus, self.poisson_ratio, self.mass_density

    def get_non_dim_values(self, collection_type):
        new_young_modulus = dv.decode_param(
                    collection_type, "", self.young_modulus/1.e9, type="young_modulus")
        new_mass_density = dv.decode_param(
                    collection_type, "", self.mass_density, type="mass_density")
        return ElasticityMaterial(self.name, new_young_modulus, self.poisson_ratio, new_mass_density)


# An example of this is material stored in Nastran mesh with material type MAT1
class ThermoElasticityMaterial(ElasticityMaterial):
    __slots__ = 'name', 'young_modulus', 'poisson_ratio', 'thermal_expansion', 'mass_density'

    def __init__(self, name, young_modulus, poisson_ratio, thermal_expansion, mass_density):
        ElasticityMaterial.__init__(self, name, young_modulus, poisson_ratio, mass_density)
        self.thermal_expansion = thermal_expansion

    @staticmethod
    def create(name, young_modulus, poisson_ratio, thermal_expansion, mass_density):
        return ThermoElasticityMaterial(
            name, young_modulus, poisson_ratio, thermal_expansion, mass_density)

    def get_properties(self):
        return self.name, self.young_modulus, self.poisson_ratio, \
               self.thermal_expansion, self.mass_density

    def get_non_dim_values(self, collection_type):
        new_young_modulus = dv.decode_param(
                    collection_type, "", self.young_modulus/1.e9, type="young_modulus")
        new_thermal_expansion = dv.decode_param(
                    collection_type, "", self.thermal_expansion, type="thermal_expansion")
        new_mass_density = dv.decode_param(
                    collection_type, "", self.mass_density, type="mass_density")
        return ThermoElasticityMaterial(
            self.name, new_young_modulus, self.poisson_ratio, new_thermal_expansion,
            new_mass_density)


# An example of this is material stored in Nastran mesh with material type MAT9
class AnisotropicThermoElasticityMaterial(Material):
    __slots__ = 'name', 'stiffness_coefficients', 'mass_density', 'thermal_coefficients'

    def __init__(self, name, stiffness_coefficients, mass_density, thermal_coefficients):
        Material.__init__(self, name)
        self.stiffness_coefficients = stiffness_coefficients # 21 coefficients
        self.mass_density = mass_density
        self.thermal_coefficients = thermal_coefficients # 3 coefficients

    @staticmethod
    def create(name, stiffness_coefficients, mass_density, thermal_coefficients):
        return AnisotropicThermoElasticityMaterial(
            name, stiffness_coefficients, mass_density, thermal_coefficients)

    def get_properties(self):
        return self.name, self.stiffness_coefficients, self.mass_density, \
               self.thermal_coefficients

    def get_non_dim_values(self, collection_type):
        raise NotImplemented
