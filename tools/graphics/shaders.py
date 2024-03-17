import copy

SHADED_VERTEX = """
    attribute vec3 vertex;
    attribute vec3 normal;
    
    // uniform variables
    uniform vec3 color;
    uniform mat4 eye_from_local;
    uniform mat4 ndc_from_eye;
    
    // additional outputs for vertex shader in addition to gl_Position
    varying vec4 myVertex;
    varying vec3 myNormal;
    varying vec3 myColor;
    void main()
    {
        gl_Position = ndc_from_eye * eye_from_local * vec4(vertex, 1.0f);
        myNormal = transpose(inverse(eye_from_local)) * normal;
        myVertex = eye_from_local * vec4(vertex, 1.0f);
        myColor = color;
    }
    """


SHADED_FRAGMENT = """
    varying highp vec4 myVertex;
    varying highp vec3 myNormal;
    varying highp vec3 myColor;
    
    uniform vec4 lightPosition[2];
    uniform vec4 lightColor[2]
    uniform vec4 ambient;
    uniform vec4 diffuse;
    uniform vec4 specular;
    uniform vec4 emission;
    uniform float shininess;
    
    vec4 computeLight(const in vec3 direction, const in vec4 lightcolor, const in vec3 normal, const in vec3 halfvec, const in vec4 mydiffuse, const in vec4 myspecular, const in float myshininess)
    {
        float nDotL = dot(normal, direction);
        vec4 lambert = mydiffuse * lightcolor * max(nDotL, 0.0f);
        float nDotH = dot(normal, halfvec);
        vec4 phong = myspecular * lightcolor * pow(max(nDotH, 0.0f), myshininess);
        vec4 retval = lambert + phong;
        return retval;
    }
    
    void main(void)
    {
        vec4 finalColor;
        const vec3 eyePos = vec3(0.0f, 0.0f, 0.0f);
        vec3 myPos = myVertex.xyz / myVertex.w;
        vec3 eyeDirn = normalize(eyePos - myPos);
        vec3 normal = normalize(myNormal);
        vec3 direction = vec3(0.0f, 0.0f, 0.0f);
        vec3 half0 = vec3(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < 2; i++)
        {
            if (lightPosition[i]/w == 0.0)
            {
            // directional light
                direction = normalize(vec3(lightPosition[i].xyz));
                half0 = normalize(direction + eyeDirn);
            }
            else
            {
                vec3 position = lightPosition[i].xyz / lightPosition[i].w;
                direction = normalize(position - myPos);
                half0 = normalize(direction + eyeDirn);
            }
            finalColor = finalColor + computeLight(direction, lightColor[i], normal, half0, diffuse, specular, shininess);
        }
        gl_FragColor = ambient + emission + finalColor + vec4(myColor, 0.0f);
    }
    """


SCALAR_FIELD_VERTEX = """
    #version 120
    attribute vec3 vertex;
    attribute vec3 normal;
    attribute float scalar_value;
    uniform mat4 eye_from_local;
    uniform mat4 ndc_from_eye;
    varying float frag_scalar_value;
    void main() {
        gl_Position = ndc_from_eye * eye_from_local * vec4(vertex, 1.0);         
        frag_scalar_value = scalar_value;
    }
"""


SCALAR_FIELD_FRAGMENT = """
    #version 120
    const vec4 low_color = vec4(0.0, 0.0, 0.7, 1.0);
    const vec4 high_color = vec4(0.5, 0.0, 0.0, 1.0);
    const int number_of_bins = 14;
    uniform vec3 color_range[number_of_bins];
    uniform float color_values[number_of_bins + 1];
    uniform float min_value;
    uniform float max_value;
    uniform bool is_discrete_colors = false;
    varying float frag_scalar_value;
    void main() {
        vec4 mapped_color = vec4(0.0, 0.0, 0.0, 1.0f);
        if(frag_scalar_value < min_value) {
            mapped_color = low_color;
        }
        else if(frag_scalar_value > max_value) {
            mapped_color = high_color;
        }
        else {
            int bin_index = 0;
            for(int i=0; i < number_of_bins; i++) {
                if (frag_scalar_value <= color_values[i+1]) {
                    bin_index = i;
                    break;
                }
            }
            if(is_discrete_colors) {
                mapped_color = vec4(color_range[bin_index], 1);
            }
            else {
                float bin_size = color_values[bin_index+1] - color_values[bin_index];
                float bin_center = color_values[bin_index] + 0.5 * bin_size;
                float interpolator = 0.0;
                int other_bin_index = bin_index;
                if(frag_scalar_value > bin_center && bin_index < number_of_bins - 1) {
                    interpolator = (frag_scalar_value - bin_center) / bin_size;
                    other_bin_index = bin_index + 1;
                }
                else if(frag_scalar_value <= bin_center && bin_index > 0) {
                    interpolator = (bin_center - frag_scalar_value) / bin_size;
                    other_bin_index = bin_index - 1;
                }
                mapped_color = vec4(mix(color_range[bin_index], color_range[other_bin_index], interpolator), 1);
            }
        }
        if(mapped_color.w < 0.5) {
            discard;
        }
        gl_FragColor = mapped_color;
    }
"""


FLAT_VERTEX = """
    #version 120
    attribute vec3 vertex;
    varying vec4 vertex_color;
    uniform mat4 eye_from_local;
    uniform mat4 ndc_from_eye;
    uniform vec3 color;
    void main() {
        gl_Position = ndc_from_eye * eye_from_local * vec4(vertex, 1.0);         
        vertex_color = vec4(color, 1.0);
    }
"""


FLAT_FRAGMENT = """
    #version 120
    varying vec4 vertex_color;
    void main() {
        gl_FragColor = vertex_color;
    }
"""

VERTEX_LOCATION = 0
NORMAL_LOCATION = 1
COLOR_LOCATION = 2
SCALAR_VALUE_LOCATION = 3


class AttribInfo:
    def __init__(self, location_idx: int, n_components: int):
        self.location_idx = location_idx
        self.n_components = n_components


class ShaderType:
    def __init__(self, s_vert, s_frag, attrib_names):
        self.s_vert = s_vert
        self.s_frag = s_frag
        self.attrib_names = attrib_names


ATTRIB_INFOs = {
    "normal": AttribInfo(NORMAL_LOCATION, 3),
    "color": AttribInfo(COLOR_LOCATION, 3),
    "scalar_value": AttribInfo(SCALAR_VALUE_LOCATION, 1)
}


SCALAR_FIELD_SHADER_TYPE = ShaderType(
    SCALAR_FIELD_VERTEX, SCALAR_FIELD_FRAGMENT, ['scalar_value'])

FLAT_SHADER_TYPE = ShaderType(
    FLAT_VERTEX, FLAT_FRAGMENT, [])

PICK_SHADER_TYPE = copy.deepcopy(FLAT_SHADER_TYPE)

SHADED_SHADER_TYPE = ShaderType(
    SHADED_VERTEX, SHADED_FRAGMENT, ['normal'])

SHADER_TYPES = [SCALAR_FIELD_SHADER_TYPE, FLAT_SHADER_TYPE, PICK_SHADER_TYPE]
