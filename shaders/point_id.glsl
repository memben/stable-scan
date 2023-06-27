
#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
// color only for uniformity between shaders
in vec3 in_color;

out vec4 color;

uniform mat4 mvp;
uniform float point_size;

void main()
{
    float r, g, b, a;
    gl_Position = mvp * vec4(in_position, 1);
    gl_PointSize = point_size;
    int vertex_index = gl_VertexID;

    r = mod(vertex_index, 256.0) / 255.0;
    g = mod(floor(vertex_index / 256.0), 256.0) / 255.0;
    b = mod(floor(vertex_index / 65536.0), 256.0) / 255.0;
    a = floor(vertex_index / 16777216.0) / 255.0;
    color = vec4(r, g, b, a);
}

#elif defined FRAGMENT_SHADER

in vec4 color;
out vec4 f_color;

void main() {
    f_color = color;
}

#endif