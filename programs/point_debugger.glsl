#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec3 in_color;

out vec3 color;

uniform mat4 mvp;
uniform float point_size;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    gl_PointSize = point_size;
    color = in_color;
}

#elif defined FRAGMENT_SHADER

in vec3 color;

out vec4 fragColor;

void main() {
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
#endif
