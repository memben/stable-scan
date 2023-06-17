#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec3 in_color;

out vec3 color;

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;

void main() {
    mat4 MVP = m_proj * m_camera * m_model;
    gl_Position = MVP * vec4(in_position, 1.0);
    color = in_color;
}

#elif defined FRAGMENT_SHADER

in vec3 color;

out vec4 fragColor;

void main() {
    fragColor = vec4(color, 1.0);
}
#endif
