#version 330

in vec4 fragment_color;

out vec4 f_color;

void main() {
    f_color = fragment_color;
}