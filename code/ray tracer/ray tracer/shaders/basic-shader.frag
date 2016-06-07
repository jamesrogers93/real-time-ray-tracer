#version 330 core

in vec2 TexCoord;

out vec4 colour;

uniform sampler2D displayTexture;

void main()
{
    colour = texture(displayTexture, TexCoord);
	//colour = vec4(1.0f, 0.0f, 0.0f, 1.0f);
}