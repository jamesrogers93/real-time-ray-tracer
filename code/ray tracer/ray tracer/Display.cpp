//
//  Display.cpp
//  Basic Ray-Tracer
//
//  Created by James Rogers on 05/01/2016.
//  Copyright © 2016 James Rogers. All rights reserved.
//

#include "Display.h"

Display::Display(void): savedDisplayCount(0)
{

}

void Display::initDisplay(int width, int height)
{
	shader = Shader("shaders/basic-shader.vert", "shaders/basic-shader.frag");

	initGeometry();

	initVAO();

	initTexture(width, height);
}

void Display::initGeometry(void)
{
	//Vertex positons and texCoords
	//Top Right
	vertices[0] = 1.0f; vertices[1] = 1.0f; vertices[2] = 0.0f; //Vertices
	vertices[3] = 1.0f; vertices[4] = 1.0f;                     //TexCoords

	//Bottom Right
	vertices[5] = 1.0f; vertices[6] = -1.0f; vertices[7] = 0.0f; //Vertices
	vertices[8] = 1.0f; vertices[9] = 0.0f;                     //Texcoords

	//Bottom Left
	vertices[10] = -1.0f; vertices[11] = -1.0f; vertices[12] = 0.0f; //Vertices
	vertices[13] = 0.0f; vertices[14] = 0.0f;                     //TexCoords

	//Top Left
	vertices[15] = -1.0f; vertices[16] = 1.0f; vertices[17] = 0.0f; //Vertices
	vertices[18] = 0.0f; vertices[19] = 1.0f;                     //Texcoords

	//Create vertex indices
	indices[0] = 0; indices[1] = 1; indices[2] = 3;
	indices[3] = 1; indices[4] = 2; indices[5] = 3;
}

void Display::initVAO(void)
{

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	// TexCoord attribute
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
}

void Display::initTexture(int width, int height)
{
	//Init OpenGL texture
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &viewGLTexture);
	glBindTexture(GL_TEXTURE_2D, viewGLTexture);
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
}

void Display::drawDisplay(void)
{
	shader.use();

	//Bind texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, viewGLTexture);
	glUniform1i(glGetUniformLocation(shader.program, "displayTexture"), 0);

	//Draw display
	glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}