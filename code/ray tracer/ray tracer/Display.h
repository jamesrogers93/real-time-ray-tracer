#ifndef Display_h
#define Display_h

//STD
#include <stdio.h>
#include <algorithm>
#include <iterator>

//GLEW
#include <gl/glew.h>

//SOIL
#include <SOIL/SOIL.h>

//GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

//Includes
#include "Shader.h"

class Display
{
public:

	GLfloat vertices[32];
	GLint indices[6];
	glm::vec2 texCoords[4];

	Shader shader;

	GLuint VAO, VBO, EBO;

	//CUDA graphics resource
	GLuint viewGLTexture;

	int savedDisplayCount;

	Display(void);

	void drawDisplay(void);

	void updateDisplay(void);

	void initDisplay(int width, int height);

private:
	void initGeometry(void);
	void initVAO(void);
	void initTexture(int width, int height);
};

#endif