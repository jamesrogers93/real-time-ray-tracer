//
//  Shader.hpp
//  Simple Gravity
//
//  Created by James Rogers on 28/12/2015.
//  Copyright © 2015 James Rogers. All rights reserved.
//

#ifndef Shader_hpp
#define Shader_hpp

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <GL/glew.h>

using namespace std;

class Shader
{
public:

	//Program ID
	GLuint program;

	//Constructor takes vertex shader and fragment shader path
	Shader();
	Shader(const GLchar* vertexPath, const GLchar* fragmentPath);

	//Use program
	void use();
	void loadShader(const GLchar* vertexPath, const GLchar* fragmentPath);
};

#endif /* Shader_hpp */