//
//  Shader.cpp
//  Simple Gravity
//
//  Created by James Rogers on 28/12/2015.
//  Copyright © 2015 James Rogers. All rights reserved.
//

#include "Shader.h"

Shader::Shader()
{
	this->program = 0;
}

Shader::Shader(const GLchar* vertexPath, const GLchar* fragmentPath)
{
	this->loadShader(vertexPath, fragmentPath);
}

void Shader::use()
{
	glUseProgram(this->program);
}

void Shader::loadShader(const GLchar* vertexPath, const GLchar* fragmentPath)
{
	string vertexCode, fragmentCode;

	ifstream vShaderFile, fShaderFile;

	//For throwing exceptions
	vShaderFile.exceptions(ifstream::badbit);
	fShaderFile.exceptions(ifstream::badbit);

	try
	{

		//Open the files.
		vShaderFile.open(vertexPath);
		fShaderFile.open(fragmentPath);

		stringstream vShaderStream, fShaderStream;

		//Read buffer contents in to streams.
		vShaderStream << vShaderFile.rdbuf();
		fShaderStream << fShaderFile.rdbuf();

		//Close files.
		vShaderFile.close();
		fShaderFile.close();

		//Convert stream to
		vertexCode = vShaderStream.str();
		fragmentCode = fShaderStream.str();
	}
	catch (ifstream::failure e)
	{
		cout << "ERROR, shader file could not be read" << endl;
	}

	//Copy vertex and fragment code to GLchar*'s
	const GLchar* vShaderCode = vertexCode.c_str();
	const GLchar* fShaderCode = fragmentCode.c_str();

	//Vertex, Fragment shader pointers
	GLuint vertex, fragment;

	//For checking if shader loaded correctly and storing error message
	GLint success;
	GLchar infoLog[512];



	//Vertex Shader
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);

	//Check for errors
	glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertex, 512, NULL, infoLog);
		cout << "ERROR, Vertex shader compilation failed " << infoLog << endl;
	}



	//Fragment Shader
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);

	//Check for errors
	glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragment, 512, NULL, infoLog);
		cout << "ERROR. Fragment shader compilation failed " << infoLog << endl;
	}



	//Link Shaders to program
	this->program = glCreateProgram();
	glAttachShader(this->program, vertex);
	glAttachShader(this->program, fragment);
	glLinkProgram(this->program);

	//Check for errors
	glGetProgramiv(this->program, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(this->program, 512, NULL, infoLog);
		cout << "ERROR, Shader program compilation failed " << infoLog << endl;
	}



	//Now shaders are loaded into program, delete them
	glDeleteShader(vertex);
	glDeleteShader(fragment);
}