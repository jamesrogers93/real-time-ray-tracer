#ifndef _Material_h
#define _Material_h 

//GLEW
#include <GL/glew.h>

//CUDA
#include <cuda.h>
#include <cuda_runtime.h>

//GLM
#include <glm/glm.hpp>


struct Texture
{
	//Arrays to store pixel values
	float4 *textureArray;
	cudaArray *d_textureArray;

	//Number bytes to store texture
	size_t numtextureBytes;

	//Image dimensions
	int width, height;

	//Texture object
	cudaTextureObject_t textureObject;

	bool isLoaded;

	Texture() : textureArray(0), d_textureArray(0), numtextureBytes(0), width(0), height(0), textureObject(0), isLoaded(false)
	{
	}
};

struct Material
{
	glm::vec3 ambient;
	glm::vec3 diffuse;
	glm::vec3 specular;

	Texture diffuseMap;
	Texture specularMap;
	Texture bumpMap;

	bool reflective;
	float reflectiveness;

	bool refractive;
	float transparency;
	float refractiveIndex;

	__host__ __device__
	Material()
	{}

	Material(glm::vec3 ambient, glm::vec3 diffuse, glm::vec3 specular, float reflectiveness, float transparency, float refractiveIndex)
	{
		this->ambient = ambient;
		this->diffuse = diffuse;
		this->specular = specular;
		this->reflectiveness = reflectiveness;
		this->transparency = transparency;
		this->refractiveIndex = refractiveIndex;

		this->reflective = (reflectiveness > 0.0f);
		this->refractive = (transparency > 0.0f);
	}
};

#endif