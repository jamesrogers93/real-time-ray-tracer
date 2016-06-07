#ifndef Light_h
#define Light_h

//CUDA
#include <cuda_runtime.h>

//GLM
#include <glm\glm.hpp>

class Light
{
public:

	//Attenuation
	float linear, quadratic;

	//Position
	glm::vec3 origin;

	//Colour
	glm::vec3 ambient, diffuse, specular;

	Light();

	Light(glm::vec3 origin, glm::vec3 ambient, glm::vec3 diffuse, glm::vec3 specular, float linear, float quadratic);

};

#endif