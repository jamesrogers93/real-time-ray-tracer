#ifndef _Ray_h
#define _Ray_h

//CUDA
#include <CUDA/device_launch_parameters.h>

//GLM
#include <glm/glm.hpp>

//Ray Tracer
#include "Camera.h"

class Ray
{
public:

	glm::vec3 origin;
	glm::vec3 direction;

	__device__ 
	Ray() {}

	__device__
	Ray(glm::vec3 &origin, glm::vec3 &direction) : origin(origin), direction(direction)
	{
	}
};

#endif