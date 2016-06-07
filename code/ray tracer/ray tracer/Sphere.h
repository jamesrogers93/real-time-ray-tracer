#ifndef Sphere_cuh
#define Sphere_cuh

//CUDA
#include <cuda_runtime.h>

//GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

//RayTracer
#include "Material.h"

class Sphere
{
public:

	glm::vec3 origin;
	float radius, radius2;

	Material material;

	glm::mat4 translation;
	glm::mat4 modelMatrix;

	__host__ __device__ 
	Sphere(){}

	Sphere(glm::vec3 origin, float radius, Material material);

	void translate(glm::vec3 position, float dt = 1.0f);
};

#endif