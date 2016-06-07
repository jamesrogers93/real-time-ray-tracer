#ifndef _Collision_cuh
#define _Collision_cuh

//CUDA
#include <CUDA/cuda_runtime.h>
#include <CUDA/device_launch_parameters.h>

//Raytracer
#include "Ray.h"
#include "Triangle.h"
#include "Sphere.h"
#include "Model.h"
#include "Mesh.h"

struct CollisionInfo
{
	Ray ray;
	glm::vec3 point, normal;

	glm::vec2 texCoords;

	Material *material;

	float u, v;

	float t, tMin, tMax;
	float cosI;
	float eta, m1, m2;

	__device__ 
	CollisionInfo(){}

};

#endif