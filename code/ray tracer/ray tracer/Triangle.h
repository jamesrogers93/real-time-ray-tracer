#ifndef _Triangle_cuh
#define _Triangle_cuh

//CUDA
#include <CUDA/cuda_runtime.h>

//GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

//RayTracer
#include "Ray.h"

struct Vertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 texCoords;

	//Bump mapping test
	glm::vec3 tangent;
	glm::vec3 binormal;
};

class Triangle
{
public:
	//The three verticies of the triangle
	Vertex V0, V1, V2;

	//Indices of object and mesh triangle belongs to
	int modelID, meshID;

	//Precomputed edges
	glm::vec3 V0V1, V0V2;

	//Bump mapping
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec3 bitangent;


	__host__ __device__ 
	Triangle(){}

	Triangle(Vertex V0, Vertex V1, Vertex V2, int modelID, int meshID)
	{
		this->V0 = V0;
		this->V1 = V1;
		this->V2 = V2;

		this->modelID = modelID;
		this->meshID = meshID;

		//Calculate vars for colliison tests
		V0V1 = V1.position - V0.position;
		V0V2 = V2.position - V0.position;
	}

};

#endif