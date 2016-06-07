#ifndef _Mesh_cuh
#define _Mesh_cuh

//GLEW
#include <GL/glew.h>

//CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

//GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

//ASSIMP
#include <assimp/Importer.hpp>

//STD
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

//Raytracer
#include "Material.h"
#include "Triangle.h"

//Tools
#include "Debug.h"

class Mesh
{
public:

	Material material;

	Vertex *vertices;
	size_t numVertices;
	size_t numVerticesBytes;

	int *indices;
	size_t numIndices;
	size_t numIndicesBytes;

	Mesh();
	Mesh(Vertex *vertices, size_t numVertices, size_t numVerticesBytes, int *indices, size_t numIndices, size_t numIndicesBytes, Material material);

	void deleteMem();
};

#endif