#include "Mesh.h"

Mesh::Mesh() : numVertices(0), numVerticesBytes(0), numIndices(0), numIndicesBytes(0)
{

}

Mesh::Mesh(Vertex *vertices, size_t numVertices, size_t numVerticesBytes, int *indices, size_t numIndices, size_t numIndicesBytes, Material material)
{
	//VERTICES
	this->vertices = vertices;
	this->numVertices = numVertices;
	this->numVerticesBytes = numVerticesBytes;

	//INDICES
	this->indices = indices;
	this->numIndices = numIndices;
	this->numIndicesBytes = numIndicesBytes;

	//Material
	this->material = material;
}

void Mesh::deleteMem()
{
	if (vertices != 0)
		free(vertices);

	if (indices != 0)
		free(indices);
}