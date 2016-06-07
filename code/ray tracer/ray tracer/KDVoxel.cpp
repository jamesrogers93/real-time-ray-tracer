#include "KDTree.h"

namespace KDTree
{
	Voxel::Voxel(){}

	Voxel::Voxel(Triangle *triangles, int *triangleIndices, size_t numTriangleIndices, Sphere *spheres, int *sphereIndices, size_t numSphereIndices)
	{
		initVoxelTrianglesSpheres(triangles, triangleIndices, numTriangleIndices, spheres, sphereIndices, numSphereIndices);
	}

	Voxel::Voxel(glm::vec3 &V0, glm::vec3 &V1, glm::vec3 &V2)
	{
		initVoxelVertices(V0, V1, V2);
	}

	Voxel::Voxel(glm::vec3 &origin, float &radius)
	{
		initVoxelSphere(origin, radius);
	}

	/*
	*	Initalise voxel so that min and max extents encompas triangles and spheres
	*/
	void Voxel::initVoxelTrianglesSpheres(Triangle *triangles, int *triangleIndices, size_t numTriangleIndices, Sphere *spheres, int *sphereIndices, size_t numSphereIndices)
	{
		//Initalise voxel min and max
		min = glm::vec3(INFINITY);
		max = glm::vec3(-INFINITY);

		//Initalise Triangle variables
		Triangle *triangle;
		glm::vec3 V0, V1, V2;

		//Find triangles min and max extents
		for (size_t i = 0; i < numTriangleIndices; i++)
		{
			//Get first Triangle
			triangle = &triangles[triangleIndices[i]];

			//Get vertices of triangle
			V0 = triangle->V0.position;
			V1 = triangle->V1.position;
			V2 = triangle->V2.position;

			//Loop over each dimension
			for (size_t k = 0; k < 3; k++)
			{
				//Compare extents against voxel min and max
				if (V0[k] < min[k]) min[k] = V0[k];
				if (V0[k] > max[k]) max[k] = V0[k];

				if (V1[k] < min[k]) min[k] = V1[k];
				if (V1[k] > max[k]) max[k] = V1[k];

				if (V2[k] < min[k]) min[k] = V2[k];
				if (V2[k] > max[k]) max[k] = V2[k];
			}
		}

		//Initalise sphere variables
		Sphere *sphere;
		glm::vec3 origin;
		float radius, sphereMin, sphereMax;

		//Find spheres min and max extents
		for (size_t i = 0; i < numSphereIndices; i++)
		{
			//Get sphere
			sphere = &spheres[sphereIndices[i]];

			//Get position and radius of sphere
			origin = sphere->origin;
			radius = sphere->radius;

			//Loop over each dimension
			for (size_t k = 0; k < 3; k++)
			{
				//Calcualte sphere min and max extents
				sphereMin = origin[k] - radius;
				sphereMax = origin[k] + radius;

				//Compare extents against voxel min and max
				if (sphereMin < min[k]) min[k] = sphereMin;
				if (sphereMax > max[k]) max[k] = sphereMax;
			}
		}

		//Calculate the surface area of the voxel
		calculateSA();
	}

	void Voxel::initVoxelVertices(glm::vec3 &V0, glm::vec3 &V1, glm::vec3 &V2)
	{
		//Initalise voxel min and max
		min = glm::vec3(INFINITY);
		max = glm::vec3(-INFINITY);

		//Loop over each dimension
		for (size_t k = 0; k < 3; k++)
		{
			//Compare extents against voxel min and max
			if (V0[k] < min[k]) min[k] = V0[k];
			if (V0[k] > max[k]) max[k] = V0[k];

			if (V1[k] < min[k]) min[k] = V1[k];
			if (V1[k] > max[k]) max[k] = V1[k];

			if (V2[k] < min[k]) min[k] = V2[k];
			if (V2[k] > max[k]) max[k] = V2[k];
		}

		calculateSA();
	}

	void Voxel::initVoxelSphere(glm::vec3 &origin, float &radius)
	{
		//Initalise voxel min and max
		min = glm::vec3(INFINITY);
		max = glm::vec3(-INFINITY);

		float sphereMin, sphereMax;

		//Loop over each dimension
		for (size_t k = 0; k < 3; k++)
		{
			//Calcualte sphere min and max extents
			sphereMin = origin[k] - radius;
			sphereMax = origin[k] + radius;

			//Compare extents against voxel min and max
			if (sphereMin < min[k]) min[k] = sphereMin;
			if (sphereMax > max[k]) max[k] = sphereMax;
		}

		calculateSA();
	}

	/*
	*	Calcualte the surface area of the voxel
	*/
	void Voxel::calculateSA()
	{
		float w = max.x - min.x;
		float h = max.y - min.y;
		float d = max.z - min.z;

		SA = (2 * (w * h)) + (2 * (w * d)) + (2 * (d * h));
	}

	void Voxel::clipVerticesToVoxel(glm::vec3 &V0, glm::vec3 &V1, glm::vec3 &V2, Voxel &voxel)
	{

		//Calcualte voxel of vertices
		Voxel triVoxel = Voxel(V0, V1, V2);

		//CLip triVoxel to parent voxel 
		for (int k = 0; k < 3; k++)
		{
			if (voxel.min[k] > triVoxel.min[k])
				triVoxel.min[k] = voxel.min[k];
			if (voxel.max[k] < triVoxel.max[k])
				triVoxel.max[k] = voxel.max[k];
		}

		//Set variables to triVoxels
		this->min = triVoxel.min;
		this->max = triVoxel.max;
		this->SA = triVoxel.SA;
	}

	void Voxel::clipSphereToVoxel(glm::vec3 &origin, float &radius, Voxel &voxel)
	{
		Voxel sphereVoxel = Voxel(origin, radius);

		for (int k = 0; k < 3; k++)
		{
			if (voxel.min[k] > sphereVoxel.min[k])
				sphereVoxel.min[k] = voxel.min[k];
			if (voxel.max[k] < sphereVoxel.max[k])
				sphereVoxel.max[k] = voxel.max[k];
		}

		//Set variables to triVoxels
		this->min = sphereVoxel.min;
		this->max = sphereVoxel.max;
		this->SA = sphereVoxel.SA;
	}

	bool Voxel::isPerpendicular(int &k)
	{
		return max[k] - min[k] == 0;
	}
}