#ifndef _Environment_h
#define _Environment_h

//CUDA
#include <curand_kernel.h>

//Ray Tracer
#include "Model.h"
#include "Sphere.h"
#include "Light.h"
#include "Camera.h"
#include "KDTree.h"
#include "GraphicSettings.h"

struct Environment
{
	//Ray Tracer Settings
	Settings *settings, *d_settings;

	//Objects
	Model *models, *d_models;
	size_t numModels;
	size_t numModelsBytes;

	//Triangles
	//cudaTextureObject_t triangles;
	Triangle *triangles, *d_triangles;
	size_t numTriangles;
	size_t numTrianglesBytes;

	//Spheres
	//cudaTextureObject_t spheres;
	Sphere *spheres, *d_spheres;
	size_t numSpheres;
	size_t numSpheresBytes;

	//Lights
	//cudaTextureObject_t lights;
	Light *lights, *d_lights;
	size_t numLights;
	size_t numLightsBytes;

	//KDTree
	KDTree::KDNode *kdTree, *d_kdTree;
	size_t numKDTreeBytes;
	
	//Camera
	CudaCamera *camera;

	//Random numbers
	float *randomNumbers, *d_randomNumbers;
};

#endif