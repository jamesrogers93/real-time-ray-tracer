#ifndef _Intersection_h
#define _Intersection_h

//CUDA
#include <cuda_runtime.h>

//Ray Tracer
#include "Model.h"
#include "Triangle.h"
#include "Sphere.h"
#include "KDTree.h"
#include "CollisionInfo.h"

//Tools
#include "Interpolate.h"

#define HIT true;
#define NO_HIT false;
#define PI 3.141592653589793

//
//	Indicates which type of object has been hot by the ray
//
enum hitType
{
	NONE_HIT, TRI_HIT, SPHERE_HIT
};

/*
*	KDTree traversal
*/
__device__ 
bool kdSearch(Model *models, Triangle *triangles, Sphere *spheres, KDTree::KDNode *root, CollisionInfo &info);

/*
*	Ray - Voxel intersection
*/
__device__ 
bool intersectVoxel(KDTree::Voxel &voxel, Ray &ray, float &tmin, float &tmax);

/*
*	Ray - Triangle intersection (Moller Trumbore)
*/
__device__ 
bool intersectTriangle(const Ray &ray, const glm::vec3 &V0, const glm::vec3 &V0V1, const glm::vec3 &V0V2, float &t, const float tMin, const float tMax, float &u, float &v);

/*
*	Ray - Sphere intersection
*/
__device__ 
bool intersectSphere(const Ray &ray, const glm::vec3 &origin, const float &radius, float& t, const float tMin, const float tMax);

#endif