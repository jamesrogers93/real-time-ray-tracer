#ifndef _Interpolate_h
#define _Interpolate_h

//CUDA
#include <cuda_runtime.h>

//GLM
#include <glm\glm.hpp>

/*
*	Interpolate normals using barycentric coords
*/
__device__ 
glm::vec3 interpolateNormal(const glm::vec3 &N0, const glm::vec3 &N1, const glm::vec3 &N2, const float &u, const float &v);

/*
*	Interpolate texture coordinates using barycentric coords
*/
__device__
glm::vec2 interpolateTexCoords(const glm::vec2 &T0, const glm::vec2 &T1, const glm::vec2 &T2, const float &u, const float &v);

#endif