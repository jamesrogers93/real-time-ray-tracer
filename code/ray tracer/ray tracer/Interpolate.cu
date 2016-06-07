#include "Interpolate.h"

/*
*	Interpolate normals using barycentric coords
*/
__device__ 
glm::vec3 interpolateNormal(const glm::vec3 &N0, const glm::vec3 &N1, const glm::vec3 &N2, const float &u, const float &v)
{
	return glm::normalize(N0 + u * (N1 - N0) + v * (N2 - N0));
}

/*
*	Interpolate texture coordinates using barycentric coords
*/
__device__ 
glm::vec2 interpolateTexCoords(const glm::vec2 &T0, const glm::vec2 &T1, const glm::vec2 &T2, const float &u, const float &v)
{
	return T0 + u * (T1 - T0) + v * (T2 - T0);
}