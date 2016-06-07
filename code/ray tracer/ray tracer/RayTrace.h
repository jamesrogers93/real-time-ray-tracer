#ifndef _RayTrace_h
#define _RayTrace_h

//Cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <texture_indirect_functions.h>

//RayTracer
#include "Environment.h"
#include "Ray.h"
#include "CollisionInfo.h"
#include "Intersection.h"

#define BACKGROUND_COLOUR glm::vec3(0.0f)
#define MAX_DEPTH 3

void rayTraceImage(dim3 gridSize, dim3 blockSize, cudaSurfaceObject_t image, Environment *environment);

__global__ 
void rayTrace(cudaSurfaceObject_t image, Environment *environment);

__device__
glm::vec3 trace(Environment *environment, Ray &ray, int depth);

//
//	Calculates the colour of a point.
//
__device__
glm::vec3 shade(Environment *environment, CollisionInfo &info, int depth);

//
//	Calculates the light sources impact on the point and shadows
//
__device__
glm::vec3 calculateShadowedLighting(Environment *environment, CollisionInfo &info);

//
//	Calculates the light sources impact on the point and shadows
//
__device__
glm::vec3 calculateLighting(Environment *environment, CollisionInfo &info);

//
//	Apply ambient Lighting
//
__device__
glm::vec3 applyAmbientLighting(Environment *environment, CollisionInfo &info);

//
//	Calculates the colour of a point using a shading model
//
__device__
glm::vec3 applyShadingModel(const Light &light, const CollisionInfo &info, const glm::vec3 &lightDirection, const glm::vec3 &viewDirection, const float &distance);

//
//	Calculates the reflected colour of a point
//
__device__
glm::vec3 calculateReflection(Environment *environment, CollisionInfo &info, int depth);

//
//	Calculates the refracted colour of a point
//
__device__  
glm::vec3 calculateRefraction(Environment *environment, CollisionInfo &info, int depth);

//
//	Calculates the primary ray direction
//
__device__
void primary(glm::vec3 &primary, CudaCamera *camera, int x, int y, float xOffset, float yOffset);

//
//	Calculates the reflection ray direction
//
__device__
void reflect(glm::vec3 &incident, glm::vec3 &normal, glm::vec3 &reflect, float &cosI);

//
//	Calculates the refracted ray direction
//
__device__
bool refract(glm::vec3 &incident, glm::vec3 &normal, glm::vec3 &transmitted, float &eta, float &cosI);

//
//	Schlicks Approximation
//
__device__
float schlick(float &m1, float &m2, float &eta, float &cosI);

#endif