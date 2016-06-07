#include "Ray.h"

/*
__device__
Ray::Ray(){}

__device__
void Ray::genPrimaryRay(CudaCamera *camera, int x, int y)
{
	//Ray origin
	origin = camera->origin;

	//Ray direction
	//Normalize pixel from 0 : screenWidth to 0 : 1
	//The addition of 0.5f defines the center of each pixel
	float xPix = (x + 0.5f) * camera->invWidth;
	float yPix = (y + 0.5f) * camera->invHeight;

	//Transform pixels from 0 : 1 to -1 : 1
	xPix = (2 * xPix - 1) * camera->aspectRatio * camera->scale;
	yPix = (1 - 2 * yPix) * camera->scale;

	//Calculate direction vector
	direction = glm::normalize(glm::vec3(xPix, yPix, 1) - origin);

	float4 data;
	//getCameraData(camera, data);
	getCameraData();
}*/