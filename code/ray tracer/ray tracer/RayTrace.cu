#include "RayTrace.h"

void rayTraceImage(dim3 gridSize, dim3 blockSize, cudaSurfaceObject_t image, Environment *environment)
{
	//test <<< gridSize, blockSize >>>();
	rayTrace << < gridSize, blockSize >> >(image, environment);
}

__global__ 
void rayTrace(cudaSurfaceObject_t image, Environment *environment)
{
	//Get index
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	glm::vec3 colour;

	//Generate Primary Ray
	Ray P;
	P.origin = environment->camera->origin;

	//No antialiasing
	//Trace 1 ray through the center of the pixel
	if (environment->d_settings->antiAliasing == NONE)
	{
		//Calculate direction
		primary(P.direction, environment->camera, x, y, 0.5f, 0.5f);

		//Trace scene
		colour = trace(environment, P, 0);
	}

	//Super sampling
	//Trace 4 rays through the of the pixel
	else if (environment->d_settings->antiAliasing == SUPERSAMPLE)
	{
		//Calculate direction and trace scene for each ray

		//Ray 1
		primary(P.direction, environment->camera, x, y, 0.25f, 0.25f);
		glm::vec3 colour1 = trace(environment, P, 0);

		//Ray 2
		primary(P.direction, environment->camera, x, y, 0.75f, 0.25f);
		glm::vec3 colour2 = trace(environment, P, 0);

		//Ray 3
		primary(P.direction, environment->camera, x, y, 0.25f, 0.75f);
		glm::vec3 colour3 = trace(environment, P, 0);

		//Ray 4
		primary(P.direction, environment->camera, x, y, 0.75f, 0.75f);
		glm::vec3 colour4 = trace(environment, P, 0);

		//Compute avarage of the 4 colours
		colour = (colour1 + colour2 + colour3 + colour4) * 0.25f;
	}
	else if (environment->d_settings->antiAliasing == STOCHASTIC)
	{
		//Get random offsets
		float randOffsetX = environment->d_randomNumbers[x * y + x];
		float randOffsetY = environment->d_randomNumbers[(x * y + x) * 2];

		//Calculate direction and trace scene
		primary(P.direction, environment->camera, x, y, randOffsetX, randOffsetY);
		glm::vec3 colour1 = trace(environment, P, 0);

		//Get random offsets
		randOffsetX = environment->d_randomNumbers[(x * y + x) * 3];
		randOffsetY = environment->d_randomNumbers[(x * y + x) * 4];

		//Calculate direction and trace scene
		primary(P.direction, environment->camera, x, y, randOffsetX, randOffsetY);
		glm::vec3 colour2 = trace(environment, P, 0);

		//Compute avarage of the 2 colours
		colour = (colour1 + colour2) * 0.5f;
	}

	//float4 colour = tex2D<float4>(environment->texObj, 1.0, 1.0);

	float4 pixelColour = make_float4(colour.x, colour.y, colour.z, 1.0f);
	//Write pixel colour to texture
	surf2Dwrite(pixelColour, image, x*sizeof(pixelColour), y, cudaBoundaryModeClamp);
}

__device__
glm::vec3 trace(Environment *environment, Ray &ray, int depth)
{
	//Initalise colour
	glm::vec3 colour;

	// Check recursion level
	if (depth + 1 > MAX_DEPTH)
		return colour;
	depth++;

	//Initalise Collision information
	CollisionInfo info;
	info.ray = ray;
	info.tMin = 0.0001f; //Increase tmin to prevent ray from intersecting the same object
	info.tMax = 1000.0f;

	if (kdSearch(environment->d_models, environment->d_triangles, environment->d_spheres, environment->d_kdTree, info))
	{
		colour = shade(environment, info, depth);
	}
	else
	{
		colour = BACKGROUND_COLOUR;
	}

	return glm::clamp(colour, 0.0f, 1.0f);
}

//
//	Calculates the colour of a point.
//
__device__
glm::vec3 shade(Environment *environment, CollisionInfo &info, int depth)
{
	glm::vec3 lighting, reflect, refract;

	//Find the angle between the incident ray and the surface normal
	info.cosI = -glm::dot(info.ray.direction, info.normal);

	//Indicates if ray is behind normal
	bool front = (info.cosI > 0.0f);
	//if (!front)
	//	info.normal = -info.normal;

	//Indicates if reflections and refractions are to be used
	bool reflections = (environment->d_settings->reflections && info.material->reflective);
	bool refractions = (environment->d_settings->refractions && info.material->refractive);

	if (reflections || refractions)
	{
		//If the ray is behind the normal, we are inside of the
		//object, therefore we need to swap the refractive indexes
		if (front)
		{
			info.m1 = 1.0f;
			info.m2 = info.material->refractiveIndex;
		}
		else
		{
			info.m1 = info.material->refractiveIndex;
			info.m2 = 1.0f;
		}
		info.eta = info.m1 / info.m2;

	}

	//
	//	Calculates the light sources impact on the point
	//

	lighting = applyAmbientLighting(environment, info);
	if (front)
	{
		//If shadows are enabled, Calculate shadowed lighting
		if (environment->d_settings->shadows)
			lighting += calculateShadowedLighting(environment, info);
		else
			lighting += calculateLighting(environment, info);
	}
	//else
	//{
	//	lighting = applyAmbientLighting(environment, info);
	//}

	//
	//	Calculates the reflection colours on the point
	//
	if (reflections)
	{
		//Get the reflected colour
		reflect = calculateReflection(environment, info, depth);
	}

	//
	//	Calculate the refracted colours of the point
	//
	if (refractions)
	{
		//Get refracted colour
		refract = calculateRefraction(environment, info, depth);
	}

	//
	//	Apply schlicks approximation
	if (reflections && refractions)
	{
		//Calculate the strength of the reflected and refraced colours using schlicks approximation of the fresnel equation
		float fresnel = schlick(info.m1, info.m2, info.eta, info.cosI);

		//Multiply the reflected colour by schlicks approximation
		reflect *= fresnel;

		//Multiply the refracted colour by the (refracted version) of schlicks approximation
		refract *= (1 - fresnel);
	}


	return lighting + reflect + refract;
}

//
//	Calculates the light sources impact on the point
//
__device__
glm::vec3 calculateLighting(Environment *environment, CollisionInfo &info)
{
	//Initalise colour to ambient lighting
	glm::vec3 colour;// = info.material->ambient;

	//Calculate viewing direction
	glm::vec3 viewDirection = glm::normalize(environment->camera->origin - info.point);

	//Loop each light source
	for (size_t i = 0; i < environment->numLights; i++)
	{
		//Get light
		Light light = environment->d_lights[i];

		//Calcualte direction from intersection point to light
		glm::vec3 L = glm::normalize(light.origin - info.point);

		//Generate Light Ray
		Ray I = Ray(info.point, L);

		//Calculate distance from point to light
		float distance = glm::distance(info.point, light.origin);

		//Calculate colour of lights inmpact on the light source
		colour += applyShadingModel(light, info, I.direction, viewDirection, distance);
	}
	return colour;
}

//
//	Calculates the light sources impact on the point and shadows
//
__device__
glm::vec3 calculateShadowedLighting(Environment *environment, CollisionInfo &info)
{
	//Initalise colour to ambient lighting
	glm::vec3 colour;// = info.material->ambient;

	//Calculate viewing direction
	glm::vec3 viewDirection = glm::normalize(environment->camera->origin - info.point);

	//Loop each light source
	for (size_t i = 0; i < environment->numLights; i++)
	{
		//Get light
		Light light = environment->d_lights[i];

		//Calcualte direction from intersection point to light
		glm::vec3 L = glm::normalize(light.origin - info.point);

		//Generate Light Ray
		Ray I = Ray(info.point, L);

		//Calculate distance from point to light
		float distance = glm::distance(info.point, light.origin);

		//Search for objects blocking light
		//Prepare new collision info
		CollisionInfo info2;
		info2.ray = I;
		info2.tMin = 0.0001f;
		info2.tMax = distance;

		if (!kdSearch(environment->d_models, environment->d_triangles, environment->d_spheres, environment->d_kdTree, info2))
		{
			//Calculate colour of lights impact on the light source
			colour += applyShadingModel(light, info, I.direction, viewDirection, distance);
		}
	}
	return colour;
}

//
//	Apply ambient Lighting
//
__device__
glm::vec3 applyAmbientLighting(Environment *environment, CollisionInfo &info)
{
	glm::vec3 colour, ambient;

	//Get ambient colour
	if (info.material->diffuseMap.isLoaded)
	{
		float4 diffTex = tex2D<float4>(info.material->diffuseMap.textureObject, info.texCoords.x, info.texCoords.y);
		ambient = glm::vec3(diffTex.x, diffTex.y, diffTex.z);
	}
	else
		ambient = info.material->ambient;

	//Loop over all lights and calculate mbient lighting
	for (size_t i = 0; i < environment->numLights; i++)
	{
		//Multiply by the lights ambient and apply to the colour
		colour += ambient * environment->d_lights->ambient;
	}

	return colour;
}

//
//	Calculates the colour of a point using a shading model
//
__device__
glm::vec3 applyShadingModel(const Light &light, const CollisionInfo &info, const glm::vec3 &lightDirection, const glm::vec3 &viewDirection, const float &distance)
{
	// Diffuse
	float diff = fmaxf(glm::dot(info.normal, lightDirection), 0.0f);

	// Specular
	glm::vec3 halfwayvector = glm::normalize(lightDirection + viewDirection);
	float spec = powf(fmaxf(glm::dot(info.normal, halfwayvector), 0.0f), 64.0);

	// Attenuation
	float attenuation = 1.0f / (1.0f + light.linear * distance + light.quadratic * (distance * distance));

	glm::vec3 ambient;
	glm::vec3 diffuse; 
	glm::vec3 specular;

	//Get colours from texture maps
	if (info.material->diffuseMap.isLoaded)
	{
		float4 diffTex = tex2D<float4>(info.material->diffuseMap.textureObject, info.texCoords.x, info.texCoords.y);
		float4 specTex = tex2D<float4>(info.material->specularMap.textureObject, info.texCoords.x, info.texCoords.y);

		//Convert to vec3
		glm::vec3 matDiffTex = glm::vec3(diffTex.x, diffTex.y, diffTex.z);
		glm::vec3 matSpecTex = glm::vec3(specTex.x, specTex.y, specTex.z);

		//Combine colours
		ambient = light.ambient * matDiffTex;
		diffuse = light.diffuse * diff * matDiffTex;
		specular = light.specular * spec * matSpecTex;

	}
	else
	{
		//Combine colours
		ambient = light.ambient * info.material->ambient;
		diffuse = light.diffuse * diff * info.material->diffuse;
		specular = light.specular * spec * info.material->specular;
	}

	return ambient + (attenuation * (diffuse + specular));
}

//
//	Calculates the reflected colour of a point
//
__device__
glm::vec3 calculateReflection(Environment *environment, CollisionInfo &info, int depth)
{
		//Compute reflected ray
		Ray R;
		R.origin = info.point;
		reflect(info.ray.direction, info.normal, R.direction, info.cosI);

		//Compute the reflected colour
		glm::vec3 colour = trace(environment, R, depth);

		//Reduce intensity depending on the reflectivness
		colour *= info.material->reflectiveness;

		return colour;
}

//
//	Calculates the refracted colour of a point
//
__device__
glm::vec3 calculateRefraction(Environment *environment, CollisionInfo &info, int depth)
{
	glm::vec3 colour;

	Ray T;
	//Calculate new refraction vector
	//Returns false if total internal reflection occurs.
	if (refract(info.ray.direction, info.normal, T.direction, info.eta, info.cosI))
	{
		//Initalise ray with refracted direction vector = Ray(info.point, t_T);
		T.origin = info.point;

		//Compute the reflected colour
		colour = trace(environment, T, depth);

		colour *= info.material->transparency;
	}
	else
	{
		colour = calculateReflection(environment, info, depth);
	}

	return colour;
}

//
//	Calculates the primary ray
//
__device__
void primary(glm::vec3 &primary, CudaCamera *camera, int x, int y, float xOffset, float yOffset)
{

	//Ray direction
	//Normalize pixel from 0 : screenWidth to 0 : 1
	//The addition of 0.5f defines the center of each pixel
	float xPix = (x + xOffset) * camera->invWidth;
	float yPix = (y + yOffset) * camera->invHeight;

	//Transform pixels from 0 : 1 to -1 : 1
	xPix = (2 * xPix - 1) * camera->aspectRatio * camera->scale;
	yPix = (1 - 2 * yPix) * camera->scale;

	//Calculate direction vector
	primary = glm::normalize(glm::vec3(xPix, yPix, 1.0f) - glm::vec3(0.0f));
	primary = glm::vec3(glm::vec4(primary, 1.0f) * camera->viewMatrix);
}

//
//	Calculates the reflection ray direction
//
__device__
void reflect(glm::vec3 &incident, glm::vec3 &normal, glm::vec3 &reflect, float &cosI)
{
	reflect = incident + 2.0f * cosI * normal;
}

//
//	Calculates the refracted ray direction
//
__device__ 
bool refract(glm::vec3 &incident, glm::vec3 &normal, glm::vec3 &transmitted, float &eta, float &cosI)
{
	glm::vec3 N;
	if (cosI > 0.0f)
		N = normal;
	else
		N = -normal;

	float sinT2 = 1.0f - eta * eta * (1.0f - cosI * cosI);

	if (sinT2 < 0.0f)
		return false;		// total internal reflection 

	transmitted = glm::vec3(eta) * incident + glm::vec3((eta * cosI - sqrt(sinT2))) * N;

	return true;
}

//
//	Schlicks Approximation
//
__device__ 
float schlick(float &m1, float &m2, float &eta, float &cosI)
{
	float r0 = (m1 - m2) / (m1 + m2);
	r0 *= r0;

	if (m1 > m2)
	{
		float sinT2 = eta * eta * (1.0f - cosI * cosI);
		if (sinT2 > 1.0f)
			return 1.0f;
		float x = 1.0f - sqrt(1.0f - sinT2);
		return r0 + (1.0f - r0) * x * x * x * x * x;
	}

	float x = 1.0f - cosI;
	return r0 + (1.0f - r0) * x * x * x * x * x;
}