#include "Intersection.h"

/*
*	KDTree traversal
*/
__device__ 
bool kdSearch(Model *models, Triangle *triangles, Sphere *spheres, KDTree::KDNode *root, CollisionInfo &info)
{
	//Global min and max
	float gtMin, gtMax;
	
	//Check if ray intersects root voxel
	if (intersectVoxel(root->voxel, info.ray, gtMin, gtMax))
	{
		//Set min and max to global min
		float tMin = gtMin;
		float tMax = gtMin;

		//Distance of closest hit
		float tHit = INFINITY;

		//Barycentric coords
		float u, v;

		//Index of primitive hit
		Triangle *triangle;
		Sphere *sphere;
		
		int triangleIndex = -1;
		int sphereIndex	= -1;

		//Type of object hit
		hitType closestHit = NONE_HIT;
		
		while (tMax < gtMax)
		{
			KDTree::KDNode *node = root;
			tMin = tMax;
			tMax = gtMax;

			//Find closest leaf node
			while (!node->isLeaf)
			{
				float a = node->plane.pk;
				float tSplit = (node->plane.pe - info.ray.origin[a]) / info.ray.direction[a];

				//Sort near and far node
				KDTree::KDNode *first = 0, *second = 0;
				if (info.ray.origin[a] < node->plane.pe)
				{
					first = node->d_left;
					second = node->d_right;
				}
				else
				{
					first = node->d_right;
					second = node->d_left;
				}

				if (tSplit >= tMax || tSplit < 0.0f)
					node = first;
				else if (tSplit <= tMin)
					node = second;
				else
				{
					node = first;
					tMax = tSplit;
				}
			}

			//Temp distance and barycentric coords
			float t_t;
			float t_u, t_v;

			//Test ray against all triangles in leaf node
			for (size_t i = 0; i < node->numTriangleIndices; i++)
			{
				//Get triangle data from array
				int index = node->d_triangleIndices[i];
				//int index = (int)tex1Dfetch<float>(node->triangleTextureIndicesTexture, i);

				glm::vec3 V0 = triangles[index].V0.position;
				glm::vec3 V0V1 = triangles[index].V0V1;
				glm::vec3 V0V2 = triangles[index].V0V2;

				//Test triangle against ray
				if (intersectTriangle(info.ray, V0, V0V1, V0V2, t_t, info.tMin, info.tMax, t_u, t_v))
				{
					//Store closest hit data
					if (t_t < tHit)
					{
						tHit = t_t;
						u = t_u;
						v = t_v;
						triangle = &triangles[index];
						triangleIndex = index;
						closestHit = TRI_HIT;
					}
				}
			}
			
			//Test ray against all spheres in leaf node
			for (size_t i = 0; i < node->numSphereIndices; i++)
			{
				//Get Sphere data
				int index = node->d_sphereIndices[i];
				glm::vec3 origin = spheres[index].origin;
				float radius = spheres[index].radius;

				//Test sphere against ray
				if (intersectSphere(info.ray, origin, radius, t_t, info.tMin, info.tMax))
				{
					//Store closest hit data
					if (t_t < tHit)
					{
						tHit = t_t;
						sphere = &spheres[index];
						sphereIndex = index;
						closestHit = SPHERE_HIT;
					}
				}
			}
			
			//If hit found, set variables
			if (tHit <= (tMax + 0.0001)) //Apply padding to 
			{
				info.t = tHit;

				//Find point of intersection
				info.point = info.ray.origin + info.ray.direction * info.t;

				if (closestHit == TRI_HIT)
				{
					//Get material
					info.material = &models[triangle->modelID].d_meshes[triangle->meshID].material;

					//Set texture coordinates
					info.texCoords = interpolateTexCoords(triangle->V0.texCoords, triangle->V1.texCoords, triangle->V2.texCoords, u, v);

					//Calculate normal
					if (info.material->bumpMap.isLoaded)
					{
						//Get normal from bump map
						float4 texNormal = tex2D<float4>(info.material->bumpMap.textureObject, info.texCoords.x, info.texCoords.y);
						info.normal.x = (2 * texNormal.x) - 1;
						info.normal.y = (2 * texNormal.y) - 1;
						info.normal.z = (2 * texNormal.z) - 1;

						glm::vec3 inormal = interpolateNormal(triangle->V0.normal, triangle->V1.normal, triangle->V2.normal, u, v);
						glm::mat3 TBN = glm::mat3(triangle->tangent, triangle->bitangent, inormal);

						info.normal = glm::mat3(glm::transpose(glm::inverse(TBN))) * info.normal;
					}
					else
					{
						//Get normal by interpolating vertices
						info.normal = interpolateNormal(triangle->V0.normal, triangle->V1.normal, triangle->V2.normal, u, v);
					}

					//Apply the model matrix to the normal
					info.normal = glm::normalize(glm::mat3(glm::transpose(glm::inverse(models[triangle->modelID].modelMatrix))) * info.normal);

					//Move point slightly in front to prevent ray intersection same object
					//info.point = info.point + info.normal * 0.0001f;

				}
				else if (closestHit == SPHERE_HIT)
				{
					//Get material
					info.material = &sphere->material;

					//Set texture coordinates
					info.texCoords.x = ((glm::atan(info.point.x, info.point.z) / PI) + 1.0f) * 0.5f;
					info.texCoords.y = (asin(info.point.y) / PI) + 0.5f;

					//Calculate the normal
					info.normal = (info.point - sphere->origin) / sphere->radius;
					info.normal = glm::normalize(glm::mat3(glm::transpose(glm::inverse(sphere->modelMatrix))) * info.normal);

					//Move point slightly in front to prevent ray intersection same object
					//info.point = info.point + info.normal * 0.0001f;
				}

				return HIT;
			}
		}
	}

	return NO_HIT;
}

/*
*	Ray - Voxel intersection
*/
__device__ 
bool intersectVoxel(KDTree::Voxel &voxel, Ray& ray, float &tmin, float &tmax)
{
	float tymin, tymax, tzmin, tzmax;

	if (ray.direction.x >= 0)
	{
		tmin = (voxel.min.x - ray.origin.x) / ray.direction.x;
		tmax = (voxel.max.x - ray.origin.x) / ray.direction.x;
	}
	else
	{
		tmin = (voxel.max.x - ray.origin.x) / ray.direction.x;
		tmax = (voxel.min.x - ray.origin.x) / ray.direction.x;
	}

	if (ray.direction.y >= 0)
	{
		tymin = (voxel.min.y - ray.origin.y) / ray.direction.y;
		tymax = (voxel.max.y - ray.origin.y) / ray.direction.y;
	}

	else
	{
		tymin = (voxel.max.y - ray.origin.y) / ray.direction.y;
		tymax = (voxel.min.y - ray.origin.y) / ray.direction.y;
	}

	if ((tmin > tymax) || (tymin > tmax))
		return NO_HIT;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	if (ray.direction.z >= 0)
	{
		tzmin = (voxel.min.z - ray.origin.z) / ray.direction.z;
		tzmax = (voxel.max.z - ray.origin.z) / ray.direction.z;
	}
	else
	{
		tzmin = (voxel.max.z - ray.origin.z) / ray.direction.z;
		tzmax = (voxel.min.z - ray.origin.z) / ray.direction.z;
	}

	if ((tmin > tzmax) || (tzmin > tmax))
		return NO_HIT;

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;

	return HIT;
}

/*
*	Ray - Triangle intersection
*/
__device__ 
bool intersectTriangle(const Ray &ray, const glm::vec3 &V0, const glm::vec3 &V0V1, const glm::vec3 &V0V2, float &t, const float tMin, const float tMax, float &u, float &v)
{
	glm::vec3 pvec = glm::cross(ray.direction, V0V2);
	float det = glm::dot(V0V1, pvec);

	if (fabs(det) <= 0.0f)
		return NO_HIT;

	float invDet = 1 / det;

	glm::vec3 tvec = ray.origin - V0;
	u = glm::dot(tvec, pvec) * invDet;
	if (u < 0.0f || u > 1.0f)
		return NO_HIT;

	glm::vec3 qvec = glm::cross(tvec, V0V1);
	v = glm::dot(ray.direction, qvec) * invDet;
	if (v < 0.0f || u + v > 1.0f)
		return NO_HIT;

	t = glm::dot(V0V2, qvec) * invDet;

	//Check if intersecton point is closer than the minimum distance, or further away than the max distance
	if (t < tMin || t > tMax)
		return NO_HIT;

	return HIT;
}

/*
*	Ray - Sphere intersection
*/
__device__ 
bool intersectSphere(const Ray &ray, const glm::vec3 &origin, const float &radius, float& t, const float tMin, const float tMax)
{

	//Find B and C coefficients
	float B = 2 * ((ray.direction.x * (ray.origin.x - origin.x))
		+ (ray.direction.y * (ray.origin.y - origin.y))
		+ (ray.direction.z * (ray.origin.z - origin.z)));

	float C = ((ray.origin.x - origin.x) * (ray.origin.x - origin.x))
		+ ((ray.origin.y - origin.y) * (ray.origin.y - origin.y))
		+ ((ray.origin.z - origin.z) * (ray.origin.z - origin.z))
		- (radius * radius);

	//Calculate the discriminant
	float discriminant = (B*B) - 4 * C;

	//If discriminant is negative, there is no collision
	if (discriminant < 0)
	{
		return NO_HIT;
	}

	//Calculate t
	t = (-B - sqrt((B*B) - 4 * C)) / 2;

	return (t > tMin && t < tMax);

	//Get direction for ray to sphere
	/*glm::vec3 OC = origin - ray.origin;

	float L2OC = glm::dot(OC, OC);

	bool inside;

	inside = (L2OC < (radius * radius));

	float tca = glm::dot(OC, ray.direction);

	if (tca < 0.0f && !inside)
		return NO_HIT;

	float t2hc = (radius * radius) - L2OC + (tca * tca);

	if (t2hc < 0)
		return NO_HIT;

	if (inside)
		t = tca + sqrt(t2hc);
	else
		t = tca - sqrt(t2hc);

	return (t > tMin && t < tMax);*/

}