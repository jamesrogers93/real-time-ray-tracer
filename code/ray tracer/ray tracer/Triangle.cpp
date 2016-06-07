/*#include "Triangle.cuh"

__host__ __device__ Triangle::Triangle(){}

__host__ __device__ Triangle::Triangle(Vertex *V0, Vertex *V1, Vertex *V2)
{
	this->V0 = V0;
	this->V1 = V1;
	this->V2 = V2;
}

__host__ __device__ void Triangle::update(const glm::mat4 &model)
{
	//Translate triangle
	V0->position = glm::vec3(model * glm::vec4(V0->position, 1.0f));
	V1->position = glm::vec3(model * glm::vec4(V1->position, 1.0f));
	V2->position = glm::vec3(model * glm::vec4(V2->position, 1.0f));
}

__device__ bool Triangle::intersectBasic(const Ray& ray, const glm::mat4 &model, float& t, const float tMax = 100.0f, const float tMin = 0.0f)
{
	//Translate triangle
	glm::vec3 v0 = glm::vec3(model * glm::vec4(V0->position, 1.0f));
	glm::vec3 v1 = glm::vec3(model * glm::vec4(V1->position, 1.0f));
	glm::vec3 v2 = glm::vec3(model * glm::vec4(V2->position, 1.0f));

	//Calcualte Normal
	glm::vec3 A = v1 - v0;
	glm::vec3 B = v2 - v0;

	glm::vec3 N = glm::cross(A, B);

	//Find P

	//Check if ray and plane are parrallel
	float NdotRayDirection = glm::dot(N, ray.direction);
	if (fabs(NdotRayDirection) < 0.0001)
		return false; // Do not intersect if parrallel

	float d = glm::dot(N, v0);

	//Compute t
	t = (glm::dot(N, ray.origin) + d) / NdotRayDirection;

	//Check if Triangle is closer than the minimum distance or further away than the max distance
	if (t < tMin || t > tMax)
		return false;

	//Intersection point
	glm::vec3 P = ray.origin + t * ray.direction;

	//Inside-Outside test
	glm::vec3 C;

	//Edge 0
	glm::vec3 edge0 = v1 - v0;
	glm::vec3 vp0 = P - v0;
	C = glm::cross(edge0, vp0);
	if (glm::dot(N, C) < 0)
		return false;

	//Edge 1
	glm::vec3 edge1 = v2 - v1;
	glm::vec3 vp1 = P - v1;
	C = glm::cross(edge1, vp1);
	if (glm::dot(N, C) < 0)
		return false;

	//Edge 2
	glm::vec3 edge2 = v0 - v2;
	glm::vec3 vp2 = P - v2;
	C = glm::cross(edge2, vp2);
	if (glm::dot(N, C) < 0)
		return false;

	return true;
}

__device__ bool Triangle::intersect(const Ray& ray,
	const glm::mat4 &model,
	float& t, const float tMax, const float tMin,
	float &u, float &v, float &w)
{
	//Translate triangle
	glm::vec3 v0 = glm::vec3(model * glm::vec4(V0->position, 1.0f));
	glm::vec3 v1 = glm::vec3(model * glm::vec4(V1->position, 1.0f));
	glm::vec3 v2 = glm::vec3(model * glm::vec4(V2->position, 1.0f));

	glm::vec3 v0v1 = v1 - v0;
	glm::vec3 v0v2 = v2 - v0;
	glm::vec3 pvec = glm::cross(ray.direction, v0v2);
	float det = glm::dot(v0v1, pvec);

	if (fabs(det) <= 0.0f)
		return false;

	float invDet = 1 / det;

	glm::vec3 tvec = ray.origin - v0;
	u = glm::dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1)
		return false;

	glm::vec3 qvec = glm::cross(tvec, v0v1);
	v = glm::dot(ray.direction, qvec) * invDet;
	if (v < 0.0f || u + v > 1.0f)
		return false;

	w = 1 - u - v;

	t = glm::dot(v0v2, qvec) * invDet;

	//Check if Triangle is closer than the minimum distance or further away than the max distance
	if (t < tMin || t > tMax)
		return false;

	return true;
}*/