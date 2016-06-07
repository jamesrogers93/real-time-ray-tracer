#include "Sphere.h"

Sphere::Sphere(glm::vec3 origin, float radius, Material material)
{
	this->translate(origin);
	this->radius = radius;
	this->material = material;

	this->radius2 = radius * radius;
}

void Sphere::translate(glm::vec3 position, float dt)
{
	this->origin += position;

	this->translation = glm::translate(glm::mat4(), this->origin);

	this->modelMatrix = this->translation;
}

