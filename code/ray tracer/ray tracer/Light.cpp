#include "Light.h"

Light::Light()
{

}

Light::Light(glm::vec3 origin, glm::vec3 ambient, glm::vec3 diffuse, glm::vec3 specular, float linear, float quadratic) : origin(origin), ambient(ambient), diffuse(diffuse), specular(specular), linear(linear), quadratic(quadratic)
{

}
