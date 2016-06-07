#ifndef _MODEL_H
#define _MODEL_H

//GLEW
#include <GL/glew.h>

//ASSIMP
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

//SOIL
#include <SOIL/SOIL.h>

//GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

//STD
#include <vector>
#include <string>

//Ray Tracer
#include "Mesh.h"
#include "Material.h"

//Tools
#include "Debug.h"

class Model
{
public:

	glm::vec3 position;

	glm::mat4 rotation;
	glm::mat4 translation;

	//Model Matrix
	glm::mat4 modelMatrix;

	//Meshes
	Mesh *meshes, *d_meshes;
	size_t numMeshes;
	size_t numMeshesBytes;

	std::string directory;

	Model();
	Model(std::string path, Material material, glm::vec3 position, glm::vec3 angle);

	void translate(glm::vec3 position, float dt = 1.0f);
	void rotate(glm::vec3 angle, float dt = 1.0f);

	void deleteMem();

	void loadModel(std::string path);

	void processNode(aiNode* node, const aiScene* scene, std::vector<Mesh> &t_meshes);
	Mesh processMesh(aiMesh* mesh, const aiScene* scene);

	Texture loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName);
	//Texture TextureFromFile(const char* path, std::string directory);

};

#endif