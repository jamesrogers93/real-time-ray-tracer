#include "Model.h"

Model::Model()
{

}

Model::Model(std::string path, Material material, glm::vec3 position, glm::vec3 angle)
{
	this->loadModel(path);
	this->translate(position);
	this->rotate(angle);

	//Set material
	for (size_t i = 0; i < numMeshes; i++)
	{
		meshes[i].material.ambient = material.ambient;
		meshes[i].material.diffuse = material.diffuse;
		meshes[i].material.specular = material.specular;
		meshes[i].material.reflectiveness = material.reflectiveness;
		meshes[i].material.transparency = material.transparency;
		meshes[i].material.refractiveIndex = material.refractiveIndex;
		meshes[i].material.reflective = material.reflectiveness > 0.0f;
		meshes[i].material.refractive = material.transparency > 0.0f;
	}

	//Allocate and copy array to device
	gpuErrchk(cudaMalloc(&d_meshes, numMeshesBytes));
	gpuErrchk(cudaMemcpy(d_meshes, meshes, numMeshesBytes, cudaMemcpyHostToDevice));
}

void Model::deleteMem()
{
	for (size_t i = 0; i < numMeshes; i++)
		meshes[i].deleteMem();

	gpuErrchk(cudaFree(d_meshes));
}

void Model::loadModel(std::string path)
{
	Assimp::Importer import;

	//Import mesh to ASSIMP::aiScene as triangles and flipped texCoords on y axis
	const aiScene* scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

	//Check if ASSIMP::Scene loaded correctly
	if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		std::cout << "ERROR, Assimp " << import.GetErrorString() << std::endl;
		return;
	}

	directory = path.substr(0, path.find_last_of('/'));

	//Recursively process each ASSIMP::node within the ASSIMP:Scene
	std::vector<Mesh> t_meshes;
	processNode(scene->mRootNode, scene, t_meshes);

	numMeshes = t_meshes.size();
	numMeshesBytes = numMeshes * sizeof(Mesh);

	//Allocate and copy vector to array
	meshes = (Mesh *)malloc(numMeshesBytes);
	memcpy(meshes, t_meshes.data(), numMeshesBytes);
}

void Model::processNode(aiNode* node, const aiScene* scene, std::vector<Mesh> &t_meshes)
{
	//Add all meshes to this->meshes vector 
	for (GLuint i = 0; i < node->mNumMeshes; i++)
	{
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		t_meshes.push_back(this->processMesh(mesh, scene));
	}

	//Recursively proccess each node on all of its children
	for (GLuint i = 0; i < node->mNumChildren; i++)
		this->processNode(node->mChildren[i], scene, t_meshes);
}

Mesh Model::processMesh(aiMesh* mesh, const aiScene* scene)
{
	//VERTEX

	//Create and allocate vertices array
	Vertex *vertices;
	size_t numVertices = mesh->mNumVertices;
	size_t numVerticesBytes = numVertices * sizeof(Vertex);
	vertices = (Vertex *)malloc(numVerticesBytes);

	//FIll vertices array
	for (GLuint i = 0; i < numVertices; i++)
	{

		Vertex vertex;

		//Fill position data
		vertex.position = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);

		//Fill normal data
		vertex.normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);

		//Fill texCoords data
		vertex.texCoords = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);

		//Fill tangent data
		/*glm::vec3 c1 = glm::cross(vertex.normal, glm::vec3(0.0, 0.0, 1.0));
		glm::vec3 c2 = glm::cross(vertex.normal, glm::vec3(0.0, 1.0, 0.0));
		if (glm::length(c1) > glm::length(c2))
		{
			vertex.tangent = glm::normalize(c1);
		}
		else
		{
			vertex.tangent = glm::normalize(c2);
		}

		//Fill ninormal data
		vertex.binormal = glm::normalize(glm::cross(vertex.normal, vertex.tangent));*/

		//Add vertex struct to vertices array
		vertices[i] = vertex;
	}

	//INDICIES

	//Create and allocate indices array
	int *indices;
	size_t numIndices = mesh->mNumFaces * 3;
	size_t numIndicesBytes = numIndices * sizeof(int);
	indices = (int *)malloc(numIndicesBytes);

	//Fill indices array
	for (GLuint i = 0; i < mesh->mNumFaces; i++)
	{
		aiFace face = mesh->mFaces[i];

		int j = i * 3;
		indices[j] = face.mIndices[0];
		indices[j+1] = face.mIndices[1];
		indices[j+2] = face.mIndices[2];
	}

	//MATERIALS
	Material m;

	if (mesh->mMaterialIndex >= 0)
	{
		//Get material
		aiMaterial* materials = scene->mMaterials[mesh->mMaterialIndex];

		//Diffuse Map
		m.diffuseMap = loadMaterialTextures(materials, aiTextureType_DIFFUSE, "texture_diffuse");

		//Specular maps
		m.specularMap = loadMaterialTextures(materials, aiTextureType_SPECULAR, "texture_specular");

		//Normal maps
		m.bumpMap = loadMaterialTextures(materials, aiTextureType_HEIGHT, "texture_normal");

	}

	return Mesh(vertices, numVertices, numVerticesBytes, indices, numIndices, numIndicesBytes, m);
}

Texture Model::loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName)
{
	Texture texture;

	int numTextures = mat->GetTextureCount(type);

	for (GLuint i = 0; i < numTextures; i++)
	{
		aiString str;
		mat->GetTexture(type, i, &str);

		//Get file path of texture
		std::string filename = std::string(str.C_Str());
		filename = directory + '/' + filename;

		//Get image data
		unsigned char* image = SOIL_load_image(filename.c_str(), &texture.width, &texture.height, 0, SOIL_LOAD_RGBA);

		//Get number of bytes required to store texture
		texture.numtextureBytes = texture.width * texture.height *sizeof(float4);

		//Allocate texture array memory
		texture.textureArray = (float4 *)malloc(texture.numtextureBytes);

		//Fill texture array
		for (int i = 0; i < texture.width * texture.height; i ++)
		{
			texture.textureArray[i] = make_float4((float)image[i*4] / 255, (float)image[i*4 + 1] / 255, (float)image[i*4 + 2] / 255, (float)image[i*4 + 3] / 255);
		}

		//Allocate CUDA array in device memory 
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		gpuErrchk(cudaMallocArray(&texture.d_textureArray, &channelDesc, texture.width, texture.height));

		//Copy to texture array on device
		gpuErrchk(cudaMemcpyToArray(texture.d_textureArray, 0, 0, texture.textureArray, texture.numtextureBytes, cudaMemcpyHostToDevice));

		// Specify texture 
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = texture.d_textureArray;

		// Specify texture object parameters 
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 1;

		// Create texture object 
		texture.textureObject = 0;
		cudaCreateTextureObject(&texture.textureObject, &resDesc, &texDesc, NULL);

		//Set bool to true, texture loaded
		texture.isLoaded = true;

		return texture;
	}

	return texture;
}

/*Texture Model::TextureFromFile(const char* path, std::string directory)
{
	//Generate texture ID and load texture data 
	std::string filename = std::string(path);
	filename = directory + '/' + filename;
	GLuint textureID;
	glGenTextures(1, &textureID);
	int width, height;
	unsigned char* image = SOIL_load_image(filename.c_str(), &width, &height, 0, SOIL_LOAD_RGBA);
	for (int i = 0; i < width * height; i++)
	{
		char h = image[i];
	}

	// Assign texture to ID
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
	glGenerateMipmap(GL_TEXTURE_2D);

	// Parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
	SOIL_free_image_data(image);
	return textureID;

	return 0;
}*/

void Model::translate(glm::vec3 position, float dt)
{
	this->position += position;

	this->translation = glm::translate(glm::mat4(), this->position);

	this->modelMatrix = this->translation * this->rotation;
}

void Model::rotate(glm::vec3 angle, float dt)
{
	glm::vec3 rotate = angle * dt;
	glm::quat q;

	//rotation about the local x axis
	q = glm::angleAxis(glm::radians((float)rotate.x), glm::vec3(this->rotation[0][0], this->rotation[0][1], this->rotation[0][2]));
	this->rotation = glm::mat4_cast(q) * this->rotation;

	//rotation about the local y axis
	q = glm::angleAxis(glm::radians((float)rotate.y), glm::vec3(this->rotation[1][0], this->rotation[1][1], this->rotation[1][2]));
	this->rotation = glm::mat4_cast(q) * this->rotation;

	//rotation about the local z axis
	q = glm::angleAxis(glm::radians((float)rotate.z), glm::vec3(this->rotation[2][0], this->rotation[2][1], this->rotation[2][2]));
	this->rotation = glm::mat4_cast(q) * this->rotation;

	this->modelMatrix = this->translation * this->rotation;
}