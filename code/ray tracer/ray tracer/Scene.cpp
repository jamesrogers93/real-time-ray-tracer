#include "Scene.h"

Scene::Scene() : init(true), deltaTime(0.0f), lastFrame(0.0f)
{
}

void Scene::updateDelta()
{
	if (init)
	{
		lastFrame = glfwGetTime();
		init = false;
	}
	GLfloat currentFrame = glfwGetTime();
	deltaTime = currentFrame - lastFrame;
	lastFrame = currentFrame;
}

void Scene::deleteMem()
{
	//Delete Settings
	//free(environment->settings);
	gpuErrchk(cudaFree(environment->d_settings));

	//Delete Models
	for (int i = 0; i < environment->numModels; i++)
		environment->models[i].deleteMem();

	if (environment->numModels > 0)
	{
		free(environment->models);
		gpuErrchk(cudaFree(environment->d_models));
	}

	//Delete Triangles
	if (environment->numTriangles > 0)
	{
		free(environment->triangles);
		gpuErrchk(cudaFree(environment->d_triangles));
	}

	//Delete Spheres
	if (environment->numSpheres > 0)
	{
		free(environment->spheres);
		gpuErrchk(cudaFree(environment->d_spheres));
	}

	//Delete Lights
	if (environment->numLights > 0)
	{
		free(environment->lights);
		gpuErrchk(cudaFree(environment->d_lights));
	}

	//Delete KDTree
	environment->kdTree->deleteMem();
	free(environment->kdTree);
	gpuErrchk(cudaFree(environment->d_kdTree));

	//Delete Random Number
	free(environment->randomNumbers);
	gpuErrchk(cudaFree(environment->d_randomNumbers));

	//Delete Environment
	free(environment);
	gpuErrchk(cudaFree(d_environment));
}

void Scene::renderScene()
{
	//This starts the raytracer and writes image dat to an OpenGL Texture in Display
	generateRaytracedImage();

	//Draw Display with new texture on to screen
	display.drawDisplay();
}

void Scene::initScene(std::vector<Model> models, std::vector<Sphere> tempSpheres, std::vector<Light> tempLights, Camera *tempCamera, Settings &settings)
{
	//ENVIRONMENT//
	numEnvironmentBytes = sizeof(Environment);
	environment = (Environment *)malloc(numEnvironmentBytes);

	//SETTINGS//
	environment->settings = (Settings *)malloc(sizeof(Settings));
	environment->settings = &settings;

	//Store settings on the device
	gpuErrchk(cudaMalloc(&environment->d_settings, sizeof(Settings)));
	gpuErrchk(cudaMemcpy(environment->d_settings, environment->settings, sizeof(Settings), cudaMemcpyHostToDevice));

	//MODELS//

	//Get number of model and bytes required to store all models
	environment->numModels = models.size();
	environment->numModelsBytes = environment->numModels * sizeof(Model);

	//Allocate and copy models vector to models array
	environment->models = (Model *)malloc(environment->numModelsBytes);
	memcpy(environment->models, models.data(), environment->numModelsBytes);

	//Allocate and copy models array to device
	gpuErrchk(cudaMalloc(&environment->d_models, environment->numModelsBytes));
	gpuErrchk(cudaMemcpy(environment->d_models, environment->models, environment->numModelsBytes, cudaMemcpyHostToDevice));

	//Count all indices from models and translate verticies
	Mesh *mesh;
	environment->numTriangles = 0;
	for (size_t i = 0; i < environment->numModels; i++)
	{
		for (size_t j = 0; j < environment->models[i].numMeshes; j++)
		{
			//Get mesh
			mesh = &environment->models[i].meshes[j];

			//Increment numTriangles counter
			environment->numTriangles += mesh->numIndices;

			//Translate vertices
			for (size_t k = 0; k < mesh->numVertices; k++)
				mesh->vertices[k].position = glm::vec3(environment->models[i].modelMatrix * glm::vec4(mesh->vertices[k].position, 1.0f));
		}
	}

	//Three indices per triangle
	environment->numTriangles /= 3;
	environment->numTrianglesBytes = environment->numTriangles * sizeof(Triangle);

	//Allocate triangles array
	environment->triangles = (Triangle *)malloc(environment->numTrianglesBytes);

	//Fill triangles array with data
	int index = 0;
	for (size_t i = 0; i < environment->numModels; i++)
	{
		for (size_t j = 0; j < environment->models[i].numMeshes; j++)
		{
			mesh = &environment->models[i].meshes[j];
			//Loop over indices in increments of 3 (3 indices per triangle)
			for (size_t k = 0; k < mesh->numIndices; k += 3)
			{
				Vertex V0 = mesh->vertices[mesh->indices[k]];
				Vertex V1 = mesh->vertices[mesh->indices[k+1]];
				Vertex V2 = mesh->vertices[mesh->indices[k+2]];
				environment->triangles[index] = Triangle(V0, V1, V2, i, j);

				//Compute triangle normal, tangnet and binormal
				Triangle *T = &environment->triangles[index];

				//Edges of triangle
				glm::vec3 deltaPos1 = T->V0V1;
				glm::vec3 deltaPos2 = T->V0V2;

				//UC delta
				glm::vec2 deltaUV1 = T->V0.texCoords - T->V1.texCoords;
				glm::vec2 deltaUV2 = T->V0.texCoords - T->V2.texCoords;

				//Tangent and bitangent
				float r = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
				T->tangent = glm::normalize((deltaPos1 * deltaUV2.y - deltaPos2 * deltaUV1.y)*r);
				T->bitangent = glm::normalize((deltaPos2 * deltaUV1.x - deltaPos1 * deltaUV2.x)*r);

				index++;
			}
		}
	}

	//Allocate and copy triangles array to device
	gpuErrchk(cudaMalloc(&environment->d_triangles, environment->numTrianglesBytes));
	gpuErrchk(cudaMemcpy(environment->d_triangles, environment->triangles, environment->numTrianglesBytes, cudaMemcpyHostToDevice));

	//Free mesh memory in objects
	for (size_t i = 0; i < environment->numModels; i++)
	{
		for (size_t j = 0; j < environment->models[i].numMeshes; j++)
		{
		//	free(environment->models[i].meshes[j].indices);
		//	free(environment->models[i].meshes[j].vertices);

			/////////////DELETE////////////////
		}
	}

	//SPHERES//

	//Get number of spheres and bytes required to store all spheres
	environment->numSpheres = tempSpheres.size();
	environment->numSpheresBytes = environment->numSpheres * sizeof(Sphere);

	//Allocate and copy sphere vector to array
	environment->spheres = (Sphere *)malloc(environment->numSpheresBytes);
	memcpy(environment->spheres, tempSpheres.data(), environment->numSpheresBytes);

	//Allocate and copy array to device
	gpuErrchk(cudaMalloc(&environment->d_spheres, environment->numSpheresBytes));
	gpuErrchk(cudaMemcpy(environment->d_spheres, environment->spheres, environment->numSpheresBytes, cudaMemcpyHostToDevice));

	//LIGHTS//

	//Get number of lights and bytes required to store all lights
	environment->numLights = tempLights.size();
	environment->numLightsBytes = environment->numLights * sizeof(Light);

	//Allocate and copy light vector to array
	environment->lights = (Light *)malloc(environment->numLightsBytes);
	memcpy(environment->lights, tempLights.data(), environment->numLightsBytes);

	//Allocate and copy array to device
	gpuErrchk(cudaMalloc(&environment->d_lights, environment->numLightsBytes));
	gpuErrchk(cudaMemcpy(environment->d_lights, environment->lights, environment->numLightsBytes, cudaMemcpyHostToDevice));

	//KDTREE//

	//Get number of bytes needed for KDTree
	environment->numKDTreeBytes = sizeof(KDTree::KDNode);

	//Allocate memory for KDTree
	environment->kdTree = (KDTree::KDNode *)malloc(environment->numKDTreeBytes);

	//Build tree
	environment->kdTree = environment->kdTree->buildTree(environment->triangles, environment->numTriangles, environment->spheres, environment->numSpheres);

	int count = environment->kdTree->count();

	//Allocate memory and store kdtree on the gpu
	gpuErrchk(cudaMalloc(&environment->d_kdTree, environment->numKDTreeBytes));
	gpuErrchk(cudaMemcpy(environment->d_kdTree, environment->kdTree, environment->numKDTreeBytes, cudaMemcpyHostToDevice));

	//CAMERA//
	camera = tempCamera;
	environment->camera = camera->d_cameraData;

	//RANDOM NUMBERS//
	size_t numRandomFloatBytes = camera->cameraData->width * camera->cameraData->height * 4 * sizeof(float);
	environment->randomNumbers = (float *)malloc(numRandomFloatBytes);

	for (int i = 0; i < camera->cameraData->width * camera->cameraData->height * 4; i++)
	{
		float random = rand() / ((float)RAND_MAX);
		environment->randomNumbers[i] = random;
	}

	gpuErrchk(cudaMalloc(&environment->d_randomNumbers, numRandomFloatBytes));
	gpuErrchk(cudaMemcpy(environment->d_randomNumbers, environment->randomNumbers, numRandomFloatBytes, cudaMemcpyHostToDevice));

	//ENVIRONMENT//
	//Copy environment to device
	gpuErrchk(cudaMalloc(&d_environment, numEnvironmentBytes));
	gpuErrchk(cudaMemcpy(d_environment, environment, numEnvironmentBytes, cudaMemcpyHostToDevice));
}

void Scene::initDisplay(int width, int height)
{
	display.initDisplay(width, height);
}

void Scene::initRaytracer()
{

	//Init Cuda texture resource
	gpuErrchk(cudaGraphicsGLRegisterImage(&viewCudaResource, display.viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

	//Set block and grid size
	blockSize.x = 4;
	blockSize.y = 4;
	gridSize.x = camera->cameraData->width / blockSize.x;
	gridSize.y = camera->cameraData->height / blockSize.y;

	//Increase stack size limit
	size_t size_heap, size_stack;
	gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, 15000));
	gpuErrchk(cudaDeviceGetLimit(&size_heap, cudaLimitMallocHeapSize));
	gpuErrchk(cudaDeviceGetLimit(&size_stack, cudaLimitStackSize));
	printf("Heap size found to be %d; Stack size found to be %d\n", (int)size_heap, (int)size_stack);
}

void Scene::updateScene()
{
	updateDelta();

	camera->updateCamera();

	//Update the settings
	gpuErrchk(cudaMemcpy(environment->d_settings, environment->settings, sizeof(Settings), cudaMemcpyHostToDevice));

}

void Scene::generateRaytracedImage()
{
	//Ready cuda for cuda-opengl interop
	gpuErrchk(cudaGraphicsMapResources(1, &viewCudaResource));
	cudaArray_t viewCudaArray;
	gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, viewCudaResource, 0, 0));
	cudaResourceDesc viewCudaArrayResourceDesc;
	memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
	viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
	viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
	cudaSurfaceObject_t viewCudaSurfaceObject;
	gpuErrchk(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));

	//Call cuda kernal to ray trace scene
	rayTraceImage(gridSize, blockSize, viewCudaSurfaceObject, d_environment);

	//Clear up variables for cuda-opengl interop
	gpuErrchk(cudaDestroySurfaceObject(viewCudaSurfaceObject));
	gpuErrchk(cudaGraphicsUnmapResources(1, &viewCudaResource));
}