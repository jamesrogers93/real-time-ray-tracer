#ifndef Scene_cuh
#define Scene_cuh

//STD
#include <time.h>

//Raytracer
#include "Environment.h"
#include "Display.h"
#include "Debug.h"
#include "RayTrace.h"

//GLM
#include <glm/glm.hpp>

//GLFW
#include <GLFW\glfw3.h>

class Scene
{
public:

	bool init;
	float deltaTime, lastFrame;

	//Objects, Spheres, Lights etc.
	Environment *environment, *d_environment;
	size_t numEnvironmentBytes;

	//Camera
	Camera *camera;
	
	//CUDA graphics resource
	cudaGraphicsResource_t viewCudaResource;
	Display display;

	//Kernel thread and block size
	dim3 blockSize;
	dim3 gridSize;

	Scene();

	void deleteMem();

	void renderScene();

	void updateScene();

	void initScene(std::vector<Model> tempModels, std::vector<Sphere> tempSpheres, std::vector<Light> tempLights, Camera *tempCamera, Settings &settings);
	void initDisplay(int width, int height);
	void initRaytracer();

	void generateRaytracedImage();

	void updateDelta();
};

#endif