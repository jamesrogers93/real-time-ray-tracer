//STD
#include <iostream>
#include <fstream>

// GLEW
#define GLEW_STATIC
#include <GL/glew.h>

//GLFW
#include <GLFW/glfw3.h>

//Raytracer
#include "Scene.h"
#include "GraphicSettings.h"

//Tools
#include "Debug.h"

//Scene
Scene scene = Scene();

float ambience = 3.0f;

//Graphics settings
Settings settings;

//Control structures
struct Keyboard
{
	bool keys[1024];

	bool init;
	float keyTimer, lastKey;

	Keyboard() : init(true)
	{
		for (size_t i = 0; i < 1024; i++)
			keys[i] = false;

		keyTimer = (float)glfwGetTime();
	}
};

struct Mouse
{
	bool activated, first;
	float lastX, lastY, xPos, yPos;

	Mouse() : activated(false), first(true){}
};

//Control variables
Keyboard keyboard;
Mouse mouse;

//Control input callback functions
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void processUserInput();

//Scene initalise functions
void initScene();
void loadSettings();
void loadSceneData(std::vector<Model> &models, std::vector<Sphere> &spheres, std::vector<Light> &lights);
void loadModel(std::vector<Model> &models, std::ifstream &infile);
void loadSphere(std::vector<Sphere> &spheres, std::ifstream &infile);
void loadLight(std::vector<Light> &lights, std::ifstream &infile);

int main(int argc, const char * argv[])
{
	//Load settings
	loadSettings();

	// Init GLFW
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	GLFWwindow* gWindow = glfwCreateWindow(settings.width, settings.height, "Ray Tracer", NULL, NULL);

	// GLFW settings
	glfwMakeContextCurrent(gWindow);

	// Set the required callback functions
	glfwSetKeyCallback(gWindow, key_callback);
	glfwSetCursorPosCallback(gWindow, mouse_callback);

	//Hide the cursor
	glfwSetInputMode(gWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// initialise GLEW
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK)
		throw std::runtime_error("glewInit failed");

	glGetError();

	// make sure OpenGL version 4.1 API is available
	if (!GLEW_VERSION_4_1)
		throw std::runtime_error("OpenGL 4.1 API is not available.");

	//Configure CUDA
	int count;
	gpuErrchk(cudaGetDeviceCount(&count));

	cudaDeviceProp prop;
	int dev;

	gpuErrchk(cudaGetDevice(&dev));
	printf("ID of current CUDA device: %d\n", dev);

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 3;
	gpuErrchk(cudaChooseDevice(&dev, &prop));
	printf("ID of CUDA device closest to revision 1.3: %d\n", dev);

	gpuErrchk(cudaSetDevice(dev));
	gpuErrchk(cudaGLSetGLDevice(dev));

	//Set up scene
	initScene();

	// run while the window is open
	while (!glfwWindowShouldClose(gWindow)){

		// process pending events
		glfwPollEvents();

		//Process User Input
		processUserInput();

		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT);

		//Update the scene
		scene.updateScene();

		//Render the scene
		scene.renderScene();

		// swap the display buffers (displays what was just drawn)
		glfwSwapBuffers(gWindow);
	}

	// clean up and exit
	scene.deleteMem();

	//Close GLFW
	glfwTerminate();
}

void initScene()
{	
	std::vector<Model> models;
	std::vector<Sphere> spheres;
	std::vector<Light> lights;

	float pov = 90.0f;
	glm::vec3 origin = glm::vec3(0.0f, 4.8f, -8.0f);
	Camera *camera = new Camera(settings.width, settings.height, glm::radians(pov), origin);

	loadSceneData(models, spheres, lights);

	scene.initScene(models, spheres, lights, camera, settings);
	scene.initDisplay(settings.width, settings.height);
	scene.initRaytracer();
	
}

void loadSceneData(std::vector<Model> &models, std::vector<Sphere> &spheres, std::vector<Light> &lights)
{
	std::ifstream in("scene_files/scene_layout/scene.txt");
	std::string sceneFile;
	in >> sceneFile;

	std::ifstream infile("scene_files/scene_layout/" + sceneFile);

	char type;
	while (infile >> type)
	{
		switch (type)
		{
		case 'm':
			loadModel(models, infile);
			break;
		case 's':
			loadSphere(spheres, infile);
			break;
		case 'l':
			loadLight(lights, infile);
			break;
		case 'c':
			std::cout << "Unhandled case. Camera in scene file loader not yet implemented" << std::endl;
			exit(0);
			break;
		default:
			std::cout << "Unhandled case. Scene file incorrect format" << std::endl;
			exit(0);
			break;
		}
	}
}

void loadModel(std::vector<Model> &models, std::ifstream &infile)
{
	std::string path = "scene_files/objects/models/";
	std::string modelFile;
	glm::vec3 origin, rotation;
	float reflect, refract, refractIndex;
	glm::vec3 diffuse;

	//Read model data
	infile  >> modelFile								//Model file name
			>> origin.x >> origin.y >> origin.z			//Model position
			>> rotation.x >> rotation.y >> rotation.z	//Model rotation
			>> reflect >> refract >> refractIndex		//Model reflection/refraction properties
			>> diffuse.x >> diffuse.y >> diffuse.z;		//Model diffuse colour


	path = path + modelFile;

	Material material = Material();
	material.ambient = diffuse / ambience;
	material.diffuse = diffuse;
	material.specular = glm::vec3(1.0f);
	material.reflectiveness = reflect;
	material.transparency = refract;
	material.refractiveIndex = refractIndex;

	Model model = Model(path, material, origin, rotation);

	models.push_back(model);
}

void loadSphere(std::vector<Sphere> &spheres, std::ifstream &infile)
{
	std::string file;
	glm::vec3 origin;
	float radius;
	float reflect, refract, refractIndex;
	glm::vec3 diffuse;

	infile	>> origin.x >> origin.y >> origin.z			//Sphere position
			>> radius									//Sphere radius
			>> reflect >> refract >> refractIndex		//Sphere reflection/refraction properties
			>> diffuse.x >> diffuse.y >> diffuse.z;		//Sphere diffuse colour

	Material material = Material();
	material.ambient = diffuse / ambience; 
	material.ambient = glm::clamp(material.ambient, 0.0f, 1.0f);
	material.diffuse = diffuse;
	material.specular = glm::vec3(1.0f);
	material.reflectiveness = reflect;
	material.transparency = refract;
	material.refractiveIndex = refractIndex;
	material.reflective = material.reflectiveness > 0.0f;
	material.refractive = material.transparency > 0.0f;

	Sphere sphere = Sphere(origin, radius, material);

	spheres.push_back(sphere);
}

void loadLight(std::vector<Light> &lights, std::ifstream &infile)
{
	std::string file;
	glm::vec3 origin;
	glm::vec3 ambient, diffuse, specular;
	float linear, quadratic;

	infile	>> origin.x >> origin.y >> origin.z			//Light position
			>> diffuse.x >> diffuse.y >> diffuse.z		//Light diffuse colour
			>> linear >> quadratic;						//Light attenuation

	//Calculate light ambient and specular
	ambient = diffuse / ambience; 
	specular = diffuse * 10.0f;

	ambient = glm::clamp(ambient, 0.0f, 1.0f);
	specular = glm::clamp(specular, 0.0f, 1.0f);

	Light light = Light(origin, ambient, diffuse, specular, linear, quadratic);

	lights.push_back(light);
}

void loadSettings()
{
	std::ifstream infile("graphic_effects/settings.txt");

	int width, height;
	int rayTraceDepth;
	bool shadows, reflections, refractions;
	int antiAliasing;

	infile >> width >> height >> rayTraceDepth >> shadows >> reflections >> refractions >> antiAliasing;

	settings = Settings(width, height, rayTraceDepth, shadows, reflections, refractions, static_cast<AntiAliasing>(antiAliasing));
}

/*
*	Handles the keyboard input
*/
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, GL_TRUE);
	}

	//set keyboard keys
	if (action == GLFW_PRESS)
	{
		keyboard.keys[key] = true;
	}
	else if (action == GLFW_RELEASE)
	{
		keyboard.keys[key] = false;
	}
}

/*
*	Handles the mouse movement
*/
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	//Set mouse values
	mouse.activated = true;
	mouse.xPos = (float)xpos;
	mouse.yPos = (float)ypos;
}

/*
*	Process the user input
*/
void processUserInput()
{
	//Settings controls
	float time = (float)glfwGetTime();
	if (time - keyboard.keyTimer > 0.2f)
	{
		keyboard.keyTimer = time;

		if (keyboard.keys[GLFW_KEY_KP_1])
		{
			if (settings.shadows)
			{
				settings.shadows = false;
				std::cout << "Shadows: OFF" << std::endl;
			}
			else
			{
				settings.shadows = true;
				std::cout << "Shadows: ON" << std::endl;
			}
		}

		if (keyboard.keys[GLFW_KEY_KP_2])
		{
			if (settings.reflections)
			{
				settings.reflections = false;
				std::cout << "Reflections: OFF" << std::endl;
			}
			else
			{
				settings.reflections = true;
				std::cout << "Reflections: ON" << std::endl;
			}
		}

		if (keyboard.keys[GLFW_KEY_KP_3])
		{
			if (settings.refractions)
			{
				settings.refractions = false;
				std::cout << "Refractions: OFF" << std::endl;
			}
			else
			{
				settings.refractions = true;
				std::cout << "Refractions: ON" << std::endl;
			}
		}

		if (keyboard.keys[GLFW_KEY_KP_4])
		{
			settings.antiAliasing = NONE;
			std::cout << "AntiAliasing: NONE" << std::endl;
		}

		if (keyboard.keys[GLFW_KEY_KP_5])
		{
			settings.antiAliasing = SUPERSAMPLE;
			std::cout << "AntiAliasing: SUPERSAMPLE" << std::endl;
		}

		if (keyboard.keys[GLFW_KEY_KP_6])
		{
			settings.antiAliasing = STOCHASTIC;
			std::cout << "AntiAliasing: STOCHASTIC" << std::endl;
		}
	}

	// Camera controls
	if (keyboard.keys[GLFW_KEY_W])
		scene.camera->processKeyboard(FORWARD, scene.deltaTime);
	if (keyboard.keys[GLFW_KEY_S])
		scene.camera->processKeyboard(BACKWARD, scene.deltaTime);
	if (keyboard.keys[GLFW_KEY_A])
		scene.camera->processKeyboard(LEFTWARD, scene.deltaTime);
	if (keyboard.keys[GLFW_KEY_D])
		scene.camera->processKeyboard(RIGHTWARD, scene.deltaTime);

	if (mouse.activated)
	{
		if (mouse.first)
		{
			mouse.lastX = mouse.xPos;
			mouse.lastY = mouse.yPos;
			mouse.first = false;
		}

		GLfloat xoffset = mouse.xPos - mouse.lastX;
		GLfloat yoffset = mouse.lastY - mouse.yPos;

		mouse.lastX = mouse.xPos;
		mouse.lastY = mouse.yPos;

		scene.camera->processMouseMovement(xoffset, yoffset);

		mouse.activated = false;
	}
}