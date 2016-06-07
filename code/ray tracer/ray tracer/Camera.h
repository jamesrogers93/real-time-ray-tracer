#ifndef _Camera_h
#define _Camera_h

//CUDA
#include <cuda_runtime.h>

//GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

//STD
#include <iostream>

//Raytracer
#include "Debug.h"

// Default camera values
const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 6.0f;
const float SENSITIVTY = 0.20f;
const float FOV = glm::radians(45.0f);

//Constant memory test
//extern __constant__ 
//float4 cd_DATA[1];

enum Camera_Movement {
	FORWARD,
	BACKWARD,
	LEFTWARD,
	RIGHTWARD
};

//Struct of variables that will be used in cuda
struct CudaCamera
{

	//Camera origin
	glm::vec3 origin;

	//Viewing matrix
	glm::mat4 viewMatrix;

	int width, height;
	float invWidth, invHeight;
	float scale, aspectRatio;

	//(width, height), (scale. fov, aspectRatio)
	//cudaTextureObject_t d_dataTexture;
	//float4 *d_data;

};

//Constant memory test
//__device__ void getConstantMemData(float4 &data);

class Camera
{
public:

	CudaCamera *cameraData, *d_cameraData;
	size_t sizeOfCudaCamera;

	glm::vec3 front;
	glm::vec3 right;
	glm::vec3 up;
	glm::vec3 worldUp;

	// Eular Angles
	float yaw;
	float pitch;

	// Camera options
	float movementSpeed;
	float mouseSensitivity;

	Camera(){}

	Camera(int screenWidth, int screentHeight, float fov, glm::vec3 origin, glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH);
	
	void initOrigin();
	void initRays();
	void calculateRays();

	void updateCamera();
	void updateCameraVectors();
	glm::mat4 getViewMatrix();

	void processKeyboard(Camera_Movement direction, float deltaTime);
	void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);
	void processMouseScroll(float yoffset);

private:
	void initTexture(float scale, float aspectRatio);
	float calculateAspectRatio(void);
};

#endif