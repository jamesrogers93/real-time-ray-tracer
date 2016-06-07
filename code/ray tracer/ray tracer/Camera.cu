#include "Camera.h"

Camera::Camera(int screenWidth, int screentHeight, float fov, glm::vec3 origin, glm::vec3 up, float yaw, float pitch) : front(glm::vec3(0.0f, 0.0f, -1.0f)), movementSpeed(SPEED), mouseSensitivity(SENSITIVTY)
{
	//Allocate camera data on host and device
	sizeOfCudaCamera = sizeof(CudaCamera);
	cameraData = (CudaCamera *)malloc(sizeOfCudaCamera);

	this->cameraData->origin = origin;
	this->cameraData->width = screenWidth;
	this->cameraData->height = screentHeight;
	this->cameraData->invWidth = 1 / (float)screenWidth;
	this->cameraData->invHeight = 1 / (float)screentHeight;

	this->worldUp = up;
	this->yaw = yaw;
	this->pitch = pitch;

	//Update the camera vectors
	this->updateCameraVectors();

	//Calculate the view matrix
	cameraData->viewMatrix = getViewMatrix();

	//Calculate aspect ratio
	cameraData->aspectRatio = calculateAspectRatio();

	cameraData->scale = tan(fov * 0.5f);

	//initTexture(cameraData->scale, cameraData->aspectRatio);

	//Allocate and copy data to device
	gpuErrchk(cudaMalloc(&d_cameraData, sizeOfCudaCamera));
	gpuErrchk(cudaMemcpy(d_cameraData, cameraData, sizeOfCudaCamera, cudaMemcpyHostToDevice));
}

float Camera::calculateAspectRatio()
{

	//Check if width and height have been set
	if (cameraData->width == 0)
	{
		std::cout << "ERROR: Aspect Ratio can not be calculated, width has not been set" << std::endl;
		exit(0);
	}

	if (cameraData->height == 0)
	{
		std::cout << "ERROR: Aspect Ratio can not be calculated, height has not been set" << std::endl;
		exit(0);
	}


	//If width is greater than height, divide w by h. Otherwise aspectratio will be negative
	if (cameraData->width > cameraData->height)
	{
		return (float)cameraData->width / (float)cameraData->height;
	}

	//Same as above but with fliped arguments
	if (cameraData->width < cameraData->height)
	{
		return (float)cameraData->height / (float)cameraData->width;
	}

	//If the values are the same, return 1.0f
	return 1.0f;
}

/*void Camera::initTexture(float scale, float aspectRatio)
{
	//Create data array to copy to cuda
	float4 *data = (float4 *)malloc(sizeof(float4));

	//Fill array with camera data
	data[0].x = cameraData->invWidth;
	data[0].y = cameraData->invHeight;
	data[0].z = scale;
	data[0].w = aspectRatio;

	//Allocate cuda array and copy camera data to device
	gpuErrchk(cudaMalloc(&cameraData->d_data, sizeof(float4)));
	gpuErrchk(cudaMemcpy(cameraData->d_data, data, sizeof(float4), cudaMemcpyHostToDevice));

	//Create texture object
	cudaResourceDesc camera_resDesc;
	memset(&camera_resDesc, 0, sizeof(camera_resDesc));
	camera_resDesc.resType = cudaResourceTypeLinear;
	camera_resDesc.res.linear.devPtr = cameraData->d_data;
	camera_resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	camera_resDesc.res.linear.desc.x = 32;
	camera_resDesc.res.linear.desc.y = 32;
	camera_resDesc.res.linear.desc.z = 32;
	camera_resDesc.res.linear.desc.w = 32;
	camera_resDesc.res.linear.sizeInBytes = sizeof(float4);

	cudaTextureDesc camera_texDesc;
	memset(&camera_texDesc, 0, sizeof(camera_texDesc));
	camera_texDesc.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&cameraData->d_dataTexture, &camera_resDesc, &camera_texDesc, NULL);

	//constant memory
	//Copy data to cuda constant memory
	gpuErrchk(cudaMemcpyToSymbol(cd_DATA, data, sizeof(float4)));

	//Make cd_data point ot cuda constant memory
	//cameraData->cd_data = &cd_DATA;
}*/

void Camera::processKeyboard(Camera_Movement direction, float deltaTime)
{
	float velocity = this->movementSpeed * deltaTime;
	if (direction == FORWARD)
		this->cameraData->origin -= this->front * velocity;
	if (direction == BACKWARD)
		this->cameraData->origin += this->front * velocity;
	if (direction == LEFTWARD)
		this->cameraData->origin -= this->right * velocity;
	if (direction == RIGHTWARD)
		this->cameraData->origin += this->right * velocity;
}

void Camera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch)
{
	xoffset *= this->mouseSensitivity;
	yoffset *= this->mouseSensitivity;

	this->yaw -= xoffset;
	this->pitch -= yoffset;

	// Make sure that when pitch is out of bounds, screen doesn't get flipped
	if (constrainPitch)
	{
		if (this->pitch > 89.0f)
			this->pitch = 89.0f;
		if (this->pitch < -89.0f)
			this->pitch = -89.0f;
	}

	// Update Front, Right and Up Vectors using the updated Eular angles
	this->updateCameraVectors();
}

void Camera::processMouseScroll(float yoffset)
{
}

void Camera::updateCamera()
{
	updateCameraVectors();
	cameraData->viewMatrix = getViewMatrix();

	//Copy data to device
	gpuErrchk(cudaMemcpy(d_cameraData, cameraData, sizeOfCudaCamera, cudaMemcpyHostToDevice));
}

void Camera::updateCameraVectors()
{
	//Calculate the new Front vector
	glm::vec3 front;
	front.x = cos(glm::radians(this->yaw)) * cos(glm::radians(this->pitch));
	front.y = sin(glm::radians(this->pitch));
	front.z = sin(glm::radians(this->yaw)) * cos(glm::radians(this->pitch));

	this->front = glm::normalize(front);
	this->right = glm::normalize(glm::cross(this->front, this->worldUp));
	this->up = glm::normalize(glm::cross(this->right, this->front));
}

glm::mat4 Camera::getViewMatrix()
{
	//Look in its own direction from its own position
	return glm::lookAt(cameraData->origin, cameraData->origin + front, up);
}