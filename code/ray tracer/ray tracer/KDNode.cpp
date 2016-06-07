#include "KDTree.h"

namespace KDTree
{

	KDNode::KDNode() : isLeaf(false)
	{

	}

	KDNode::KDNode(int *triangleIndices, size_t numTriangleIndices, int *sphereIndices, size_t numSphereIndices, Voxel voxel) : isLeaf(true)
	{

		//TRIANGLES//
		this->numTriangleIndices = numTriangleIndices;
		if (numTriangleIndices > 0)
		{
			this->triangleIndices = triangleIndices;
			this->numTriangleIndicesBytes = numTriangleIndices * sizeof(int);

			//Allocate and copy triangle index data onto the device
			gpuErrchk(cudaMalloc(&this->d_triangleIndices, this->numTriangleIndicesBytes));
			gpuErrchk(cudaMemcpy(this->d_triangleIndices, this->triangleIndices, this->numTriangleIndicesBytes, cudaMemcpyHostToDevice));
		}

		//SPHERES//
		this->numSphereIndices = numSphereIndices;
		if (numSphereIndices > 0)
		{
			this->sphereIndices = sphereIndices;
			this->numSphereIndicesBytes = numSphereIndices * sizeof(int);

			//Allocate and copy spheres index data onto the device
			gpuErrchk(cudaMalloc(&this->d_sphereIndices, this->numSphereIndicesBytes));
			gpuErrchk(cudaMemcpy(this->d_sphereIndices, this->sphereIndices, this->numSphereIndicesBytes, cudaMemcpyHostToDevice));
		}

		//VOXEL//
		this->voxel = voxel;

		//TRIANGLE TEXTURE TEST//

		//Allcoate and copy data
		triangleTextureIndices = (float *)malloc(numTriangleIndices * sizeof(float));
		for (size_t i = 0; i < numTriangleIndices; i++)
		{
			triangleTextureIndices[i] = triangleIndices[i];
		}

		//Allocate and copy data to device
		gpuErrchk(cudaMalloc(&d_triangleTextureIndices, numTriangleIndices * sizeof(float)));
		gpuErrchk(cudaMemcpy(d_triangleTextureIndices, triangleTextureIndices, numTriangleIndices * sizeof(float), cudaMemcpyHostToDevice));

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeLinear;
		resDesc.res.linear.devPtr = d_triangleTextureIndices;
		resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
		resDesc.res.linear.desc.x = 32; // bits per channel
		resDesc.res.linear.sizeInBytes = numTriangleIndices * sizeof(float);

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.readMode = cudaReadModeElementType;

		// create texture object: we only have to do this once!
		triangleTextureIndicesTexture = 0;
		cudaCreateTextureObject(&triangleTextureIndicesTexture, &resDesc, &texDesc, NULL);

	}

	KDNode::KDNode(Plane plane, KDNode *left, KDNode *right, Voxel voxel) : isLeaf(false)
	{
		this->plane = plane;
		this->left = left;
		this->right = right;
		this->voxel = voxel;

		//Store left and right nodes on the device
		gpuErrchk(cudaMalloc(&d_left, sizeof(KDNode)));
		gpuErrchk(cudaMemcpy(d_left, left, sizeof(KDNode), cudaMemcpyHostToDevice));

		gpuErrchk(cudaMalloc(&d_right, sizeof(KDNode)));
		gpuErrchk(cudaMemcpy(d_right, right, sizeof(KDNode), cudaMemcpyHostToDevice));
	}

	KDNode* KDNode::buildTree(Triangle *triangles, size_t &numTriangles, Sphere *spheres, size_t &numSpheres)
	{
		//Prepare tree to be built

		//Allocate storge for triangle and spheres indices
		int *triangleIndices = (int *)malloc(numTriangles * sizeof(int));
		int *sphereIndices = (int *)malloc(numSpheres * sizeof(int));

		//Fill indices array with indices (0 - n)
		for (int i = 0; i < numTriangles; i++)
			triangleIndices[i] = i;

		for (int i = 0; i < numSpheres; i++)
			sphereIndices[i] = i;

		//Create voxel which encompasses all objects in the scene
		Voxel voxel = Voxel(triangles, triangleIndices, numTriangles, spheres, sphereIndices, numSpheres);

		return build(triangles, triangleIndices, numTriangles, spheres, sphereIndices, numSpheres, voxel, Plane(), 0);
	}
	
	KDNode* KDNode::build(Triangle *triangles, int *triangleIndices, size_t &numTriangleIndices, Sphere *spheres, int *sphereIndices, size_t &numSphereIndices, Voxel&voxel, Plane &plane, int depth)
	{
		float CP;
		Plane P;
		PlaneSide PS;
		findPlane(triangles, triangleIndices, numTriangleIndices, spheres, sphereIndices, numSphereIndices, voxel, CP, P, PS);

		//Exit early and return leaf node if some condition is met
		size_t totalPrims = numTriangleIndices + numSphereIndices;
		if (terminate(totalPrims, CP) || P == plane)
			return new KDNode(triangleIndices, numTriangleIndices, sphereIndices, numSphereIndices, voxel);

		//Use plane to split voxel
		Voxel leftVoxel, rightVoxel;
		splitVoxel(P, voxel, leftVoxel, rightVoxel);

		//Split Primitives

		//Split triangles to fill left and right voxels
		std::vector<int> t_triangleIndicesLeft, t_triangleIndicesRight;
		size_t numTrianglesLeft = 0, numTrianglesRight = 0;
		splitTriangles(triangles, triangleIndices, numTriangleIndices, t_triangleIndicesLeft, numTrianglesLeft, t_triangleIndicesRight, numTrianglesRight, P, PS);

		//Split spheres to fill left and right voxels
		std::vector<int> t_sphereIndicesLeft, t_sphereIndicesRight;
		size_t numSpheresLeft = 0, numSpheresRight = 0;
		splitSpheres(spheres, sphereIndices, numSphereIndices, t_sphereIndicesLeft, numSpheresLeft, t_sphereIndicesRight, numSpheresRight, P, PS);

		//Convert triangle vectors to arrays
		//Left
		size_t numTriangleIndicesBytes = numTrianglesLeft * sizeof(int);
		int *triangleIndicesLeft = (int *)malloc(numTriangleIndicesBytes);
		memcpy(triangleIndicesLeft, t_triangleIndicesLeft.data(), numTriangleIndicesBytes);

		//Right
		numTriangleIndicesBytes = numTrianglesRight * sizeof(int);
		int *triangleIndicesRight = (int *)malloc(numTriangleIndicesBytes);
		memcpy(triangleIndicesRight, t_triangleIndicesRight.data(), numTriangleIndicesBytes);
		
		//Convert Sphere vectors to arrays
		//Left
		size_t numSphereIndicesBytes = numSpheresLeft * sizeof(int);
		int *sphereIndicesLeft = (int *)malloc(numSphereIndicesBytes);
		memcpy(sphereIndicesLeft, t_sphereIndicesLeft.data(), numSphereIndicesBytes);

		//Right
		numSphereIndicesBytes = numSpheresRight * sizeof(int);
		int *sphereIndicesRight = (int *)malloc(numSphereIndicesBytes);
		memcpy(sphereIndicesRight, t_sphereIndicesRight.data(), numSphereIndicesBytes);

		return new KDNode(P, build(triangles, triangleIndicesLeft, numTrianglesLeft, spheres, sphereIndicesLeft, numSpheresLeft, leftVoxel, P, depth + 1), build(triangles, triangleIndicesRight, numTrianglesRight, spheres, sphereIndicesRight, numSpheresRight, rightVoxel, P, depth + 1), voxel);
	}

	void KDNode::findPlane(Triangle *triangles, int *triangleIndices, size_t &numTriangleIndices, Sphere *spheres, int *sphereIndices, size_t &numSphereIndices, Voxel &voxel, float &CP, Plane &P, PlaneSide &PS)
	{
		//Set cost probability to infinity
		CP = INFINITY;

		//Temporary cost probability
		float t_CP;

		//Temporary Triangle
		Triangle *triangle;

		//Temporary Sphere
		Sphere *sphere;

		//Temporary plane
		Plane t_plane;

		//Temporary voxel
		Voxel t_voxel;

		//Number of primitives left, right and on plane
		size_t numLeft, numPlane, numRight;

		//Event list
		std::vector<Event> events;
		size_t numEvents;

		//Loop over all dimensions
		for (int k = 0; k < 3; k++)
		{
			//Clear events
			events.clear();
			numEvents = 0;

			//Find all events for triangles
			for (size_t i = 0; i < numTriangleIndices; i++)
			{
				//Get vertices from triangle
				glm::vec3 V0, V1, V2;

				//Get first Triangle
				int index = triangleIndices[i];
				triangle = &triangles[index];

				//Get vertices of triangle
				V0 = triangle->V0.position;
				V1 = triangle->V1.position;
				V2 = triangle->V2.position;

				//Clip triangle to box
				t_voxel.clipVerticesToVoxel(V0, V1, V2, voxel);

				//Store events
				if (t_voxel.isPerpendicular(k))
				{
					events.push_back(Event(t_voxel.min[k], Event::PLANAR)); numEvents++;
				}
				else
				{
					events.push_back(Event(t_voxel.min[k], Event::START)); numEvents++;
					events.push_back(Event(t_voxel.max[k], Event::END)); numEvents++;
				}
			}

			//Find all events for spheres
			for (size_t i = 0; i < numSphereIndices; i++)
			{
				//Clip sphere to box
				sphere = &spheres[i];
				t_voxel.clipSphereToVoxel(sphere->origin, sphere->radius, voxel);

				//Store events
				if (t_voxel.isPerpendicular(k))
				{
					events.push_back(Event(t_voxel.min[k], Event::PLANAR)); numEvents++;
				}
				else
				{
					events.push_back(Event(t_voxel.min[k], Event::START)); numEvents++;
					events.push_back(Event(t_voxel.max[k], Event::END)); numEvents++;
				}
			}

			//Sort events into ascending plane order. If planes are equal, order by event type
			sort(events.begin(), events.end());

			//Start with all triangles to the right
			numLeft = 0; numPlane = 0; numRight = numTriangleIndices + numSphereIndices;

			//Iteratively sweep planes to find best splitting plane 
			for (size_t i = 0; i < numEvents; i++)
			{
				//create plane using dimension, k, and plane position stored in events
				t_plane = Plane(k, events[i].pe);

				int planar = 0, start = 0, end = 0;

				//Ending on plane
				while (i < numEvents && events[i].pe == t_plane.pe && events[i].type == Event::END)
				{
					end++;
					i++;
				}

				//Lying on plane
				while (i < numEvents && events[i].pe == t_plane.pe && events[i].type == Event::PLANAR)
				{
					planar++;
					i++;
				}

				//Starting on plane
				while (i < numEvents && events[i].pe == t_plane.pe && events[i].type == Event::START)
				{
					start++;
					i++;
				}

				//Decrement i to account for the double increment
				i--;

				numPlane = planar;
				numRight -= planar;
				numRight -= end;

				//Find smallest cost probability using SAH
				PlaneSide t_PS = UNKNOWN;
				SAH(t_plane, voxel, numLeft, numRight, numPlane, t_CP, t_PS);
				if (t_CP < CP)
				{
					CP = t_CP;
					P = t_plane;
					PS = t_PS;
				}

				numLeft += start;
				numLeft += planar;
				numPlane = 0;
			}
		}
	}

	void KDNode::SAH(Plane &plane, Voxel &voxel, size_t &numLeft, size_t &numRight, size_t &numPlane, float &Cp, PlaneSide &Pside)
	{
		//Calculate left and right voxel, vl, vr, given plane P
		Voxel voxelLeft, voxelRight;
		splitVoxel(plane, voxel, voxelLeft, voxelRight);

		//Probabilities, Pl, Pr, of ray hitting child voxel given parent was hit
		//Probabiility subVoxel (Pl, Pr)
		float Pl, Pr;
		Pl = voxelLeft.SA / voxel.SA;
		Pr = voxelRight.SA / voxel.SA;

		//Compute cost of sub dividing voxel with plane. 
		//Twice to test probability with planar triangles in left and right voxels 
		//Cost Probability Voxel (CPl, CPr)
		float CPl, CPr;
		CPl = C(Pl, Pr, numLeft + numPlane, numRight);
		CPr = C(Pl, Pr, numLeft, numRight + numPlane);

		//Return smallest cost probability and side of plane
		if (CPl < CPr)
		{
			Cp = CPl;
			Pside = LEFT;
		}
		else
		{
			Cp = CPr;
			Pside = RIGHT;
		}
	}

	float KDNode::C(float &Pl, float &Pr, int numLeft, int numRight)
	{
		return (lambda(numLeft, numRight) * (KT + (KI * ((Pl * numLeft) + (Pr * numRight)))));
	}

	float KDNode::lambda(int &numLeft, int &numRight)
	{
		if (numLeft == 0 || numRight == 0)
			return 0.8f;
		return 1.0f;
	}

	void KDNode::splitVoxel(Plane &plane, Voxel &voxel, Voxel &voxelLeft, Voxel &voxelRight)
	{
		//Set left and right voxel to parent voxel
		voxelLeft = voxelRight = voxel;

		//Split voxels along plane
		voxelLeft.max[plane.pk] = plane.pe;
		voxelRight.min[plane.pk] = plane.pe;

		//Recalculate left and right voxel's surface area
		voxelLeft.calculateSA();
		voxelRight.calculateSA();
	}

	bool KDNode::terminate(size_t &numPrims, float &CP)
	{
		return CP > (KI * numPrims);
	}

	void KDNode::splitTriangles(Triangle *triangles, int *triangleIndices, size_t &numTriangleIndices, std::vector<int> &triangleIndicesLeft, size_t &numTrianglesLeft, std::vector<int> &triangleIndicesRight, size_t &numTrianglesRight, Plane &P, PlaneSide &PS)
	{
		//Set number of triangles to 0
		numTrianglesLeft = 0; numTrianglesRight = 0;

		glm::vec3 V0, V1, V2;
		Voxel t_voxel;
		int index;
		Triangle *triangle;

		//Iterate over all triangles
		for (size_t i = 0; i < numTriangleIndices; i++)
		{
			index = triangleIndices[i];

			//Get triangle
			triangle = &triangles[index];

			//Get vertices of triangle
			V0 = triangle->V0.position;
			V1 = triangle->V1.position;
			V2 = triangle->V2.position;

			t_voxel = Voxel(V0, V1, V2);

			//If triangle lies on plane, assign to optimal node
			if (t_voxel.min[P.pk] == P.pe && t_voxel.max[P.pk] == P.pe)
			{
				if (PS == LEFT)
				{
					triangleIndicesLeft.push_back(index);
					numTrianglesLeft++;
				}
				else if (PS == RIGHT)
				{
					triangleIndicesRight.push_back(index);
					numTrianglesRight++;
				}
			}
			else
			{
				//Triangles that overlap plane are stored in both left and right nodes

				//Place triangle that is on left of plane to left node
				if (t_voxel.min[P.pk] < P.pe)
				{
					triangleIndicesLeft.push_back(index);
					numTrianglesLeft++;
				}

				//Place triangle that is on right of plane to right node
				if (t_voxel.max[P.pk] > P.pe)
				{
					triangleIndicesRight.push_back(index);
					numTrianglesRight++;
				}
			}
		}
	}

	void KDNode::splitSpheres(Sphere *spheres, int *sphereIndices, size_t &numSphereIndices, std::vector<int> &sphereIndicesLeft, size_t &numSpheresLeft, std::vector<int> &sphereIndicesRight, size_t &numSpheresRight, Plane &P, PlaneSide &PS)
	{
		//Set number of triangles to 0
		numSpheresLeft = 0; numSpheresRight = 0;

		Voxel t_voxel;
		int index;
		Sphere *sphere;

		//Iterate over all triangles
		for (size_t i = 0; i < numSphereIndices; i++)
		{
			index = sphereIndices[i];

			//Get triangle
			sphere = &spheres[index];

			//Get vertices of triangle
			t_voxel = Voxel(sphere->origin, sphere->radius);

			//If triangle lies on plane, assign to optimal node
			if (t_voxel.min[P.pk] == P.pe && t_voxel.max[P.pk] == P.pe)
			{
				if (PS == LEFT)
				{
					sphereIndicesLeft.push_back(index);
					numSpheresLeft++;
				}
				else if (PS == RIGHT)
				{
					sphereIndicesRight.push_back(index);
					numSpheresRight++;
				}
			}
			else
			{
				//Triangles that overlap plane are stored in both left and right nodes

				//Place triangle that is on left of plane to left node
				if (t_voxel.min[P.pk] < P.pe)
				{
					sphereIndicesLeft.push_back(index);
					numSpheresLeft++;
				}

				//Place triangle that is on right of plane to right node
				if (t_voxel.max[P.pk] > P.pe)
				{
					sphereIndicesRight.push_back(index);
					numSpheresRight++;
				}
			}
		}
	}

	void KDNode::deleteMem()
	{
		//Clear triangle data if node is a leaf
		if (isLeaf)
		{
			//Delete triangle and sphere indices on the host & device
			if (numTriangleIndices > 0)
			{
				free(triangleIndices);
				gpuErrchk(cudaFree(d_triangleIndices));

			}
			if (numSphereIndices > 0)
			{
				free(sphereIndices);
				gpuErrchk(cudaFree(d_sphereIndices));
			}	
		}
		else
		//Clear child node data if node is not a leaf
		{
			//Delete left and right node on the host & device
			left->deleteMem();
			free(left);
			gpuErrchk(cudaFree(d_left));

			right->deleteMem();
			free(right);
			gpuErrchk(cudaFree(d_right));
		}
	}
}