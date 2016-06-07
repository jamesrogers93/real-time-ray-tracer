#ifndef _KDTree_h
#define _KDTree_h

//STD
#include <vector>
#include <algorithm>

//GLM
#include <glm/glm.hpp>

//Raytracer
#include "Triangle.h"
#include "Sphere.h"

//Tools
#include "Debug.h"

namespace KDTree
{
#define T_TARGET 2
#define V_Target 0.02f
#define KT 1.0f
#define KI 1.5f
#define T_SLACK 0.001f

	enum PlaneSide
	{
		LEFT = -1, RIGHT = 1, UNKNOWN = 0
	};

	struct Plane
	{
		//dimension
		short pk;
		//point
		float pe;

		Plane(short pk = -1, float pe = 0.0f)
		{
			this->pk = pk;
			this->pe = pe;
		}

		inline bool operator==(const Plane& p) const {
			return (pk == p.pk && pe == p.pe);
		}
	};

	struct Event
	{
		//Type of event
		typedef enum
		{
			END = 0, PLANAR = 1, START = 2
		} EventType;

		EventType type;

		//Position of plane 
		float pe;

		Event(float pe, EventType type)
		{
			this->pe = pe;
			this->type = type;
		}

		inline bool operator<(const Event& e) const
		{
			return((pe < e.pe) || (pe == e.pe && type < e.type));
		}
	};

	class KDNode;

	struct Stack
	{
		KDNode *node;
		float tMin;
		float tMax;
	};

	/*
	*	KDNode voxel class
	*/
	class Voxel
	{
	public:

		//Minimum and maximum extents
		glm::vec3 min, max;

		//Surface area
		float SA;

		//Constructors
		Voxel();

		Voxel(Triangle *triangles, int *triangleIndices, size_t numTriangleIndices, Sphere *spheres, int *sphereIndices, size_t numSphereIndices);

		Voxel(glm::vec3 &V0, glm::vec3 &V1, glm::vec3 &V2);

		Voxel(glm::vec3 &origin, float &radius);

		//Initalising methods
		void initVoxelTrianglesSpheres(Triangle *triangles, int *triangleIndices, size_t numTriangleIndices, Sphere *spheres, int *sphereIndices, size_t numSphereIndices);

		void initVoxelVertices(glm::vec3 &V0, glm::vec3 &V1, glm::vec3 &V2);

		void initVoxelSphere(glm::vec3 &origin, float &radius);

		//Misc
		void calculateSA();

		void clipVerticesToVoxel(glm::vec3 &V0, glm::vec3 &V1, glm::vec3 &V2, Voxel &voxel);
		
		void clipSphereToVoxel(glm::vec3 &origin, float &radius, Voxel &V);

		bool isPerpendicular(int &k);
	};

	/*
	*	KDNode class
	*/
	class KDNode
	{
	public:

		//Seperating plane
		Plane plane;

		//Left and right nodes
		KDNode *left, *d_left;
		KDNode *right, *d_right;

		//Node bounding volume
		Voxel voxel;

		//Indicates whether node is a leaf
		bool isLeaf;

		//Triangles
		int *triangleIndices, *d_triangleIndices;
		size_t numTriangleIndices;
		size_t numTriangleIndicesBytes;

		//Spheres
		int *sphereIndices, *d_sphereIndices;
		size_t numSphereIndices;
		size_t numSphereIndicesBytes;

		//TEXTURE TEST
		float *triangleTextureIndices, *d_triangleTextureIndices;
		size_t numTriangleTextureIndices;
		cudaTextureObject_t triangleTextureIndicesTexture;


		//Constructors
		KDNode();

		KDNode(int *triangleIndices, size_t numTriangleIndices, int *sphereIndices, size_t numSphereIndices, Voxel voxel);

		KDNode(Plane plane, KDNode *left, KDNode *right, Voxel voxel);

		//Build
		KDNode* buildTree(Triangle *triangles, size_t &numTriangles, Sphere *spheres, size_t &numSpheres);

		KDNode* build(Triangle *triangles, int *triangleIndices, size_t &numTriangleIndices, Sphere *spheres, int *sphereIndices, size_t &numSphereIndices, Voxel&voxel, Plane &plane, int depth);

		void findPlane(Triangle *triangles, int *triangleIndices, size_t &numTriangleIndices, Sphere *spheres, int *sphereIndices, size_t &numSphereIndices, Voxel &voxel, float &CP, Plane &P, PlaneSide &PS);

		void SAH(Plane &plane, Voxel &voxel, size_t &numLeft, size_t &numRight, size_t &numPlane, float &Cp, PlaneSide &Pside);
		
		float C(float &Pl, float &Pr, int Nl, int Nr);
		
		float lambda(int &Nl, int &Nr);

		void splitVoxel(Plane &plane, Voxel &voxel, Voxel &voxelLeft, Voxel &voxelRight);

		bool terminate(size_t &numPrims, float &CP);

		void splitTriangles(Triangle *triangles, int *triangleIndices, size_t &numTriangleIndices, std::vector<int> &triangleIndicesLeft, size_t &numTrianglesLeft, std::vector<int> &triangleIndicesRight, size_t &numTrianglesRight, Plane &P, PlaneSide &PS);

		void splitSpheres(Sphere *spheres, int *sphereIndices, size_t &numSphereIndices, std::vector<int> &sphereIndicesLeft, size_t &numSpheresLeft, std::vector<int> &sphereIndicesRight, size_t &numSpheresRight, Plane &P, PlaneSide &PS);

		//Traverse


		//Misc
		void deleteMem();

		int count()
		{
			int c = 1;

			if (!isLeaf)
			{
				c += left->count();
				c += right->count();
			}

			return c;
		}
	};
}
#endif
	