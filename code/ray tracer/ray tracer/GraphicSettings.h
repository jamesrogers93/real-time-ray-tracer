#ifndef _GraphicSettings_h
#define _GraphicSettings_h

enum AntiAliasing
{
	NONE = 0, SUPERSAMPLE = 1, STOCHASTIC = 2
};

struct Settings
{
	//Width and height of the window
	int width, height;

	//Ray tracing recursion level
	int rayTraceDepth;

	//Raytrace effects
	bool shadows, reflections, refractions;

	//Anti Aliasing
	AntiAliasing antiAliasing;

	Settings(){}

	Settings(int width, int height, int depth, bool shadows, bool reflections, bool refractions, AntiAliasing antiAliasing) : width(width), height(height), rayTraceDepth(depth), shadows(shadows), reflections(reflections), refractions(refractions), antiAliasing(antiAliasing)
	{
	}
};

#endif