# Real-time Ray Tracing

My final project submitted at the University of East Anglia.

The goal of this project was to research and implement a real-time ray tracer on consumer-level hardware.

This repository includes all the code and a 53 page technical report.

Notable features:

  - Shadows, Reflections and Refractions
  - Schlick's approximation
  - Supersampling and stochastic anti-aliasing 
  - Surface Area Heuristic KD-Tree
  - Diffuse, Specular and Bump Mapping
  - .OBJ model support
  - GPU Parallel Processing using CUDA

 
### Controls

Camera controls
  - W - forward
  - S - backward
  - A - left
  - D - right

Effects toggle (numpad)
  - 1 - shadows
  - 2 - reflections
  - 3 - refractions
  - 4 - no anti-aliasing
  - 5 - supersampling
  - 6 - stochastic sampling

### Scene Files

Scene files can be loaded in by specifying the file name in raytracer/scene_files/scene_layout/scene.txt

A guide to the formatting of the scene files can be found in raytracer/scene_files/scene_layout/guide.txt

### Installation

This project requires Visual C++ Redistributable for Visual Studio 2015, CUDA 7.5 and OpenGL 3.5 support:

Download links:

Visual C++ Redistributable for Visual Studio 2015
https://www.microsoft.com/en-gb/download/details.aspx?id=48145

CUDA 7.5
https://developer.nvidia.com/cuda-downloads
