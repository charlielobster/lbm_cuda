# lbm_cuda
Lattice Boltzmann Method Using CUDA

lbm_cuda is a Visual Studio 2019 solution created with a default CUDA 10.2 project. All projects in the solution are set to Debug configuration mode. Most of the conventions used by CUDA Samples for v10.2 will be used for these projects. They include:

* additional include directories ./common/inc
* lib files location ./common/lib 
* output directories for executables ./bin/win64/Debug, 
* two dlls are required in ./bin/win64/Debug to avoid run-time errors:
	1. glew64.dll
	1. freeglut.dll


The lbm project is taken and modified from Tom Scherlis and Henry Friedlander's "Lattice Boltzmann Simulator GPU accelerated with CUDA" codebase.

