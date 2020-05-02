# lbm_cuda
Lattice Boltzmann Methods Using CUDA

lbm_cuda is a Visual Studio 2019 solution created with a default Cuda 10.2 project.
All projects in this space are set to Debug configuration.
Most of the conventions used by Nvidia Corporation's Cuda Samples for v10.2 were used for the project space. 

These include:
	additional include directories ./common/inc
	lib files location ./common/lib 
	output directories for executables ./bin/win64/Debug, 
	three dlls are required in ./bin/win64/Debug to avoid run-time errors:
		glew64.dll
		freeglut.dll
		FreeImage.dll

The fluidsGL sample project from Nvidia's Cuda Samples was added to the project space.
Then the lbm project was created from a another default Cuda 10.2 project.
The code was taken from 
LATTICE BOLTZMANN SIMULATOR GPU accelerated with CUDA 
by Tom Scherlis and Henry Friedlander (2017) 
Settings for lbm were configured to match Cuda Samples for v10.2.
glew64.lib was added to the Additional Dependencies for lbm in the project linker settings.

