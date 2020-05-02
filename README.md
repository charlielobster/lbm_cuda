# lbm_cuda
Lattice Boltzmann Methods Using CUDA

cfd_cuda is a Visual Studio 2019 solution created with a default Cuda 10.2 project.
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

The fluidsGL sample project from the cuda samples was added to the project space.
