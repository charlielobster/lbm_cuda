# lbm_cuda
Lattice Boltzmann Method Using CUDA

Software investigated:
* Microsoft Visual Studio Community 2017 and 2019
* NVIDIA GPU Computing Toolkit (CUDA) v10.2.89 441.22 for Windows 10
* NVIDIA Corporation's CUDA Samples v10.2 
* LATTICE BOLTZMANN SIMULATOR GPU accelerated with CUDA 
  by Tom Scherlis and Henry Friedlander (2017) 
* NVIDIA Nsight for Visual Studio Community 2019

About NVIDIA CUDA Samples v10.2:

The CUDA Samples is a collection of 176 projects spread across several subject areas including Graphics, Finance, Simulation, and Imaging. Mostly C with some C++ coding styles used, many of the samples are fairly short and digestible. There is often brief commented discussion of best-practices for structure and style. Given its instructional nature, the CUDA Samples are a pretty good standard for GPU-related development best-practices for this project. 

Thus, as a preliminary for this project, all the CUDA v10.2 samples were compiled and built using both 2017 and 2019 versions of Visual Studio Community Edition. 173 projects were built without errors, and there were 6 errors spread across 3 projects. Two of these projects related to missing Vulkan libraries, the other error is a heap exception.  

CUDA Samples v10.2 Undeclared Identifier and Other Errors reported by Visual Studio Intellisense

Both the 2017 and 2019 versions of the Visual Studio IDE's Intellisense found undeclared identifier errors in almost all samples for CUDA-specific objects (threadIdx, blockIdx, some CUDA functions, '<<<' and '>>>' operators, references to tex2D, group_operator, etc) in both 2019 and 2017 IDEs. Short of turning off Intellisense altogether, the best you can do to fix these errors is with the inclusion of the following header files:

* #include <cuda.h>
* #include <cuda_runtime.h>
* #include <device_launch_parameters.h> 

However, this does not fix all the errors Intellisense finds. The '<<<', '>>>' operators and some other references, such as 'tex2D<T>', 'group_operator' and other symbols continue to be a nuisance and have no affect on the build.

About lbm_cuda

lbm_cuda is a Visual Studio 2019 solution created with a default CUDA 10.2 project. All projects in the solution are set to Debug configuration. Most of the conventions used by CUDA Samples for v10.2 will be used for these projects. They include:

* additional include directories ./common/inc
* lib files location ./common/lib 
* output directories for executables ./bin/win64/Debug, 
* three dlls are required in ./bin/win64/Debug to avoid run-time errors:
	1. glew64.dll
	1. freeglut.dll

The fluidsGL CUDA Samples v10.2 project was added to the project space. Then the lbm project was created from a another default CUDA v10.2 project. A CUDA source file (.cu) was created called lbm.cu and was initially taken whole from the following example:

* LATTICE BOLTZMANN SIMULATOR GPU accelerated with CUDA 
* by Tom Scherlis and Henry Friedlander (2017) 

Settings for the lbm project were then configured to match conventions in the CUDA Samples v10.2 projects. glew64.lib was added to the Additional Dependencies for lbm (in Solution Explorer right-click Project, select Properties. Configuration Properties->Linker->Input Additional Dependencies).

The unmodified version of Sherlis and Friedlander's code was written as a single .cu implementation file 1167 lines long. There are several global variables and definitions shared between CPU and GPU threads. For this project, the code was first refactored into multiple files with a kernel-focused structure. In addition to providing some improved organization, these steps were also helpful in getting a better understanding of the underlying structure of the program. Some of these changes include:

* Defines and structs shared between the CPU and GPU were split into a separate header file. 
* Global variables were moved into a new .cpp file and references to those variables marked extern in the .cu file. 
* All CPU-related functions were then moved to the .cpp file. 
* GPU entry points were declared as extern function prototypes in the .cpp file, and then defined in the .cu file. 
* Two new GPU-related functions were created for handling initialization and cleanup in the new, more modular file structure. 
* Original comments were edited and the code reformatted. 



