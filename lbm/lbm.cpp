/* ============================================================ */
/* LATTICE BOLTZMANN SIMULATOR                                  */
/* GPU accelerated with CUDA                                    */
/*                                                              */
/* Copyright (c) 2017 Tom Scherlis and Henry Friedlander        */
/* For SSA Physics 3                                            */
/* ============================================================ */

//comment out this line to hide prints:

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <malloc.h>

// OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>

//Cuda includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>

#include "lbm.h"

//-----------------------------------------------------------------------//
//                     GLOBAL VARS AND STRUCT DEFS                       //
//-----------------------------------------------------------------------//

// texture and pixel objects
GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource* cuda_pbo_resource;

//timing variables:
unsigned long last_draw_time = 0;
unsigned long current_draw_time = 0;
float delta_t = 1;

//GPU/CPU interop memory pointers:
unsigned char state = 0;
lbm_node* array1;
lbm_node* array2;
unsigned char* barrier;

int prex = -1;
int prey = -1;

char needsUpdate = 1;

parameter_set params;

char waitingForSpeed = 0;
char waitingForViscosity = 0;
char waitingForRate = 0;

float fps;

int current_button = GLUT_LEFT_BUTTON;

extern "C" void kernelLauncher(uchar4* image);
extern "C" void cudaExit();
extern "C" void initFluid_gpu(d2q9_node* d2q9);

void getParams(parameter_set* params)
{
	params->viscosity = 0.005;
	params->contrast = 75;
	params->v = 0.1;
	params->mode = Curl;
	params->height = 200;
	params->width = 300;
	params->stepsPerRender = 10;
}

//get 1d flat index from row and col
int getIndex_cpu(int x, int y)
{
	return y * params.width + x;
}


void clearBarriers()
{
	for (int i = 0; i < params.width; i++)
	{
		for (int j = 0; j < params.height; j++)
		{
			barrier[getIndex_cpu(i, j)] = 0;
		}
	}
}

void drawLineDiagonal()
{
	for (int i = 0; i < params.height / 4; i++)
	{

		barrier[getIndex_cpu((params.width / 3) + (i / 3), params.height / 3 + i)] = 1;
	}
}

void drawSquare()
{
	for (int i = 0; i < params.height / 4; i++)
	{

		for (int j = 0; j < params.height / 4; j++)
		{
			//if(i==0 || i== params.height / 4-1 || j==0 || j == params.height / 4-1)
			barrier[getIndex_cpu(i + params.width / 3, j + params.height * 3 / 8)] = 1;
		}

	}
}

//provide LBM constants for d2q9 style nodes
//assumes positive is up and right, whereas our program assumes positive down and right.
void init_d2q9(d2q9_node* d2q9)
{
	d2q9[0].ex = 0;		d2q9[0].ey = 0;		d2q9[0].wt = 4.0 / 9.0;	d2q9[0].op = 0;
	d2q9[1].ex = 1;		d2q9[1].ey = 0;		d2q9[1].wt = 1.0 / 9.0;	d2q9[1].op = 3;
	d2q9[2].ex = 0;		d2q9[2].ey = 1;		d2q9[2].wt = 1.0 / 9.0;	d2q9[2].op = 4;
	d2q9[3].ex = -1;	d2q9[3].ey = 0;		d2q9[3].wt = 1.0 / 9.0;	d2q9[3].op = 1;
	d2q9[4].ex = 0;		d2q9[4].ey = -1;	d2q9[4].wt = 1.0 / 9.0;	d2q9[4].op = 2;
	d2q9[5].ex = 1;		d2q9[5].ey = 1;		d2q9[5].wt = 1.0 / 36.0;	d2q9[5].op = 7;
	d2q9[6].ex = -1;	d2q9[6].ey = 1;		d2q9[6].wt = 1.0 / 36.0;	d2q9[6].op = 8;
	d2q9[7].ex = -1;	d2q9[7].ey = -1;	d2q9[7].wt = 1.0 / 36.0;	d2q9[7].op = 5;
	d2q9[8].ex = 1;		d2q9[8].ey = -1;	d2q9[8].wt = 1.0 / 36.0;	d2q9[8].op = 6;
}

void initFluid()
{
	int W = params.width;
	int H = params.height;
	float v = params.v;

	barrier = (unsigned char*)calloc(W * H, sizeof(unsigned char));
	array1 = (lbm_node*)calloc(W * H, sizeof(lbm_node));
	array2 = (lbm_node*)calloc(W * H, sizeof(lbm_node));

	lbm_node* before = array1;

	d2q9_node* d2q9 = (d2q9_node*)calloc(9, sizeof(d2q9_node));
	init_d2q9(d2q9);

	DEBUG_PRINT(("\tTESTWEIGHT = %.6f", d2q9[E].wt));

	int i;
	for (int x = 0; x < params.width; x++)
	{
		for (int y = 0; y < params.height; y++)
		{
			i = getIndex_cpu(x, y);
			(before[i].f)[d0] = d2q9[d0].wt * (1 - 1.5 * v * v);
			(before[i].f)[E] = d2q9[E].wt * (1 + 3 * v + 3 * v * v);
			(before[i].f)[W] = d2q9[W].wt * (1 - 3 * v + 3 * v * v);
			(before[i].f)[N] = d2q9[N].wt * (1 - 1.5 * v * v);
			(before[i].f)[S] = d2q9[S].wt * (1 - 1.5 * v * v);
			(before[i].f)[NE] = d2q9[NE].wt * (1 + 3 * v + 3 * v * v);
			(before[i].f)[SE] = d2q9[SE].wt * (1 + 3 * v + 3 * v * v);
			(before[i].f)[NW] = d2q9[NW].wt * (1 - 3 * v + 3 * v * v);
			(before[i].f)[SW] = d2q9[SW].wt * (1 - 3 * v + 3 * v * v);
			before[i].rho = 1;
			before[i].ux = params.v;
			before[i].uy = 0;
		}
	}

	initFluid_gpu(d2q9);
}

//-----------------------------------------------------------//
//              OPENGL CALLBACK FUNCTIONS                    //
//-----------------------------------------------------------//


//keyboard callback
void keyboard(unsigned char a, int b, int c)
{
	DEBUG_PRINT(("%x pressed\n", a));

	if (!(waitingForSpeed || waitingForViscosity || waitingForRate))
	{
		switch (a)
		{
		case'1':
			params.mode = Rho;
			printf("render mode set to rho\n");
			break;
		case'2':
			params.mode = Curl;
			printf("render mode set to curl\n");
			break;
		case'3':
			params.mode = Speed;
			printf("render mode set to speed\n");
			break;
		case'4':
			params.mode = Ux;
			printf("render mode set to Ux\n");
			break;
		case'5':
			params.mode = Uy;
			printf("render mode set to Uy\n");
			break;
		case'q':
			clearBarriers();
			printf("Barriers Cleared!\n");
			break;
		case'w':
			initFluid();
			printf("Field Reset!\n");
			break;
		case'a':
			clearBarriers();
			//drawLineLong();
			break;
		case's':
			clearBarriers();
			//drawLineShort();
			break;
		case'd':
			clearBarriers();
			drawLineDiagonal();
			break;
		case'f':
			clearBarriers();
			drawSquare();
			break;
		case'z':
			printf("Enter speed using 1-0:\n");
			waitingForSpeed = 1;
			break;
		case'x':
			printf("Enter viscosity using 1-0:\n");
			waitingForViscosity = 1;
			break;
		case'c':
			printf("Enter refresh rate using 1-0:\n");
			waitingForRate = 1;
			break;

		default: break;
		}
	}
	else if (waitingForViscosity)
	{
		switch (a)
		{
		case '1': params.viscosity = 0.003; break;
		case '2': params.viscosity = 0.005; break;
		case '3': params.viscosity = 0.008; break;
		case '4': params.viscosity = 0.011; break;
		case '5': params.viscosity = 0.016; break;
		case '6': params.viscosity = 0.02; break;
		case '7': params.viscosity = 0.04; break;
		case '8': params.viscosity = 0.08; break;
		case '9': params.viscosity = 0.13; break;
		case '0': params.viscosity = 0.2; break;
		default: break;
		}
		waitingForViscosity = 0;
		printf("viscosity set to %.3f\n", params.viscosity);
	}
	else if (waitingForSpeed)
	{
		switch (a)
		{
		case '1': params.v = 0.01; break;
		case '2': params.v = 0.03; break;
		case '3': params.v = 0.05; break;
		case '4': params.v = 0.07; break;
		case '5': params.v = 0.09; break;
		case '6': params.v = 0.11; break;
		case '7': params.v = 0.13; break;
		case '8': params.v = 0.14; break;
		case '9': params.v = 0.17; break;
		case '0': params.v = 0.2; break;
		default: break;
		}
		waitingForSpeed = 0;
		printf("speed set to %.2f\n", params.v);
	}
	else if (waitingForRate)
	{
		switch (a)
		{
		case '1': params.stepsPerRender = 1; break;
		case '2': params.stepsPerRender = 2; break;
		case '3': params.stepsPerRender = 3; break;
		case '4': params.stepsPerRender = 4; break;
		case '5': params.stepsPerRender = 5; break;
		case '6': params.stepsPerRender = 6; break;
		case '7': params.stepsPerRender = 7; break;
		case '8': params.stepsPerRender = 8; break;
		case '9': params.stepsPerRender = 9; break;
		case '0': params.stepsPerRender = 10; break;
		default: break;
		}
		waitingForRate = 0;
		printf("refresh rate set to %d\n", params.stepsPerRender);
	}
	needsUpdate = 1;
}

//special keyboard callback
void handleSpecialKeypress(int a, int b, int c)
{

}

void mouseClick(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		if (button == GLUT_LEFT_BUTTON)
		{
			current_button = GLUT_LEFT_BUTTON;
			int lx, ly; // lattice coordinates
			lx = x * params.width / glutGet(GLUT_WINDOW_WIDTH);
			ly = y * params.height / glutGet(GLUT_WINDOW_HEIGHT);

			if (lx >= params.width || ly >= params.height)
				return;

			barrier[getIndex_cpu(lx, ly)] = 1;
			needsUpdate = 1;
		}
		else if (button == GLUT_RIGHT_BUTTON)
		{
			current_button = GLUT_RIGHT_BUTTON;
			int lx, ly; // lattice coordinates
			lx = x * params.width / glutGet(GLUT_WINDOW_WIDTH);
			ly = y * params.height / glutGet(GLUT_WINDOW_HEIGHT);

			if (lx >= params.width || ly >= params.height)
				return;

			barrier[getIndex_cpu(lx, ly)] = 0;
			needsUpdate = 1;
		}
	}
}

//mouse move callback
void mouseMove(int x, int y)
{

	int lx, ly; // lattice coordinates
	lx = x * params.width / glutGet(GLUT_WINDOW_WIDTH);
	ly = y * params.height / glutGet(GLUT_WINDOW_HEIGHT);

	if (lx >= params.width || ly >= params.height)
		return;

	prex = lx;
	prey = ly;
}

//mouse drag callback
void mouseDrag(int x, int y)
{
	int lx, ly; // lattice coordinates
	lx = x * params.width / glutGet(GLUT_WINDOW_WIDTH);
	ly = y * params.height / glutGet(GLUT_WINDOW_HEIGHT);

	if (lx >= params.width || ly >= params.height)
		return;

	prex = lx;
	prey = ly;

	if (current_button == GLUT_LEFT_BUTTON)
	{
		barrier[getIndex_cpu(lx, ly)] = 1;
	}
	else if (current_button == GLUT_RIGHT_BUTTON)
	{
		barrier[getIndex_cpu(lx, ly)] = 0;
	}

	needsUpdate = 1;
}

//gl exit callback
void exitfunc()
{
	//empty all cuda resources
	if (pbo)
	{
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}

	cudaExit();
}

//display stats of all detected cuda capable devices,
//and return the number
int deviceQuery()
{
	cudaDeviceProp prop;
	int nDevices = 1;
	cudaError_t ierr;


	ierr = cudaGetDeviceCount(&nDevices);

	int i = 0;
	for (i = 0; i < nDevices; ++i)
	{
		ierr = cudaGetDeviceProperties(&prop, i);
		printf("Device number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("  Max threads in X-dimension of block: %d\n", prop.maxThreadsDim[0]);
		printf("  Max threads in Y-dimension of block: %d\n", prop.maxThreadsDim[1]);
		printf("  Max threads in Z-dimension of block: %d\n\n", prop.maxThreadsDim[2]);
		if (ierr != cudaSuccess) { printf("error: %s\n", cudaGetErrorString(ierr)); }
	}

	return nDevices;
}

//render the image (but do not display it yet)
void render(int delta_t) {
	//reset image pointer
	uchar4* d_out = 0;
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	//set d_out as a texture memory pointer
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_out, NULL, cuda_pbo_resource);


	//launch cuda kernels to calculate LBM step
	for (int i = 0; i < params.stepsPerRender; i++)
	{
		kernelLauncher(d_out);
	}
	//unmap the resources for next time
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

//update textures to reflect texture memory
void drawTextureScaled() {
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, params.width, params.height,
		0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0, params.height);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(params.width, params.height);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(params.width, 0);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}

//update the live display
void display(int delta_t) {

	//launch cuda kernels to update Lattice-Boltzmann,
	//flip front and back LBM buffers,
	//and update texture memory
	render(delta_t);

	//redraw textures
	drawTextureScaled();

	//swap the buffers
	glutSwapBuffers();
}

// (gl idle callback) handle frame limitting, fps calculating, and call display functions
// triggered when glutmainloop() is idle
void update()
{
	//find time since last frame update. Will replace with timers later for precision beyond 1ms
	current_draw_time = clock();
	delta_t = current_draw_time - last_draw_time;

	//limit framerate to 5Hz
	if (delta_t < DEBUG_DELAY)
	{
		return;
	}

	last_draw_time = current_draw_time;

	//calculate fps
	fps = delta_t != 0 ? 1000.0 / delta_t : 0;
	//DEBUG_PRINT("in render: delta t = %.6f\n", delta_t);
	display(delta_t);
}

//creates and binds texture memory
void initPixelBuffer() 
{
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * params.width * params.height
		* sizeof(GLubyte), 0, GL_STREAM_DRAW);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void initGLUT(int* argc, char** argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(params.width, params.height);
	glutCreateWindow("LBM GPU");

#ifndef __APPLE__
	glewInit();
#endif

	gluOrtho2D(0, params.width, params.height, 0);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(handleSpecialKeypress);
	glutPassiveMotionFunc(mouseMove);
	glutMouseFunc(mouseClick);
	glutMotionFunc(mouseDrag);
	glutDisplayFunc(update);
	glutIdleFunc(update);
	initPixelBuffer();
}

//---------------------------------------------------------------------------------------//
//                                MAIN FUNCTION                                          //
//---------------------------------------------------------------------------------------//

int main(int argc, char** argv) {

	//discover all Cuda-capable hardware
	int i = deviceQuery();
	//DEBUG_PRINT(("num devices is %d\n", i));

	if (i < 1)
	{
		//DEBUG_PRINT(("ERROR: no cuda capable hardware detected!\n"));
		getchar();
		return 0; //return if no cuda-capable hardware is present
	}

	//allocate memory and initialize fluid arrays
	getParams(&params);
	initFluid();

	//construct output window
	initGLUT(&argc, argv);

	//run gl main loop!
	glutMainLoop();

	//declare exit callback
	atexit(exitfunc);
	getchar();
	return 0;
}

