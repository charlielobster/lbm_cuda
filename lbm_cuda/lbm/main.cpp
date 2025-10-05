// main.cpp
// OpenGL boilerplate

#include <malloc.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cstdlib>

#include "d2q9_delegate.h"

// texture and pixel objects
GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object

// mouse state & UI flags
float rotate_x = 0.0;
float rotate_y = 0.0;
float translate_z = -2.0;
int previous_mouse_x;
int previous_mouse_y;
int mouse_buttons = 0;
int current_button = GLUT_LEFT_BUTTON; 

// encapsulate LBM-related activities
lbm_delegate* lbm = new d2q9_delegate(); 
lbm_render_mode mode = CURL;

//keyboard callback
void keyboard(unsigned char a, int b, int c)
{
	switch (a)
	{
	case'1':
		mode = CURL;
		printf("render mode set to curl\n");
		break;
	case'2':
		mode = SPEED;
		printf("render mode set to speed\n");
		break;
	case'3':
		mode = UX;
		printf("render mode set to Ux\n");
		break;
	case'4':
		mode = UY;
		printf("render mode set to Uy\n");
		break;
	case'q':
		lbm->clearBarrier();
		printf("Barrier cleared\n");
		break;
	case'w':
		lbm->resetLattice(pbo);
		printf("Field reset\n");
		break;		
	case'd':
		lbm->drawLineDiagonal();
		break;
	case'f':
		lbm->drawSquare();
		break;
	default: break;
	}
}

void exitFunc()
{
	glDeleteBuffers(1, &pbo);
	glDeleteTextures(1, &tex);
	lbm->freeCUDA();
}

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	previous_mouse_x = x;
	previous_mouse_y = y;
	glutPostRedisplay();
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - previous_mouse_x);
	dy = (float)(y - previous_mouse_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	previous_mouse_x = x;
	previous_mouse_y = y;
	glutPostRedisplay();
}

//update the display
void display() 
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	lbm->launchKernels(mode);

	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, LATTICE_WIDTH, LATTICE_HEIGHT, 
		0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0);	glVertex3f(-2.0, -1.0, 0.0);
	glTexCoord2f(0.0, 1.0); glVertex3f(-2.0, 1.0, 0.0);
	glTexCoord2f(1.0, 1.0); glVertex3f(2.0, 1.0, 0.0);
	glTexCoord2f(1.0, 0.0); glVertex3f(2.0, -1.0, 0.0);
	glEnd();

	glDisable(GL_TEXTURE_2D);

	glutSwapBuffers();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 1.0, 30.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void initGLUT(int* argc, char** argv) 
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(LATTICE_WIDTH, LATTICE_HEIGHT);
	glutCreateWindow("lbm");
	glewInit();
	gluOrtho2D(0, LATTICE_WIDTH, LATTICE_HEIGHT, 0);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutIdleFunc(display);
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 
		4 * LATTICE_DIMENSION * sizeof(GLubyte), 0, GL_STREAM_DRAW);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

//discover all Cuda-capable hardware
void printDeviceInfo()
{
	int nDevices = 0;
	cudaError_t ce = cudaGetDeviceCount(&nDevices);
	cudaDeviceProp prop;
	for (int i = 0; i < nDevices; ++i)
	{
		ce = cudaGetDeviceProperties(&prop, i);
		printf("Device number: %d\n", i);
		printf("Device name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max threads in X-dimension of block: %d\n", prop.maxThreadsDim[0]);
		printf("Max threads in Y-dimension of block: %d\n", prop.maxThreadsDim[1]);
		printf("Max threads in Z-dimension of block: %d\n\n", prop.maxThreadsDim[2]);
		if (ce != cudaSuccess) { printf("error: %s\n", cudaGetErrorString(ce)); }
	}
}

int main(int argc, char** argv) 
{
	printDeviceInfo();

	initGLUT(&argc, argv);

	lbm->resetLattice(pbo);
	lbm->drawSquare();

	glutMainLoop();

	atexit(exitFunc);
	return 0;
}