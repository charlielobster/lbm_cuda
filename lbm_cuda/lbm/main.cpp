#include <malloc.h>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cstdlib>

#include "lbm.h"
#include "lbm_delegate.h"

// texture and pixel objects
GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object

float rotate_x = 0.0;
float rotate_y = 0.0;
float translate_z = -2.0;
int previous_mouse_x;
int previous_mouse_y;
int mouse_buttons = 0;

bool barriersUpdated = true;
render_mode mode = CURL;

// memory pointers:
unsigned char* barrier;
unsigned char out[LATTICE_DIMENSION];

int current_button = GLUT_LEFT_BUTTON; 

lbm_delegate lbm; // encapsulate LBM-related activity

void clearBarriers()
{
	for (int i = 0; i < LATTICE_WIDTH; i++)
	{
		for (int j = 0; j < LATTICE_HEIGHT; j++)
		{
			barrier[INDEX(i, j)] = 0;
		}
	}
}

void drawLineDiagonal()
{
	for (int i = 0; i < LATTICE_HEIGHT / 4; i++)
	{

		barrier[INDEX((LATTICE_WIDTH / 3) + (i / 3), LATTICE_HEIGHT / 3 + i)] = 1;
	}
}

void drawSquare()
{
	for (int i = 0; i < LATTICE_HEIGHT / 4; i++)
	{
		for (int j = 0; j < LATTICE_HEIGHT / 4; j++)
		{
			barrier[INDEX(i + LATTICE_WIDTH / 3, j + LATTICE_HEIGHT * 3 / 8)] = 1;
		}
	}
}

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
		clearBarriers();
		printf("Barriers Cleared!\n");
		break;
	case'w':
		barrier = (unsigned char*)calloc(LATTICE_DIMENSION, sizeof(unsigned char));
		lbm.resetLattice(pbo, barrier);
		printf("Field Reset!\n");
		break;		
	case'd':
		clearBarriers();
		drawLineDiagonal();
		break;
	case'f':
		clearBarriers();
		drawSquare();
		break;
	default: break;
	}
	barriersUpdated = true;
}

void exitFunc()
{
	glDeleteBuffers(1, &pbo);
	glDeleteTextures(1, &tex);
	lbm.freeCUDA();
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

	lbm.launchKernels(mode, barriersUpdated, barrier, out);
	barriersUpdated = false;

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

int main(int argc, char** argv) 
{
	//discover all Cuda-capable hardware
	lbm_delegate::printDeviceInfo();
	initGLUT(&argc, argv);

	barrier = (unsigned char*)calloc(LATTICE_DIMENSION, sizeof(unsigned char));
	lbm.resetLattice(pbo, barrier);

	drawSquare();

	glutMainLoop();

	atexit(exitFunc);
	return 0;
}