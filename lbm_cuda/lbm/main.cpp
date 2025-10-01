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
/*
// memory pointers:
d2q9_lbm_node* array1;
d2q9_lbm_node* array2;*/
unsigned char* barrier;
unsigned char out[LATTICE_DIMENSION];

int current_button = GLUT_LEFT_BUTTON; 
lbm_delegate lbm;

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
/*
//provide LBM constants for d2q9 style nodes
//assumes positive is up and right, whereas our program assumes positive down and right.
void initD2q9(d2q9_position_weight* d2q9)
{
	d2q9[0].x_position = 0;		d2q9[0].y_position = 0;		d2q9[0].weight = 4.0 / 9.0;		d2q9[0].opposite = 0;
	d2q9[1].x_position = 1;		d2q9[1].y_position = 0;		d2q9[1].weight = 1.0 / 9.0;		d2q9[1].opposite = 3;
	d2q9[2].x_position = 0;		d2q9[2].y_position = 1;		d2q9[2].weight = 1.0 / 9.0;		d2q9[2].opposite = 4;
	d2q9[3].x_position = -1;	d2q9[3].y_position = 0;		d2q9[3].weight = 1.0 / 9.0;		d2q9[3].opposite = 1;
	d2q9[4].x_position = 0;		d2q9[4].y_position = -1;	d2q9[4].weight = 1.0 / 9.0;		d2q9[4].opposite = 2;
	d2q9[5].x_position = 1;		d2q9[5].y_position = 1;		d2q9[5].weight = 1.0 / 36.0;	d2q9[5].opposite = 7;
	d2q9[6].x_position = -1;	d2q9[6].y_position = 1;		d2q9[6].weight = 1.0 / 36.0;	d2q9[6].opposite = 8;
	d2q9[7].x_position = -1;	d2q9[7].y_position = -1;	d2q9[7].weight = 1.0 / 36.0;	d2q9[7].opposite = 5;
	d2q9[8].x_position = 1;		d2q9[8].y_position = -1;	d2q9[8].weight = 1.0 / 36.0;	d2q9[8].opposite = 6;
}

void initArray1(d2q9_position_weight* d2q9)
{
	//out = (unsigned char*)calloc(LATTICE_DIMENSION, sizeof(unsigned char));
	array1 = (d2q9_lbm_node*)calloc(LATTICE_DIMENSION, sizeof(d2q9_lbm_node));	
	int i;
	for (int x = 0; x < LATTICE_WIDTH; x++)
	{
		for (int y = 0; y < LATTICE_HEIGHT; y++)
		{
			i = INDEX(x, y);
			array1[i].vectors[ZERO] = d2q9[ZERO].weight * (1 - 1.5 * VELOCITY_SQUARED);
			array1[i].vectors[EAST] = d2q9[EAST].weight * (1 + _3V+ _3V2);
			array1[i].vectors[WEST] = d2q9[WEST].weight * (1 - _3V+ _3V2);
			array1[i].vectors[NORTH] = d2q9[NORTH].weight * (1 - 1.5 * VELOCITY_SQUARED);
			array1[i].vectors[SOUTH] = d2q9[SOUTH].weight * (1 - 1.5 * VELOCITY_SQUARED);
			array1[i].vectors[NORTHEAST] = d2q9[NORTHEAST].weight * (1 + _3V+ _3V2);
			array1[i].vectors[SOUTHEAST] = d2q9[SOUTHEAST].weight * (1 + _3V+ _3V2);
			array1[i].vectors[NORTHWEST] = d2q9[NORTHWEST].weight * (1 - _3V+ _3V2);
			array1[i].vectors[SOUTHWEST] = d2q9[SOUTHWEST].weight * (1 - _3V+ _3V2);
			array1[i].rho = 1;
			array1[i].ux = VELOCITY;
			array1[i].uy = 0;
		}
	}
}

void resetLattice() 
{
	barrier = (unsigned char*)calloc(LATTICE_DIMENSION, sizeof(unsigned char));
	lbm.resetLattice(pbo, barrier);
}
*/
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