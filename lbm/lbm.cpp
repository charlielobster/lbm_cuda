#include <malloc.h>

#include <helper_gl.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>

#include "lbm.cuh"

// texture and pixel objects
GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object

float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -2.0;
int previous_mouse_x, previous_mouse_y;
int mouse_buttons = 0;

parameterSet params;

//GPU/CPU interop memory pointers:
unsigned char state = 0;
lbm_node* array1;
lbm_node* array2;
unsigned char* barrier;

char waitingForSpeed = 0;
char waitingForViscosity = 0;
char waitingForRate = 0;
int current_button = GLUT_LEFT_BUTTON; 

extern "C" int deviceQuery();
extern "C" void initCUDA(d2q9_node * d2q9, parameterSet * params, int W, int H,
	lbm_node * array1, lbm_node * array2, unsigned char* barrier);
extern "C" void initPboResource(GLuint pbo); 
extern "C" void render(parameterSet* params, unsigned char* barrier);
extern "C" void freeCUDA();

void initParams(parameterSet* params)
{
	params->needsUpdate = 1;
	params->viscosity = 0.005;
	params->contrast = 75;
	params->v = 0.1;
	params->mode = CURL;
	params->height = 128;
	params->width = 256;
	params->stepsPerRender = 10;
	params->prex = -1;
	params->prey = -1;
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
			barrier[getIndex_cpu(i + params.width / 3, j + params.height * 3 / 8)] = 1;
		}
	}
}

//provide LBM constants for d2q9 style nodes
//assumes positive is up and right, whereas our program assumes positive down and right.
void initD2q9(d2q9_node* d2q9)
{
	d2q9[0].ex = 0;		d2q9[0].ey = 0;		d2q9[0].wt = 4.0 / 9.0;		d2q9[0].op = 0;
	d2q9[1].ex = 1;		d2q9[1].ey = 0;		d2q9[1].wt = 1.0 / 9.0;		d2q9[1].op = 3;
	d2q9[2].ex = 0;		d2q9[2].ey = 1;		d2q9[2].wt = 1.0 / 9.0;		d2q9[2].op = 4;
	d2q9[3].ex = -1;	d2q9[3].ey = 0;		d2q9[3].wt = 1.0 / 9.0;		d2q9[3].op = 1;
	d2q9[4].ex = 0;		d2q9[4].ey = -1;	d2q9[4].wt = 1.0 / 9.0;		d2q9[4].op = 2;
	d2q9[5].ex = 1;		d2q9[5].ey = 1;		d2q9[5].wt = 1.0 / 36.0;	d2q9[5].op = 7;
	d2q9[6].ex = -1;	d2q9[6].ey = 1;		d2q9[6].wt = 1.0 / 36.0;	d2q9[6].op = 8;
	d2q9[7].ex = -1;	d2q9[7].ey = -1;	d2q9[7].wt = 1.0 / 36.0;	d2q9[7].op = 5;
	d2q9[8].ex = 1;		d2q9[8].ey = -1;	d2q9[8].wt = 1.0 / 36.0;	d2q9[8].op = 6;
}

void initArray1(d2q9_node * d2q9, float v, int W, int H)
{
	array1 = (lbm_node*)calloc(W * H, sizeof(lbm_node));	
	int i;
	for (int x = 0; x < params.width; x++)
	{
		for (int y = 0; y < params.height; y++)
		{
			i = getIndex_cpu(x, y);
			(array1[i].f)[NONE] = d2q9[NONE].wt * (1 - 1.5 * v * v);
			(array1[i].f)[EAST] = d2q9[EAST].wt * (1 + 3 * v + 3 * v * v);
			(array1[i].f)[WEST] = d2q9[WEST].wt * (1 - 3 * v + 3 * v * v);
			(array1[i].f)[NORTH] = d2q9[NORTH].wt * (1 - 1.5 * v * v);
			(array1[i].f)[SOUTH] = d2q9[SOUTH].wt * (1 - 1.5 * v * v);
			(array1[i].f)[NORTHEAST] = d2q9[NORTHEAST].wt * (1 + 3 * v + 3 * v * v);
			(array1[i].f)[SOUTHEAST] = d2q9[SOUTHEAST].wt * (1 + 3 * v + 3 * v * v);
			(array1[i].f)[NORTHWEST] = d2q9[NORTHWEST].wt * (1 - 3 * v + 3 * v * v);
			(array1[i].f)[SOUTHWEST] = d2q9[SOUTHWEST].wt * (1 - 3 * v + 3 * v * v);
			array1[i].rho = 1;
			array1[i].ux = params.v;
			array1[i].uy = 0;
		}
	}
}

void initFluid() 
{
	int W = params.width;
	int H = params.height;
	float v = params.v;

	initPboResource(pbo);

	barrier = (unsigned char*)calloc(W * H, sizeof(unsigned char));
	array2 = (lbm_node*)calloc(W * H, sizeof(lbm_node));

	d2q9_node* d2q9 = (d2q9_node*)calloc(9, sizeof(d2q9_node));
	initD2q9(d2q9);
	initArray1(d2q9, v, W, H);	
	initCUDA(d2q9, &params, W, H, array1, array2, barrier);
}

//keyboard callback
void keyboard(unsigned char a, int b, int c)
{
	if (!(waitingForSpeed || waitingForViscosity || waitingForRate))
	{
		switch (a)
		{
		case'1':
			params.mode = RHO;
			printf("render mode set to rho\n");
			break;
		case'2':
			params.mode = CURL;
			printf("render mode set to curl\n");
			break;
		case'3':
			params.mode = SPEED;
			printf("render mode set to speed\n");
			break;
		case'4':
			params.mode = UX;
			printf("render mode set to Ux\n");
			break;
		case'5':
			params.mode = UY;
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
	params.needsUpdate = 1;
}

//gl exit callback
void exitfunc()
{
	glDeleteBuffers(1, &pbo);
	glDeleteTextures(1, &tex);
	freeCUDA();
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

//update the live display
void display() 
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	render(&params, barrier);

	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 
		params.width, params.height, 0, 
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -0.5, 0.0);
	glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 0.5, 0.0);
	glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 0.5, 0.0);
	glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -0.5, 0.0);
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
	glutInitWindowSize(params.width, params.height);
	glutCreateWindow("lbm");
	glewInit();
	gluOrtho2D(0, params.width, params.height, 0);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutIdleFunc(display);
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 
		4 * params.width * params.height * sizeof(GLubyte), 0, GL_STREAM_DRAW);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

int main(int argc, char** argv) 
{
	//discover all Cuda-capable hardware
	int i = deviceQuery();

	if (i < 1)
	{
		getchar();
		return 0; //return if no cuda-capable hardware is present
	}

	initParams(&params);
	initGLUT(&argc, argv);
	initFluid();

	glutMainLoop();

	//declare exit callback
	atexit(exitfunc);
	getchar();
	return 0;
}