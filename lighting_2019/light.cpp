//
// Lighthouse3D.com OpenGL Lighting Sample
//
//
// This demo was built for learning purposes only.
// Some code could be severely optimised, but I tried to
// keep as simple and clear as possible.
//
// The code comes with no warranties, use it at your own risk.
// You may use it, or parts of it, wherever you want.
//
// If you do use it I would love to hear about it. Just post a comment
// at Lighthouse3D.com

// This demo requires GLEW and FreeGLUT

// Have Fun :-)

/*

In the setupShaders function we can set the light type and shading model!

*/


#include <math.h>
#include <string>

// include GLEW to access OpenGL 3.3 functions
#include <GL/glew.h>

// GLUT is the toolkit to interface with the OS
#include <GL/freeglut.h>

// Use Very Simple Libs
#include "vsMathLib.h"
#include "vsShaderLib.h"
#include "vsResSurfRevLib.h"

VSMathLib *vsml;
VSShaderLib shader, shaderF;
VSResSurfRevLib mySurfRev;

// Camera Position
float camX, camY, camZ;

// Mouse Tracking Variables
int startX, startY, tracking = 0;

// Camera Spherical Coordinates
float alpha = 39.0f, beta = 51.0f;
float r = 10.0f;

// Frame counting and FPS computation
long myTime,timebase = 0,frame = 0;
char s[32];

// light direction
float lightDir[4] = {1.0f, 1.0f, 1.0f, 0.0f};
float lightPos[4] = {4.0f, 6.0f, 2.0f, 1.0f};
float spotDir[4]  = {-4.0f, -6.0f, -2.0f, 0.0f};


// ------------------------------------------------------------
//
// Reshape Callback Function
//

void changeSize(int w, int h) {

	float ratio;
	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if(h == 0)
		h = 1;

	// Set the viewport to be the entire window
    glViewport(0, 0, w, h);

	ratio = (1.0f * w) / h;
	vsml->loadIdentity(VSMathLib::PROJECTION);
	vsml->perspective(53.13f, ratio, 0.1f, 10000.0f);
}


// ------------------------------------------------------------
//
// Render stuff
//



void renderScene(void) {

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	vsml->loadIdentity(VSMathLib::VIEW);
	vsml->loadIdentity(VSMathLib::MODEL);
	// set camera
	vsml->lookAt(camX, camY, camZ, 0,0,0, 0,1,0);

	// transform light to camera space and send it to GLSL
	float res[4];
	vsml->multMatrixPoint(VSMathLib::VIEW, lightDir, res);
	vsml->normalize(res);
	shader.setBlockUniform("Lights", "l_dir", res);

	vsml->multMatrixPoint(VSMathLib::VIEW, lightPos, res);
	shader.setBlockUniform("Lights", "l_pos", res);

	vsml->multMatrixPoint(VSMathLib::VIEW, spotDir, res);
	shader.setBlockUniform("Lights", "l_spotDir", res);

	// use our shader
	glUseProgram(shader.getProgramIndex());

	int modelLines = 2;
	for (int i = -modelLines+1 ; i < modelLines; ++i) {
		for (int j = -modelLines+1; j < modelLines; ++j) {
			vsml->pushMatrix(VSMathLib::MODEL);
			vsml->translate(i*2.0f, 0.0f, j*2.0f);
			mySurfRev.render();
			vsml->popMatrix(VSMathLib::MODEL);
		}
	}

	//swap buffers
	glutSwapBuffers();
}


// ------------------------------------------------------------
//
// Events from the Keyboard
//

void processKeys(unsigned char key, int xx, int yy)
{
	switch(key) {

		case 27:

			glutLeaveMainLoop();
			break;
		case 'c': printf("Camera Spherical Coordinates (%f, %f, %f)\n", alpha, beta, r);
			break;
		case 'm': glEnable(GL_MULTISAMPLE); break;
		case 'n': glDisable(GL_MULTISAMPLE); break;
	}

//  uncomment this if not using an idle func
//	glutPostRedisplay();
}


// ------------------------------------------------------------
//
// Mouse Events
//

void processMouseButtons(int button, int state, int xx, int yy)
{
	// start tracking the mouse
	if (state == GLUT_DOWN)  {
		startX = xx;
		startY = yy;
		if (button == GLUT_LEFT_BUTTON)
			tracking = 1;
		else if (button == GLUT_RIGHT_BUTTON)
			tracking = 2;
	}

	//stop tracking the mouse
	else if (state == GLUT_UP) {
		if (tracking == 1) {
			alpha -= (xx - startX);
			beta += (yy - startY);
		}
		else if (tracking == 2) {
			r += (yy - startY) * 0.01f;
			if (r < 0.1f)
				r = 0.1f;
		}
		tracking = 0;
	}
}

// Track mouse motion while buttons are pressed

void processMouseMotion(int xx, int yy)
{

	int deltaX, deltaY;
	float alphaAux, betaAux;
	float rAux;

	deltaX =  - xx + startX;
	deltaY =    yy - startY;

	// left mouse button: move camera
	if (tracking == 1) {


		alphaAux = alpha + deltaX;
		betaAux = beta + deltaY;

		if (betaAux > 85.0f)
			betaAux = 85.0f;
		else if (betaAux < -85.0f)
			betaAux = -85.0f;
		rAux = r;
	}
	// right mouse button: zoom
	else if (tracking == 2) {

		alphaAux = alpha;
		betaAux = beta;
		rAux = r + (deltaY * 0.01f);
		if (rAux < 0.1f)
			rAux = 0.1f;
	}

	camX = rAux * sin(alphaAux * 3.14f / 180.0f) * cos(betaAux * 3.14f / 180.0f);
	camZ = rAux * cos(alphaAux * 3.14f / 180.0f) * cos(betaAux * 3.14f / 180.0f);
	camY = rAux *   						       sin(betaAux * 3.14f / 180.0f);

//  uncomment this if not using an idle func
//	glutPostRedisplay();
}





void mouseWheel(int wheel, int direction, int x, int y) {

	r += direction * 0.1f;
	if (r < 0.1f)
		r = 0.1f;

	camX = r * sin(alpha * 3.14f / 180.0f) * cos(beta * 3.14f / 180.0f);
	camZ = r * cos(alpha * 3.14f / 180.0f) * cos(beta * 3.14f / 180.0f);
	camY = r *   						     sin(beta * 3.14f / 180.0f);

//  uncomment this if not using an idle func
//	glutPostRedisplay();
}




// --------------------------------------------------------
//
// Shader Stuff
//


GLuint setupShaders() {

	// Shader for fonts
	shaderF.init();
	shaderF.loadShader(VSShaderLib::VERTEX_SHADER, "shaders/color.vert");
	shaderF.loadShader(VSShaderLib::FRAGMENT_SHADER, "shaders/color.frag");

	// set semantics for the shader variables
	shaderF.setProgramOutput(0,"outputF");
	shaderF.setVertexAttribName(VSShaderLib::VERTEX_COORD_ATTRIB, "position");
	shaderF.setVertexAttribName(VSShaderLib::TEXTURE_COORD_ATTRIB, "texCoord");

	shaderF.prepareProgram();

	// add sampler uniforms
	shaderF.setUniform("texUnit", 0);

	printf("InfoLog for Font Shader\n%s\n\n", shaderF.getAllInfoLogs().c_str());

	// Shader for models
	shader.init();


#define _DIF			0
#define _DIF_AMB		1	
#define _DIF_AMB_SPEC	2
#define _PER_PIXEL		3
#define _POINT			4
#define _SPOT			5

#define _LIGHT 5

#if (_LIGHT == _DIF)
	shader.loadShader(VSShaderLib::VERTEX_SHADER, "shaders/dirdif.vert");
	shader.loadShader(VSShaderLib::FRAGMENT_SHADER, "shaders/dirdif.frag");
#elif (_LIGHT == _DIF_AMB)
	shader.loadShader(VSShaderLib::VERTEX_SHADER, "shaders/dirdifamb.vert");
	shader.loadShader(VSShaderLib::FRAGMENT_SHADER, "shaders/dirdifamb.frag");
#elif (_LIGHT == _DIF_AMB_SPEC)
	shader.loadShader(VSShaderLib::VERTEX_SHADER, "shaders/dirdifambspec.vert");
	shader.loadShader(VSShaderLib::FRAGMENT_SHADER, "shaders/dirdifambspec.frag");
#elif (_LIGHT == _PER_PIXEL)
	shader.loadShader(VSShaderLib::VERTEX_SHADER, "shaders/pixeldirdifambspec.vert");
	shader.loadShader(VSShaderLib::FRAGMENT_SHADER, "shaders/pixeldirdifambspec.frag");
#elif (_LIGHT == _POINT)
	shader.loadShader(VSShaderLib::VERTEX_SHADER, "shaders/pointlight.vert");
	shader.loadShader(VSShaderLib::FRAGMENT_SHADER, "shaders/pointlight.frag");
#else
	shader.loadShader(VSShaderLib::VERTEX_SHADER, "shaders/spotlight.vert");
	shader.loadShader(VSShaderLib::FRAGMENT_SHADER, "shaders/spotlight.frag");
#endif

	// set semantics for the shader variables
	shader.setProgramOutput(0,"outputF");
	shader.setVertexAttribName(VSShaderLib::VERTEX_COORD_ATTRIB, "position");
	shader.setVertexAttribName(VSShaderLib::NORMAL_ATTRIB, "normal");
	shader.setVertexAttribName(VSShaderLib::TEXTURE_COORD_ATTRIB, "texCoord");
	shader.prepareProgram();

	shader.setUniform("texUnit", 0);
	float f3 = 0.90f;
	shader.setBlockUniform("Lights", "l_spotCutOff", &f3);


	printf("InfoLog for Model Shader\n%s\n\n", shader.getAllInfoLogs().c_str());
	
	return(shader.isProgramValid());
}



// ------------------------------------------------------------
//
// Model loading and OpenGL setup
//


void init()
{

	camX = r * sin(alpha * 3.14f / 180.0f) * cos(beta * 3.14f / 180.0f);
	camZ = r * cos(alpha * 3.14f / 180.0f) * cos(beta * 3.14f / 180.0f);
	camY = r *   						     sin(beta * 3.14f / 180.0f);

	mySurfRev.createPawn();
	mySurfRev.setMaterialBlockName("Materials");

	float f[4] = {0.8f, 0.6f, 0.4f, 1.0f};
	mySurfRev.setColor(VSResourceLib::DIFFUSE, f);
	float f2[4] = {0.2f, 0.15f, 0.1f, 1.0f};
	mySurfRev.setColor(VSResourceLib::AMBIENT, f2);

	// some GL settings
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_MULTISAMPLE);

	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
}


void initVSL() {

//	 Init VSML
	vsml = VSMathLib::getInstance();
	vsml->setUniformBlockName("Matrices");
	vsml->setUniformName(VSMathLib::PROJ_VIEW_MODEL, "m_pvm");
	vsml->setUniformName(VSMathLib::NORMAL, "m_normal");
	vsml->setUniformName(VSMathLib::VIEW_MODEL, "m_viewModel");
}

// ------------------------------------------------------------
//
// Main function
//


int main(int argc, char **argv) {

//  GLUT initialization
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH|GLUT_DOUBLE|GLUT_RGBA|GLUT_MULTISAMPLE);

	glutInitContextVersion (3, 3);
	glutInitContextProfile (GLUT_CORE_PROFILE );

	glutInitWindowPosition(100,100);
	glutInitWindowSize(512,512);
	glutCreateWindow("Lighthouse3D - Lighting Demo");


//  Callback Registration
	glutDisplayFunc(renderScene);
	glutReshapeFunc(changeSize);
	glutIdleFunc(renderScene);

//	Mouse and Keyboard Callbacks
	glutKeyboardFunc(processKeys);
	glutMouseFunc(processMouseButtons);
	glutMotionFunc(processMouseMotion);

	glutMouseWheelFunc ( mouseWheel ) ;

//	return from main loop
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

//	Init GLEW
	glewExperimental = GL_TRUE;
	glewInit();

   printf ("Vendor: %s\n", glGetString (GL_VENDOR));
   printf ("Renderer: %s\n", glGetString (GL_RENDERER));
   printf ("Version: %s\n", glGetString (GL_VERSION));
   printf ("GLSL: %s\n", glGetString (GL_SHADING_LANGUAGE_VERSION));
   int param;
   glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &param);
   if (param == GL_CONTEXT_CORE_PROFILE_BIT)
	   printf("Context Profile: Core\n");
   else
	   printf("Context Profile: Compatibility\n");

	if (!setupShaders())
		exit(1);

	init();

	initVSL();

	//  GLUT main loop
	glutMainLoop();

	return(0);

}

