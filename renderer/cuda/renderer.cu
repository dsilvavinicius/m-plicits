#include "neural_implicits.cu"

 // USE_TEXSUBIMAGE2D uses glTexSubImage2D() to update the final result
 // commenting it will make the sample use the other way :
 // map a texture in CUDA and blit the result into it
#define USE_TEXSUBIMAGE2D

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#pragma warning(disable:4996)
#endif

#include <chrono>

// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_functions.h>
#include <rendercheck_gl.h>

// Dear ImGUI
#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl3.h"

// Shared Library Test Functions
#define MAX_EPSILON 10
#define REFRESH_DELAY     10 //ms

const char* sSDKname = "simpleCUDA2GL";

unsigned int g_TotalErrors = 0;

// CheckFBO/BackBuffer class objects
CheckRender* g_CheckRender = NULL;

////////////////////////////////////////////////////////////////////////////////
// constants / global variables
unsigned int image_width = RESOLUTION;
unsigned int image_height = RESOLUTION;
unsigned int window_width = RESOLUTION;
unsigned int window_height = RESOLUTION;

int iGLUTWindowHandle = 0;          // handle to the GLUT window

// pbo and fbo variables
#ifdef USE_TEXSUBIMAGE2D
GLuint pbo_dest;
struct cudaGraphicsResource* cuda_pbo_dest_resource;
#else
unsigned int* cuda_dest_resource;
GLuint shDrawTex;  // draws a texture
struct cudaGraphicsResource* cuda_tex_result_resource;
#endif

GLuint fbo_source;
struct cudaGraphicsResource* cuda_tex_screen_resource;

unsigned int size_tex_data;
unsigned int num_texels;
unsigned int num_values;

// (offscreen) render target fbo variables
GLuint tex_screen;      // where we render the image
GLuint tex_cudaResult;  // where we will copy the CUDA result

char* ref_file = NULL;

// -benchmark=<N>: render N frames (after a short warmup), print the average
// FPS, append it to benchmark.csv, and exit. Produces the Tab. 4 numbers.
int g_benchmark_frames = 0;
static int g_bench_counted = 0;
static int g_bench_warmup = 30;
static std::chrono::high_resolution_clock::time_point g_bench_t0;
bool enable_cuda = true;

int* pArgc = NULL;
char** pArgv = NULL;


// Timer
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface* timer = NULL;
std::chrono::high_resolution_clock::time_point cam_t1;
std::chrono::high_resolution_clock::time_point time_t1;

bool show_gui = true;
bool save_snapshot = false;

// 4D Neural Implicits constants.
float neural_implicit_time = 0.f; // 4th dimension value
float animation_param = 0.f;
float distance_threshold = 0.05f; // Distance treshold for SDF rendering
int lod_0_sphere_tracing_iters = 20; // Sphere tracing iterations
int lod_1_sphere_tracing_iters = 0; // Sphere tracing iterations
int lod_2_sphere_tracing_iters = 0; // Sphere tracing iterations
float lod_0_delta = 0.f;
float lod_1_delta = 0.f;
int lod_to_show = 0;
bool skip_lod_0 = false;
Shading shading = PHONG;
bool animate_cam = false;
bool animate_time = false;
bool is_residual = true;
float cam_time = 0.f;
float cam_velocity = 5.0f;
Arcball arcball(image_width, image_height);

#ifndef USE_TEXTURE_RGBA8UI
#   pragma message("Note: Using Texture fmt GL_RGBA16F_ARB")
#else
// NOTE: the current issue with regular RGBA8 internal format of textures
// is that HW stores them as BGRA8. Therefore CUDA will see BGRA where users
// expected RGBA8. To prevent this issue, the driver team decided to prevent this to happen
// instead, use RGBA8UI which required the additional work of scaling the fragment shader
// output from 0-1 to 0-255. This is why we have some GLSL code, in this case
#   pragma message("Note: Using Texture RGBA8UI + GLSL for rendering")
#endif
GLuint shDraw;

////////////////////////////////////////////////////////////////////////////////

// Forward declarations
void runStdProgram(int argc, char** argv);
void FreeResource();
void Cleanup(int iExitCode);

// GL functionality
bool initGL(int* argc, char** argv);

#ifdef USE_TEXSUBIMAGE2D
void createPBO(GLuint* pbo, struct cudaGraphicsResource** pbo_resource);
void deletePBO(GLuint* pbo);
#endif

void createTextureDst(GLuint* tex_cudaResult, unsigned int size_x, unsigned int size_y);
void deleteTexture(GLuint* tex);

// rendering callbacks
void display();
void idle();
void keyboard(unsigned char key, int x, int y);
void reshape(int w, int h);
void mainMenu(int i);

#ifdef USE_TEXSUBIMAGE2D
////////////////////////////////////////////////////////////////////////////////
//! Create PBO
////////////////////////////////////////////////////////////////////////////////
void
createPBO(GLuint* pbo, struct cudaGraphicsResource** pbo_resource)
{
    // set up vertex data parameter
    num_texels = image_width * image_height;
    num_values = num_texels * 4;
    size_tex_data = sizeof(GLubyte) * num_values;
    void* data = malloc(size_tex_data);

    // create buffer object
    glGenBuffers(1, pbo);
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
    free(data);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(pbo_resource, *pbo, cudaGraphicsMapFlagsNone));

    SDK_CHECK_ERROR_GL();
}

void
deletePBO(GLuint* pbo)
{
    glDeleteBuffers(1, pbo);
    SDK_CHECK_ERROR_GL();
    *pbo = 0;
}
#endif

const GLenum fbo_targets[] =
{
    GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT,
    GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT
};

#ifndef USE_TEXSUBIMAGE2D
static const char* glsl_drawtex_vertshader_src =
"void main(void)\n"
"{\n"
"	gl_Position = gl_Vertex;\n"
"	gl_TexCoord[0].xy = gl_MultiTexCoord0.xy;\n"
"}\n";

static const char* glsl_drawtex_fragshader_src =
"#version 130\n"
"uniform usampler2D texImage;\n"
"void main()\n"
"{\n"
"   vec4 c = texture(texImage, gl_TexCoord[0].xy);\n"
"	gl_FragColor = c / 255.0;\n"
"}\n";
#endif

static const char* glsl_draw_fragshader_src =
//WARNING: seems like the gl_FragColor doesn't want to output >1 colors...
//you need version 1.3 so you can define a uvec4 output...
//but MacOSX complains about not supporting 1.3 !!
// for now, the mode where we use RGBA8UI may not work properly for Apple : only RGBA16F works (default)
#if defined(__APPLE__) || defined(MACOSX)
"void main()\n"
"{"
"  gl_FragColor = vec4(gl_Color * 255.0);\n"
"}\n";
#else
"#version 130\n"
"out uvec4 FragColor;\n"
"void main()\n"
"{"
"  FragColor = uvec4(gl_Color.xyz * 255.0, 255.0);\n"
"}\n";
#endif

// copy image and process using CUDA
void generateCUDAImage()
{
    // run the Cuda kernel
    unsigned int* out_data;

#ifdef USE_TEXSUBIMAGE2D
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_dest_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&out_data, &num_bytes,
        cuda_pbo_dest_resource));
    //printf("CUDA mapped pointer of pbo_out: May access %ld bytes, expected %d\n", num_bytes, size_tex_data);
#else
    out_data = cuda_dest_resource;
#endif

    if (animate_cam) {

        std::chrono::high_resolution_clock::time_point cam_t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::deca> time_span(cam_t2 - cam_t1);
        cam_time += time_span.count();
        cam_t1 = cam_t2;
    }

    if (animate_time) {
    
        std::chrono::high_resolution_clock::time_point time_t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::deca> time_span(time_t2 - time_t1);
        neural_implicit_time += time_span.count();
        time_t1 = time_t2;

        float t = (cos(neural_implicit_time+ 3.14159265359)+1.f);
        //animation_param = precision_t(t*t*t*0.5f+0.5f);
        animation_param = precision_t(t*0.25f);
    }

    // rotation of 180 degrees
    //cam_time = precision_t(3.14159265359f*0.5f);

    // execute CUDA kernel
    //launch_cudaProcess(grid, block, 0, out_data, image_width);
    //SdfInference(lod_0_sphere_tracing_iters, lod_1_sphere_tracing_iters, neural_implicit_time, cam_time_0, distance_threshold, out_data, lod_to_show, skip_lod_0, shading);

    glm::mat4 inv_view_matrix = arcball.getInvViewMatrix();
    glm::mat4 inv_proj_matrix = arcball.getInvProjectionMatrix();
    float h_inv_view_matrix[16]; // Assuming column-major order for GLM
    float h_inv_proj_matrix[16]; // Assuming column-major order for GLM

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            h_inv_view_matrix[i * 4 + j] = inv_view_matrix[i][j];
            h_inv_proj_matrix[i * 4 + j] = inv_proj_matrix[i][j];
        }
    }

    SdfInference(lod_0_sphere_tracing_iters, lod_1_sphere_tracing_iters, lod_2_sphere_tracing_iters, lod_0_delta, lod_1_delta, animation_param, cam_time * cam_velocity, distance_threshold, out_data, lod_to_show, skip_lod_0, is_residual, shading, h_inv_view_matrix, h_inv_proj_matrix);
    //testRays(out_data, make_uint2(image_width, image_height));

    // CUDA generated data in cuda memory or in a mapped PBO made of BGRA 8 bits
    // 2 solutions, here :
    // - use glTexSubImage2D(), there is the potential to loose performance in possible hidden conversion
    // - map the texture and blit the result thanks to CUDA API
#ifdef USE_TEXSUBIMAGE2D
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_dest_resource, 0));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_dest);

    glBindTexture(GL_TEXTURE_2D, tex_cudaResult);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
        image_width, image_height,
        GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
    // We want to copy cuda_dest_resource data to the texture
    // map buffer objects to get CUDA device pointers
    cudaArray* texture_ptr;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0));

    int num_texels = image_width * image_height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0));
#endif

    if (save_snapshot) {
    
        vector<unsigned char> out_data_cpu(num_bytes);
        
        cudaMemcpy(out_data_cpu.data(), out_data, num_bytes, cudaMemcpyDeviceToHost);
        SaveInferenceImage(out_data_cpu.data(), image_width, image_height);
        save_snapshot = false;
    }
}

// display image to the screen as textured quad
void displayImage(GLuint texture)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, window_width, window_height);

    // if the texture is a 8 bits UI, scale the fetch with a GLSL shader
#ifndef USE_TEXSUBIMAGE2D
    glUseProgram(shDrawTex);
    GLint id = glGetUniformLocation(shDrawTex, "texImage");
    glUniform1i(id, 0); // texture unit 0 to "texImage"
    SDK_CHECK_ERROR_GL();
#endif

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);

#ifndef USE_TEXSUBIMAGE2D
    glUseProgram(0);
#endif
    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void
display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGLUT_NewFrame();
    
    // ImGui
    if(show_gui) {

        //ImGui::SetNextWindowPos(ImVec2(0.f, 10.f));
        ImGui::Begin("Options");

        if (ImGui::CollapsingHeader("Experiments", ImGuiTreeNodeFlags_DefaultOpen)) {

            if (ImGui::Button("4D Spot Bob")) {
            
                InitSdf(spot_bob_residual_1x64_tex_2x256);
            }
            ImGui::SameLine();
            if (ImGui::Button("Lucy")) {

                InitSdf(lucy);
            }
            ImGui::SameLine();
            if (ImGui::Button("Armadillo")) {

                InitSdf(armadillo);
            }
            ImGui::SameLine();
            if (ImGui::Button("Buddha")) {
            
                InitSdf(buddha_neigh_residual_64x1_128x1_256x1);
            }
            ImGui::SameLine();
            if (ImGui::Button("Vase")) {
            
                InitSdf(vase_1x256_tex_2x400);
            }
        }

        if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            // Buddha to Armadillo
            //ImGui::SliderFloat("time", &neural_implicit_time, -.5f, .5f);
            // Falcon
            //ImGui::SliderFloat("animation param", &animation_param, -.2f, .2f);
            // Max Plank
            ImGui::SliderFloat("animation param", &animation_param, -0.1f, 0.1f);
            // Others
            //ImGui::SliderFloat("animation param", &animation_param, .0f, .5f);
            //ImGui::SliderFloat("time", &neural_implicit_time, -1.0f, 3.0f);
            ImGui::SliderFloat("threshold", &distance_threshold, 0.f, .2f, "%.6f");
            ImGui::SliderInt("LOD 0 iters", &lod_0_sphere_tracing_iters, 1, 100);
            ImGui::SliderInt("LOD 1 iters", &lod_1_sphere_tracing_iters, 0, 100);
            ImGui::SliderInt("LOD 2 iters", &lod_2_sphere_tracing_iters, 0, 100);
            ImGui::SliderFloat("LOD 0 delta", &lod_0_delta, 0.f, .2f, "%.6f");
            ImGui::SliderFloat("LOD 1 delta", &lod_1_delta, 0.f, .2f, "%.6f");
            ImGui::Checkbox("Skip LOD 0", &skip_lod_0);

            ImGui::BeginGroup();
            ImGui::Text("Normal Mapping:"); ImGui::SameLine();
            ImGui::RadioButton("LOD 0", (int*)&lod_to_show, 0); ImGui::SameLine();
            ImGui::RadioButton("LOD 1", (int*)&lod_to_show, 1); ImGui::SameLine();
            ImGui::RadioButton("LOD 2", (int*)&lod_to_show, 2);
            ImGui::EndGroup();

            ImGui::BeginGroup();
            ImGui::Text("Shading:"); ImGui::SameLine();
            ImGui::RadioButton("normals", (int*)&shading, NORMALS); ImGui::SameLine();
            ImGui::RadioButton("shaded", (int*)&shading, PHONG);
            ImGui::EndGroup();
        }
        
        if (ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {

            float fov = arcball.getFov();
            if (ImGui::SliderFloat("FOV", &fov, 30.f, 120.f)) {
                    arcball.setPerspective(fov, 0.00001f, 10000.f);
            }

            if (ImGui::Checkbox("animate cam", &animate_cam)) {

                if (animate_cam) {

                    cam_t1 = std::chrono::high_resolution_clock::now();
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("reset cam")) {

                cam_time = 0.f;
            }
            if (ImGui::Checkbox("Is residual", &is_residual)) {
            }
            ImGui::SliderFloat("cam speed", &cam_velocity, 0.f, 10.f);

            if (ImGui::Checkbox("animate time", &animate_time)) {

                if (animate_time) {

                    cam_t1 = std::chrono::high_resolution_clock::now();
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("reset time")) {

                neural_implicit_time = 0.f;
            }

            if (ImGui::Button("reset both")) {

                cam_time = 0.f;
                neural_implicit_time = 0.f;
            }

            if (ImGui::Button("Save snapshot")) {

                save_snapshot = true;
            }
        }

        ImGui::End();
    }

    sdkStartTimer(&timer);

    if (enable_cuda)
    {
        generateCUDAImage();
        displayImage(tex_cudaResult);
    }

    // NOTE: I needed to add this call so the timing is consistent.
    // Need to investigate why
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    ImGui::Render();
    ImGuiIO& io = ImGui::GetIO();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // flip backbuffer
    glutSwapBuffers();

    // If specified, Check rendering against reference,
    if (ref_file && g_CheckRender && g_CheckRender->IsQAReadback())
    {

        static int pass = 0;

        if (pass > 0)
        {
            g_CheckRender->readback(window_width, window_height);
            char currentOutputPPM[256];
            sprintf(currentOutputPPM, "kilt.ppm");
            g_CheckRender->savePPM(currentOutputPPM, true, NULL);

            if (!g_CheckRender->PPMvsPPM(currentOutputPPM, sdkFindFilePath(ref_file, pArgv[0]), MAX_EPSILON, 0.30f))
            {
                g_TotalErrors++;
            }

            Cleanup((g_TotalErrors == 0) ? EXIT_SUCCESS : EXIT_FAILURE);
        }

        pass++;
    }

    // Update fps counter, fps/title display and log
    if (++fpsCount == fpsLimit)
    {
        char cTitle[256];
        float fps = 1000.0f / sdkGetAverageTimerValue(&timer);
        sprintf(cTitle, "Neural Implicit Mapping (%d x %d): %.1f fps", window_width, window_height, fps);
        glutSetWindowTitle(cTitle);
        //printf("%s\n", cTitle);
        fpsCount = 0;
        fpsLimit = (int)((fps > 1.0f) ? fps : 1.0f);
        sdkResetTimer(&timer);
    }

    if (g_benchmark_frames > 0)
    {
        // The interactive loop redraws on a REFRESH_DELAY (10 ms) timer, which
        // caps it near 100 FPS. Benchmarks must be GPU-bound, so schedule the
        // next frame immediately.
        glutPostRedisplay();
        if (g_bench_warmup > 0) {
            if (--g_bench_warmup == 0)
                g_bench_t0 = std::chrono::high_resolution_clock::now();
        }
        else if (++g_bench_counted == g_benchmark_frames) {
            save_snapshot = true;   // final frame is dumped as img<N>.ppm next display()
        }
        else if (g_bench_counted > g_benchmark_frames) {
            const double secs = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - g_bench_t0).count();
            const double bench_fps = g_bench_counted / secs;
            printf("BENCHMARK resolution=%dx%d iters=%d/%d/%d delta=%.4f normal_lod=%d frames=%d avg_fps=%.1f\n",
                   window_width, window_height, lod_0_sphere_tracing_iters, lod_1_sphere_tracing_iters,
                   lod_2_sphere_tracing_iters, lod_0_delta, lod_to_show, g_bench_counted, bench_fps);
            FILE* bench_file = fopen("benchmark.csv", "a");
            if (bench_file) {
                fprintf(bench_file, "%dx%d,%d/%d/%d,%.4f,%d,%d,%.2f\n",
                        window_width, window_height, lod_0_sphere_tracing_iters, lod_1_sphere_tracing_iters,
                        lod_2_sphere_tracing_iters, lod_0_delta, lod_to_show, g_bench_counted, bench_fps);
                fclose(bench_file);
            }
            Cleanup(EXIT_SUCCESS);
        }
    }
}

// This function is called when a mouse button is pressed or released
void onMouse(int button, int state, int x, int y) {
    ImGui_ImplGLUT_MouseFunc(button, state, x, y);
    
    if (!ImGui::GetIO().WantCaptureMouse) {
        if (button == GLUT_LEFT_BUTTON || button == GLUT_RIGHT_BUTTON) {
            switch (state) {
            case GLUT_DOWN: {arcball.onClick(x, y, button); break; }
            case GLUT_UP: {arcball.onRelease(button); break; }
            }
        }
    }
}

// This function is called when the mouse is moved while a button is pressed
void onMotion(int x, int y) {
    ImGui_ImplGLUT_MotionFunc(x, y);

    if (!ImGui::GetIO().WantCaptureMouse) {
        // Update the transformation based on the new mouse position
        arcball.onDrag(x, y);
        // Request a redraw of the window
        glutPostRedisplay();
    }
}

void onMouseWheel(int wheel, int direction, int x, int y) {
    ImGui_ImplGLUT_MouseWheelFunc(wheel, direction, x, y);

    if (!ImGui::GetIO().WantCaptureMouse) {
        // Assuming the wheel parameter is not used, adjust if your GLUT passes wheel info
        arcball.onZoom(direction);

        glutPostRedisplay();
    }
}

void timerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void
keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
    case (27):
        show_gui = !show_gui;
        break;
    case ('s'):
        save_snapshot = true;
        break;
    case ' ':
        enable_cuda ^= 1;
#ifdef USE_TEXTURE_RGBA8UI

        if (enable_cuda)
        {
            glClearColorIuiEXT(128, 128, 128, 255);
        }
        else
        {
            glClearColor(0.5, 0.5, 0.5, 1.0);
        }

#endif
        break;
    }
}

void reshape(int w, int h)
{
    int img_size = min(w, h);
    window_width = img_size;
    window_height = img_size;
    arcball.resize(img_size, img_size);
    ImGui_ImplGLUT_ReshapeFunc(w, h);
}

void mainMenu(int i)
{
    keyboard((unsigned char)i, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void
createTextureDst(GLuint* tex_cudaResult, unsigned int size_x, unsigned int size_y)
{
    // create a texture
    glGenTextures(1, tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, *tex_cudaResult);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

#ifdef USE_TEXSUBIMAGE2D
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
#else
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
    // register this texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, *tex_cudaResult,
        GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
#endif
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void
deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    SDK_CHECK_ERROR_GL();

    *tex = 0;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    printf("%s Starting...\n\n", argv[0]);

    if (checkCmdLineFlag(argc, (const char**)argv, "file"))
    {

        getCmdLineArgumentString(argc, (const char**)argv, "file", &ref_file);
    }

    pArgc = &argc;
    pArgv = argv;

    if (checkCmdLineFlag(argc, (const char**)argv, "list")) {
        PrintExperiments();
        return EXIT_SUCCESS;
    }
    char* data_root = NULL;
    if (getCmdLineArgumentString(argc, (const char**)argv, "data_root", &data_root)) {
        g_data_root = data_root;
    }
    if (checkCmdLineFlag(argc, (const char**)argv, "benchmark")) {
        g_benchmark_frames = getCmdLineArgumentInt(argc, (const char**)argv, "benchmark");
        if (g_benchmark_frames <= 0) g_benchmark_frames = 500;
    }

    // Sphere-tracing configuration. Defaults render the coarse level only
    // (20 iterations). The paper's full-detail setting is -iters=20,5,5 with
    // -delta set to the coarse band width; neural normal mapping is
    // -iters=20,5,0 -normal_lod=2 (trace coarse+medium, shade with the fine
    // level's normals). The same values are exposed as ImGui sliders.
    char* cfg = NULL;
    if (getCmdLineArgumentString(argc, (const char**)argv, "iters", &cfg)) {
        int a = lod_0_sphere_tracing_iters, b = lod_1_sphere_tracing_iters, c = lod_2_sphere_tracing_iters;
        if (sscanf(cfg, "%d,%d,%d", &a, &b, &c) >= 1) {
            lod_0_sphere_tracing_iters = a;
            lod_1_sphere_tracing_iters = b;
            lod_2_sphere_tracing_iters = c;
        }
    }
    if (getCmdLineArgumentString(argc, (const char**)argv, "delta", &cfg)) {
        float d0 = lod_0_delta, d1 = lod_1_delta;
        if (sscanf(cfg, "%f,%f", &d0, &d1) >= 1) {
            lod_0_delta = d0;
            lod_1_delta = d1;
        }
    }
    if (checkCmdLineFlag(argc, (const char**)argv, "normal_lod"))
        lod_to_show = getCmdLineArgumentInt(argc, (const char**)argv, "normal_lod");
    if (checkCmdLineFlag(argc, (const char**)argv, "threshold"))
        distance_threshold = getCmdLineArgumentFloat(argc, (const char**)argv, "threshold");
    if (getCmdLineArgumentString(argc, (const char**)argv, "shading", &cfg))
        shading = (std::string(cfg) == "normals") ? NORMALS : PHONG;
    if (checkCmdLineFlag(argc, (const char**)argv, "skip_lod0")) skip_lod_0 = true;
    if (checkCmdLineFlag(argc, (const char**)argv, "no_residual")) is_residual = false;
    printf("sphere tracing: iters=%d/%d/%d delta=%.4f/%.4f normal_lod=%d residual=%d\n",
           lod_0_sphere_tracing_iters, lod_1_sphere_tracing_iters, lod_2_sphere_tracing_iters,
           lod_0_delta, lod_1_delta, lod_to_show, (int)is_residual);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if (checkCmdLineFlag(argc, (const char**)argv, "device"))
    {
        printf("[%s]\n", argv[0]);
        printf("   Does not explicitly support -device=n\n");
        printf("   This sample requires OpenGL.  Only -file=<reference> are supported\n");
        printf("exiting...\n");
        exit(EXIT_WAIVED);
    }

    if (ref_file)
    {
        printf("(Test with OpenGL verification)\n");
        runStdProgram(argc, argv);
    }
    else
    {
        printf("(Interactive OpenGL Demo)\n");
        runStdProgram(argc, argv);
    }

    exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void FreeResource()
{
    sdkDeleteTimer(&timer);

    // unregister this buffer object with CUDA
    //    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_tex_screen_resource));
#ifdef USE_TEXSUBIMAGE2D
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_dest_resource));
    deletePBO(&pbo_dest);
#else
    cudaFree(cuda_dest_resource);
#endif
    deleteTexture(&tex_screen);
    deleteTexture(&tex_cudaResult);

    if (iGLUTWindowHandle)
    {
        glutDestroyWindow(iGLUTWindowHandle);
    }

    // finalize logs and leave
    printf("simpleCUDA2GL Exiting...\n");
}

void Cleanup(int iExitCode)
{
    FreeResource();
    printf("PPM Images are %s\n", (iExitCode == EXIT_SUCCESS) ? "Matching" : "Not Matching");

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGLUT_Shutdown();
    ImGui::DestroyContext();

    exit(iExitCode);
}


////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
GLuint compileGLSLprogram(const char* vertex_shader_src, const char* fragment_shader_src)
{
    GLuint v, f, p = 0;

    p = glCreateProgram();

    if (vertex_shader_src)
    {
        v = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(v, 1, &vertex_shader_src, NULL);
        glCompileShader(v);

        // check if shader compiled
        GLint compiled = 0;
        glGetShaderiv(v, GL_COMPILE_STATUS, &compiled);

        if (!compiled)
        {
            //#ifdef NV_REPORT_COMPILE_ERRORS
            char temp[256] = "";
            glGetShaderInfoLog(v, 256, NULL, temp);
            printf("Vtx Compile failed:\n%s\n", temp);
            //#endif
            glDeleteShader(v);
            return 0;
        }
        else
        {
            glAttachShader(p, v);
        }
    }

    if (fragment_shader_src)
    {
        f = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(f, 1, &fragment_shader_src, NULL);
        glCompileShader(f);

        // check if shader compiled
        GLint compiled = 0;
        glGetShaderiv(f, GL_COMPILE_STATUS, &compiled);

        if (!compiled)
        {
            //#ifdef NV_REPORT_COMPILE_ERRORS
            char temp[256] = "";
            glGetShaderInfoLog(f, 256, NULL, temp);
            printf("frag Compile failed:\n%s\n", temp);
            //#endif
            glDeleteShader(f);
            return 0;
        }
        else
        {
            glAttachShader(p, f);
        }
    }

    glLinkProgram(p);

    int infologLength = 0;
    int charsWritten = 0;

    glGetProgramiv(p, GL_INFO_LOG_LENGTH, (GLint*)&infologLength);

    if (infologLength > 0)
    {
        char* infoLog = (char*)malloc(infologLength);
        glGetProgramInfoLog(p, infologLength, (GLsizei*)&charsWritten, infoLog);
        printf("Shader compilation error: %s\n", infoLog);
        free(infoLog);
    }

    return p;
}

////////////////////////////////////////////////////////////////////////////////
//! Allocate the "render target" of CUDA
////////////////////////////////////////////////////////////////////////////////
#ifndef USE_TEXSUBIMAGE2D
void initCUDABuffers()
{
    // set up vertex data parameter
    num_texels = image_width * image_height;
    num_values = num_texels * 4;
    size_tex_data = sizeof(GLubyte) * num_values;
    checkCudaErrors(cudaMalloc((void**)&cuda_dest_resource, size_tex_data));
    //checkCudaErrors(cudaHostAlloc((void**)&cuda_dest_resource, size_tex_data, ));
}
#endif

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void initGLBuffers()
{
    // create pbo
#ifdef USE_TEXSUBIMAGE2D
    createPBO(&pbo_dest, &cuda_pbo_dest_resource);
#endif
    // create texture that will receive the result of CUDA
    createTextureDst(&tex_cudaResult, image_width, image_height);
    // load shader programs
    shDraw = compileGLSLprogram(NULL, glsl_draw_fragshader_src);

#ifndef USE_TEXSUBIMAGE2D
    shDrawTex = compileGLSLprogram(glsl_drawtex_vertshader_src, glsl_drawtex_fragshader_src);
#endif
    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Run standard demo loop with or without GL verification
////////////////////////////////////////////////////////////////////////////////
void
runStdProgram(int argc, char** argv)
{
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if (false == initGL(&argc, argv))
    {
        return;
    }

    // Now initialize CUDA context (GL context has been created already)
    findCudaDevice(argc, (const char**)argv);

    // Experiment selection: -experiment=<name> (see -list); default: armadillo.
    const Experiment* selected = &armadillo;
    char* exp_file = NULL;
    char* exp_name = NULL;
    // -experiment_file=<descriptor>: any released checkpoint set exported by
    // renderer/scripts/export_experiment.py, without touching the registry.
    // Checked first: the CUDA helper matches flags by prefix, so
    // "-experiment_file" would otherwise be read as "-experiment".
    if (getCmdLineArgumentString(argc, (const char**)argv, "experiment_file", &exp_file)) {
        selected = LoadExperimentFile(exp_file);
        if (!selected) exit(EXIT_FAILURE);
    }
    else if (getCmdLineArgumentString(argc, (const char**)argv, "experiment", &exp_name)) {
        selected = FindExperiment(exp_name);
        if (!selected) {
            fprintf(stderr, "Unknown experiment '%s'.\n", exp_name);
            PrintExperiments();
            exit(EXIT_FAILURE);
        }
    }
    InitSdf(*selected);

    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // create menu
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Toggle GUI", '\033');
    glutAddMenuEntry("Snapshot", 's');
    //glutAttachMenu(GLUT_RIGHT_BUTTON);

    initGLBuffers();
#ifndef USE_TEXSUBIMAGE2D
    initCUDABuffers();
#endif

    // Creating the Auto-Validation Code
    if (ref_file)
    {
        g_CheckRender = new CheckBackBuffer(window_width, window_height, 4);
        g_CheckRender->setPixelFormat(GL_RGBA);
        g_CheckRender->setExecPath(argv[0]);
        g_CheckRender->EnableQAReadback(true);
    }

    printf("\n"
        "\tControls\n"
        "\t(right click mouse button for Menu)\n"
        "\t[esc] - Quit\n\n"
    );

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGLUT_Init();
    ImGui_ImplGLUT_InstallFuncs();
    ImGui_ImplOpenGL3_Init();

    // Registering mouse callbacks after ImGui to overwrite.
    glutMouseFunc(onMouse);
    glutMotionFunc(onMotion);
    glutReshapeFunc(reshape);
    glutMouseWheelFunc(onMouseWheel);

    cam_t1 = std::chrono::high_resolution_clock::now();

    // start rendering mainloop
    glutMainLoop();

    // Normally unused return path
    Cleanup(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool
initGL(int* argc, char** argv)
{
    // Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    iGLUTWindowHandle = glutCreateWindow("MIP-plicits");

    cout << "OpenGL version: " << (const char*)glGetString(GL_VERSION) << endl;

    // initialize necessary OpenGL extensions
    if (!isGLVersionSupported(2, 0) ||
        !areGLExtensionsSupported(
            "GL_ARB_pixel_buffer_object "
            "GL_EXT_framebuffer_object"
        ))
    {
        printf("ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
#ifndef USE_TEXTURE_RGBA8UI
    glClearColor(0.5, 0.5, 0.5, 1.0);
#else
    glClearColorIuiEXT(128, 128, 128, 255);
#endif
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1f, 10.0f);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_LIGHT0);
    float red[] = { 1.0f, 0.1f, 0.1f, 1.0f };
    float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0f);

    SDK_CHECK_ERROR_GL();

    return true;
}