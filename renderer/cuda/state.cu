// ==============
// Global state.
// ==============

#ifndef RESOLUTION
#define RESOLUTION 512   // override via CMake: -DMIP_RESOLUTION=1024
#endif

#include <glm/mat4x4.hpp>
#include <iomanip>

// Experiments
struct Experiment {

    // Coarse Neural Implicit
    int n_lod_0_layers;
    int lod_0_hidden_U_size;
    string lod_0_weights_file;
    string lod_0_biases_file;
    
    // Fine Neural Implicit
    int n_lod_1_layers;
    int lod_1_hidden_U_size;
    string lod_1_weights_file;
    string lod_1_biases_file;
    
    // Normal Mapping Neural Implicit
    int n_normal_layers;
    int normal_hidden_U_size;
    string normal_weights_file;
    string normal_biases_file;
    
    // Textures Neural Implicit
    int n_textures_layers;
    int textures_hidden_U_size;
    string textures_weights_file;
    string textures_biases_file;

    bool is3D;
    int surface_W0;
    int textures_W0;

    // Geometry orientation (previously inferred from checkpoint filename substrings).
    bool swap_y_and_z = false;
    bool invert_z = false;
};

const Experiment thai_statue = {
    1,
    128,
    "data/thai_eccv_coarse_weights.bin", "data/thai_eccv_coarse_biases.bin",

    1,
    256,
    "data/thai_eccv_medium_weights.bin", "data/thai_eccv_medium_biases.bin",

    1,
    400,
    "data/thai_eccv_fine_weights.bin", "data/thai_eccv_fine_biases.bin",

    0,
    0,
    "", "",

    true,
    1,
    0
};

const Experiment asian_dragon = {
    1,
    128,
    "data/asian_dragon_coarse_128x1_weights.bin", "data/asian_dragon_coarse_128x1_biases.bin",

    1,
    256,
    "data/asian_dragon_medium_256x1_weights.bin", "data/asian_dragon_medium_256x1_biases.bin",

    2,
    256,
    "data/asian_dragon_fine_256x2_weights.bin", "data/asian_dragon_fine_256x2_biases.bin",

    0,
    0,
    "", "",

    true,
    1,
    0,

    false,   // swap_y_and_z
    true    // invert_z
};

const Experiment spot_bob_residual_1x64_tex_2x256 = {
    1,
    64,
    "data/spot_bob_coarse_64x1_weights.bin", "data/spot_bob_coarse_64x1_biases.bin",

    1,
    64,
    "data/spot_bob_residual_64x1_weights.bin", "data/spot_bob_residual_64x1_biases.bin",

    1,
    64,
    "data/spot_bob_residual_64x1_weights.bin", "data/spot_bob_residual_64x1_biases.bin",

    0,
    0,
    "", "",

    false,
    1,
    0
};

const Experiment vase_1x256_tex_2x400 = {
    1,
    256,
    "data/textured/vase_surface_256x1_w0_1_weights.bin", "data/textured/vase_surface_256x1_w0_1_biases.bin",

    1,
    256,
    "data/textured/vase_surface_256x1_w0_1_weights.bin", "data/textured/vase_surface_256x1_w0_1_biases.bin",

    1,
    256,
    "data/textured/vase_surface_256x1_w0_1_weights.bin", "data/textured/vase_surface_256x1_w0_1_biases.bin",

    2,
    400,
    "data/textured/vase_texture_400x2_w0_128_weights.bin", "data/textured/vase_texture_400x2_w0_128_biases.bin",

    true,
    1,
    128
};

const Experiment egg_1x64_tex_2x256 = {
    1,
    64,
    "data/textured/egg_net_1x64_w0_10_weights.bin", "data/textured/egg_net_1x64_w0_10_biases.bin",

    1,
    64,
    "data/textured/egg_net_1x64_w0_10_weights.bin", "data/textured/egg_net_1x64_w0_10_biases.bin",

    1,
    64,
    "data/textured/egg_net_1x64_w0_10_weights.bin", "data/textured/egg_net_1x64_w0_10_biases.bin",

    2,
    256,
    "data/textured/egg_net_tex_2x256_w0_60_weights.bin", "data/textured/egg_net_tex_2x256_w0_60_biases.bin",

    true,
    10,
    60
};

const Experiment bob_1x64_tex_2x256 = {
    1,
    64,
    "data/textured/bob_surface_1x64_w0-16_weights.bin", "data/textured/bob_surface_1x64_w0-16_biases.bin",

    1,
    64,
    "data/textured/bob_surface_1x64_w0-16_weights.bin", "data/textured/bob_surface_1x64_w0-16_biases.bin",

    1,
    64,
    "data/textured/bob_surface_1x64_w0-16_weights.bin", "data/textured/bob_surface_1x64_w0-16_biases.bin",

    2,
    256,
    "data/textured/bob_texture_w0-60_weights.bin", "data/textured/bob_texture_w0-60_biases.bin",

    true,
    16,
    60
};

const Experiment lucy = {
    1,
    64,
    "data/lucy_coarse_weights.bin", "data/lucy_coarse_biases.bin",

    1,
    256,
    "data/lucy_medium_weights.bin", "data/lucy_medium_biases.bin",

    1,
    400,
    "data/lucy_fine_weights.bin", "data/lucy_fine_biases.bin",

    0,
    0,
    "", "",

    true,
    1,
    0,

    true,   // swap_y_and_z
    false    // invert_z
};

const Experiment armadillo = {
    1,
    128,
    "data/armadillo_eccv_coarse_weights.bin", "data/armadillo_eccv_coarse_biases.bin",

    1,
    256,
    "data/armadillo_eccv_medium_weights.bin", "data/armadillo_eccv_medium_biases.bin",

    1,
    400,
    "data/armadillo_eccv_fine_weights.bin", "data/armadillo_eccv_fine_biases.bin",

    0,
    0,
    "", "",

    true,
    1,
    0
};

const Experiment armadillo_64 = {
    1,
    64,
    "data/armadillo_coarse_64_weights.bin", "data/armadillo_coarse_64_biases.bin",

    1,
    128,
    "data/armadillo_medium_64_weights.bin", "data/armadillo_medium_64_biases.bin",

    1,
    256,
    "data/armadillo_fine_64_weights.bin", "data/armadillo_fine_64_biases.bin",

    0,
    0,
    "", "",

    true,
    1,
    0
};

const Experiment armadillo_siren = {
    3,
    400,
    "data/armadillo_400x4_weights.bin", "data/armadillo_400x4_biases.bin",

    3,
    400,
    "data/armadillo_400x4_weights.bin", "data/armadillo_400x4_biases.bin",

    3,
    400,
    "data/armadillo_400x4_weights.bin", "data/armadillo_400x4_biases.bin",

    0,
    0,
    "", "",

    true,
    1,
    0
};

const Experiment buddha_neigh_residual_64x1_128x1_256x1 = {
    1,
    64,
    "data/happy_64x1_residual_weights.bin", "data/happy_64x1_residual_biases.bin",

    1,
    128,
    "data/happy_128x1_residual_weights.bin", "data/happy_128x1_residual_biases.bin",

    1,
    256,
    "data/happy_256x1_residual_weights.bin", "data/happy_256x1_residual_biases.bin",

    0,
    0,
    "", "",

    true,
    1,
    0
};

const Experiment buddha_armadillo_256x3 = {
    3,
    256,
    "data/buddha_armadillo_256x3_weights.bin", "data/buddha_armadillo_256x3_biases.bin",
    
    3,
    256,
    "data/buddha_armadillo_256x3_weights.bin", "data/buddha_armadillo_256x3_biases.bin",
    
    3,
    256,
    "data/buddha_armadillo_256x3_weights.bin", "data/buddha_armadillo_256x3_biases.bin",
    
    0,
    0,
    "", "",

    false,
    30,
    0
};

const Experiment buddha_armadillo_256x1_256x2_256x3 = {
    1,
    256,
    "data/buddha_armadillo_256x1_weights.bin", "data/buddha_armadillo_256x1_biases.bin",
    
    2,
    256,
    "data/buddha_armadillo_256x2_eikonal_weights.bin", "data/buddha_armadillo_256x2_eikonal_biases.bin",
    
    3,
    256,
    "data/buddha_armadillo_256x3_weights.bin", "data/buddha_armadillo_256x3_biases.bin",
    
    0,
    0,
    "", "",
    
    false,
    30,
    0
};

const Experiment plank_64x1_64x2_128x2 = {
    1,
    64,
    "data/max_64x1_w0_30_weights.bin", "data/max_64x1_w0_30_biases.bin",
    
    2,
    64,
    "data/max_64x2_w0_30_weights.bin", "data/max_64x2_w0_30_biases.bin",
    
    2,
    128,
    "data/max_128x2_w0_30_weights.bin", "data/max_128x2_w0_30_biases.bin",
    
    0,
    0,
    "", "",

    false,
    30,
    0
};

const Experiment buddha_armadillo_failure = {
    1,
    64,
    "data/buddha_armadillo_64x1_weights.bin", "data/buddha_armadillo_64x1_biases.bin",
    
    1,
    256,
    "data/buddha_armadillo_256x1_weights.bin", "data/buddha_armadillo_256x1_biases.bin",
    
    3,
    256,
    "data/buddha_armadillo_256x3_weights.bin", "data/buddha_armadillo_256x3_biases.bin",
    
    0,
    0,
    "", "",

    false,
    30,
    0
};

const Experiment bunny_64x1_256x1_256x3 = {
    1,
    64,
    "data/bunny_64x1_weights.bin", "data/bunny_64x1_biases.bin",
    
    1,
    256,
    "data/bunny_256x1_weights.bin", "data/bunny_256x1_biases.bin",
    
    3,
    256,
    "data/bunny_256x3_weights.bin", "data/bunny_256x3_biases.bin",
    
    0,
    0,
    "", "",

    true,
    30,
    0,

    false,   // swap_y_and_z
    true    // invert_z
};

const Experiment bunny_256x3 = {
    3,
    256,
    "data/bunny_256x3_weights.bin", "data/bunny_256x3_biases.bin",
    
    1,
    64,
    "data/bunny_64x1_weights.bin", "data/bunny_64x1_biases.bin",
    
    3,
    256,
    "data/bunny_256x3_weights.bin", "data/bunny_256x3_biases.bin",

    0,
    0,
    "", "",

    true,
    30,
    0,

    false,   // swap_y_and_z
    true    // invert_z
};

const Experiment dragon_64x1_256x1_256x3 = {
    1,
    64,
    "data/dragon_64x1_weights.bin", "data/dragon_64x1_biases.bin",
    
    1,
    256,
    "data/dragon_256x1_weights.bin", "data/dragon_256x1_biases.bin",
    
    3,
    256,
    "data/dragon_256x3_weights.bin", "data/dragon_256x3_biases.bin",
    
    0,
    0,
    "", "",
    
    true,
    30,
    0,

    false,   // swap_y_and_z
    true    // invert_z
};

const Experiment dragon_128x1_256x3 = {
    1,
    128,
    "data/dragon_128x1_weights.bin", "data/dragon_128x1_biases.bin",
    
    1,
    128,
    "data/dragon_128x1_weights.bin", "data/dragon_128x1_biases.bin",
    
    3,
    256,
    "data/dragon_256x3_weights.bin", "data/dragon_256x3_biases.bin",
    
    0,
    0,
    "", "",
    
    true,
    30,
    0,

    false,   // swap_y_and_z
    true    // invert_z
};

const Experiment dragon_256x3 = {
    3,
    256,
    "data/dragon_256x3_weights.bin", "data/dragon_256x3_biases.bin",
    
    1,
    128,
    "data/dragon_128x1_weights.bin", "data/dragon_128x1_biases.bin",
    
    3,
    256,
    "data/dragon_256x3_weights.bin", "data/dragon_256x3_biases.bin",
    
    0,
    0,
    "", "",

    true,
    30,
    0,

    false,   // swap_y_and_z
    true    // invert_z
};

const Experiment buddha_64x1_256x2_256x3 = {
    1,
    64,
    "data/buddha_64x1_weights.bin", "data/buddha_64x1_biases.bin",
    
    2,
    256,
    "data/buddha_256x2_weights.bin", "data/buddha_256x2_biases.bin",
    
    3,
    256,
    "data/buddha_256x3_weights.bin", "data/buddha_256x3_biases.bin",
    
    0,
    0,
    "", "",

    true,
    30,
    0,

    false,   // swap_y_and_z
    true    // invert_z
};

const Experiment buddha_256x3 = {
    3,
    256,
    "data/buddha_256x3_weights.bin", "data/buddha_256x3_biases.bin",
    
    1,
    128,
    "data/buddha_128x1_weights.bin", "data/buddha_128x1_biases.bin",
    
    3,
    256,
    "data/buddha_256x3_weights.bin", "data/buddha_256x3_biases.bin",
    
    0,
    0,
    "", "",
    
    true,
    30,
    0,

    false,   // swap_y_and_z
    true    // invert_z
};

const Experiment falcon_64x1_128x2 = {
    1,
    64,
    "data/falcon_witch_64x1_w0_20_t_-0.2_0.2_weights.bin", "data/falcon_witch_64x1_w0_20_t_-0.2_0.2_biases.bin",
    
    2,
    128,
    "data/falcon_witch_2x128_w-20_weights.bin", "data/falcon_witch_2x128_w-20_biases.bin",
    
    2,
    128,
    "data/falcon_witch_2x128_w-20_weights.bin", "data/falcon_witch_2x128_w-20_biases.bin",
    
    0,
    0,
    "", "",
    
    false,
    20,
    0
};

// ---------------------------------------------------------------------------
// Experiment registry: runtime selection by name (-experiment=<name>).
// ---------------------------------------------------------------------------
#include <map>

static const std::map<std::string, const Experiment*> experiment_registry = {
    {"thai_statue", &thai_statue},
    {"asian_dragon", &asian_dragon},
    {"spot_bob_4d", &spot_bob_residual_1x64_tex_2x256},
    {"vase_textured", &vase_1x256_tex_2x400},
    {"egg_textured", &egg_1x64_tex_2x256},
    {"bob_textured", &bob_1x64_tex_2x256},
    {"lucy", &lucy},
    {"armadillo", &armadillo},
    {"armadillo_64", &armadillo_64},
    {"armadillo_siren", &armadillo_siren},
    {"buddha", &buddha_neigh_residual_64x1_128x1_256x1},
    {"buddha_armadillo_256x3", &buddha_armadillo_256x3},
    {"buddha_armadillo_multiscale", &buddha_armadillo_256x1_256x2_256x3},
    {"plank", &plank_64x1_64x2_128x2},
    {"buddha_armadillo_failure", &buddha_armadillo_failure},
    {"bunny_multiscale", &bunny_64x1_256x1_256x3},
    {"bunny_256x3", &bunny_256x3},
    {"dragon_multiscale", &dragon_64x1_256x1_256x3},
    {"dragon_128x1_256x3", &dragon_128x1_256x3},
    {"dragon_256x3", &dragon_256x3},
    {"buddha_multiscale", &buddha_64x1_256x2_256x3},
    {"buddha_256x3", &buddha_256x3},
    {"falcon_4d", &falcon_64x1_128x2},
};

const Experiment* FindExperiment(const std::string& name) {
    auto it = experiment_registry.find(name);
    return it == experiment_registry.end() ? nullptr : it->second;
}

void PrintExperiments() {
    printf("Available experiments (-experiment=<name>):\n");
    for (const auto& kv : experiment_registry)
        printf("  %s\n", kv.first.c_str());
}

class MIPplicitBase
{
public:
    virtual void InitRays() = 0;
    virtual void InitSdf(const Experiment& experiment) = 0;
    virtual void SdfInference(int lod_0_sphere_tracing_iters, int lod_1_sphere_tracing_iters, int lod_2_sphere_tracing_iters, float lod_0_delta,
        float lod_1_delta, float time, float cam_time, float distance_threshold, uint* out_img_data, int lod_to_show, bool skip_lod_0, bool is_residual, Shading shading, bool swap_y_and_z, bool invert_z, float* h_inv_view_matrix, float* h_inv_proj_matrix) = 0;
    virtual void SdfInference() = 0;
    virtual void ImageFromInferenceGpu(float distance_threshold) = 0;
    virtual void ImageFromInference() = 0;
    virtual void SaveInferenceImage(bool gpu_image = true) = 0;
    virtual ~MIPplicitBase() {}
};

template<typename LayerGemms, typename NormalLayerGemms, typename TextureGemms>
class MIPplicit : public MIPplicitBase{

public:
    using Sdf = NeuralImplicit<LayerGemms>;
    using NormalCoord = Normals<NormalLayerGemms, LayerGemms::W0_>;
    using Textures = NeuralImplicit<TextureGemms>;

    ~MIPplicit() {

        if (stream) {

            CUDA_CHECK_THROW(cudaStreamDestroy(stream));
        }
        if (inv_view_matrix)
        {
            cudaFree(inv_view_matrix);
        }
        if (inv_proj_matrix)
        {
            cudaFree(inv_proj_matrix);
        }
    }

    void InitRays() override {

        // Init rays in orthographic projection. Assuming Siren model (geometry inside cube [-1, 1]^3).
        int resolution_width = LayerGemms::resolution_x;
        int resolution_height = LayerGemms::resolution_y;
        auto input_tensor = neural_implicit_lod_0.input_tensor.host_view();

        ray_directions.reset({ 3, resolution_width * resolution_height });

        for (int i = 0; i < resolution_width; ++i) {

            for (int j = 0; j < resolution_height; ++j) {

                int idx = Linearize(i, j, resolution_width, resolution_height); // Linearization and input size offset.;
                input_tensor.at({ 0, idx }) = precision_t(2.f) * i / precision_t(resolution_width) - precision_t(1.f); // almost normalized device coords [-2, 2].
                input_tensor.at({ 1, idx }) = precision_t(2.f) * j / precision_t(resolution_height) - precision_t(1.f); // almost normalized device coords [-2, 2].
                input_tensor.at({ 2, idx }) = precision_t(-2.f);// Fix z near the Siren model in the negative axis.
                input_tensor.at({ 3, idx }) = precision_t(0.f); // 0 is the simulation starting time.

                // Ray direction is the Z axis.
                ray_directions.at({ 0, idx }) = precision_t(0.f);
                ray_directions.at({ 1, idx }) = precision_t(0.f);
                ray_directions.at({ 2, idx }) = precision_t(1.f);
            }
        }

        neural_implicit_lod_0.input_tensor.sync_device();
        ray_directions.sync_device();
    }

    void InitSdf(const Experiment& experiment) override {

        CUDA_CHECK_THROW(cudaStreamCreate(&stream));

        neural_implicit_lod_0.init(experiment.n_lod_0_layers, experiment.lod_0_hidden_U_size, 1, experiment.lod_0_weights_file, experiment.lod_0_biases_file);
        neural_implicit_lod_1.init(experiment.n_lod_1_layers, experiment.lod_1_hidden_U_size, 1, experiment.lod_1_weights_file, experiment.lod_1_biases_file);
        neural_implicit_lod_2.init(experiment.n_normal_layers, experiment.normal_hidden_U_size, 1, experiment.normal_weights_file, experiment.normal_biases_file);

        normals_lod_0_x.init(experiment.n_lod_0_layers, experiment.lod_0_hidden_U_size, 1, experiment.lod_0_weights_file, experiment.lod_0_biases_file, 0);
        normals_lod_0_y.init(experiment.n_lod_0_layers, experiment.lod_0_hidden_U_size, 1, experiment.lod_0_weights_file, experiment.lod_0_biases_file, 1);
        normals_lod_0_z.init(experiment.n_lod_0_layers, experiment.lod_0_hidden_U_size, 1, experiment.lod_0_weights_file, experiment.lod_0_biases_file, 2);

        normals_lod_1_x.init(experiment.n_lod_1_layers, experiment.lod_1_hidden_U_size, 1, experiment.lod_1_weights_file, experiment.lod_1_biases_file, 0);
        normals_lod_1_y.init(experiment.n_lod_1_layers, experiment.lod_1_hidden_U_size, 1, experiment.lod_1_weights_file, experiment.lod_1_biases_file, 1);
        normals_lod_1_z.init(experiment.n_lod_1_layers, experiment.lod_1_hidden_U_size, 1, experiment.lod_1_weights_file, experiment.lod_1_biases_file, 2);

        normals_lod_2_x.init(experiment.n_normal_layers, experiment.normal_hidden_U_size, 1, experiment.normal_weights_file, experiment.normal_biases_file, 0);
        normals_lod_2_y.init(experiment.n_normal_layers, experiment.normal_hidden_U_size, 1, experiment.normal_weights_file, experiment.normal_biases_file, 1);
        normals_lod_2_z.init(experiment.n_normal_layers, experiment.normal_hidden_U_size, 1, experiment.normal_weights_file, experiment.normal_biases_file, 2);


        if (experiment.n_textures_layers > 0)
        {
            has_textures = true;
            textures.init(experiment.n_textures_layers, experiment.textures_hidden_U_size, 3, experiment.textures_weights_file, experiment.textures_biases_file);
        }

        cudaMalloc(&inv_view_matrix, sizeof(float) * 16); // Allocate space for a 4x4 matrix
        cudaMalloc(&inv_proj_matrix, sizeof(float) * 16); // Allocate space for a 4x4 matrix

        ray_directions.reset({ 3, LayerGemms::resolution_x * LayerGemms::resolution_y });
        out_img.reset(cutlass::make_Coord(LayerGemms::resolution_x, LayerGemms::resolution_y));
    }

    void SdfInference(int lod_0_sphere_tracing_iters, int lod_1_sphere_tracing_iters, int lod_2_sphere_tracing_iters, float lod_0_delta, float lod_1_delta, float time, float cam_time, float distance_threshold,
        uint* out_img_data, int lod_to_show, bool skip_lod_0, bool is_residual, Shading shading, bool swap_y_and_z, bool invert_z, float* h_inv_view_matrix, float* h_inv_proj_matrix) override {

        // DEBUG
        /*{
            cout << "res: " << is_residual << endl;
        }*/

        uint2 resolution = make_uint2(LayerGemms::resolution_x, LayerGemms::resolution_y);
        dim3 block(LayerGemms::block_x, LayerGemms::block_y, 1);
        dim3 grid(resolution.x / block.x, resolution.y / block.y, 1);
        int sbytes = 0;
        
        auto origins = neural_implicit_lod_0.input_tensor.device_data();
        auto directions = ray_directions.device_data();

        auto distances_lod_0 = neural_implicit_lod_0.output_gemm.tensor_d.device_data();
        auto distances_lod_1 = neural_implicit_lod_1.output_gemm.tensor_d.device_data();
        auto distances_lod_2 = neural_implicit_lod_0.output_gemm.tensor_d.device_data();

        if (last_lod_0_sphere_tracing_iters != lod_0_sphere_tracing_iters || last_lod_1_sphere_tracing_iters != lod_1_sphere_tracing_iters || last_lod_2_sphere_tracing_iters != lod_2_sphere_tracing_iters || last_skip_lod_0 != skip_lod_0) {

            graph = CudaGraph(); // Reset graph because of loop index.
            last_lod_0_sphere_tracing_iters = lod_0_sphere_tracing_iters;
            last_lod_1_sphere_tracing_iters = lod_1_sphere_tracing_iters;
            last_lod_2_sphere_tracing_iters = lod_2_sphere_tracing_iters;
            last_skip_lod_0 = skip_lod_0;
        }

        cudaMemcpy(inv_view_matrix, h_inv_view_matrix, sizeof(float) * 16, cudaMemcpyHostToDevice);
        cudaMemcpy(inv_proj_matrix, h_inv_proj_matrix, sizeof(float) * 16, cudaMemcpyHostToDevice);

        graph.capture_and_execute(stream, false, [&]() {

            InitializeRays_kernel << <grid, block, sbytes, stream >> > (precision_t(time), precision_t(cam_time), origins, directions, resolution, LayerGemms::point_size, swap_y_and_z, invert_z, inv_view_matrix, inv_proj_matrix);

            if (!skip_lod_0) {

                for (int i = 0; i < lod_0_sphere_tracing_iters; ++i) {

                    neural_implicit_lod_0.inference(stream, neural_implicit_lod_0.input_tensor);

                    IterateSphereTracing_kernel << < grid, block, sbytes, stream >> > (origins, directions, distances_lod_0, lod_0_delta, distances_lod_1, lod_1_delta, distances_lod_2, resolution, true, false, false, is_residual, LayerGemms::point_size);
                }
            }

            for (int i = 0; i < lod_1_sphere_tracing_iters; ++i) {

                neural_implicit_lod_0.inference(stream, neural_implicit_lod_0.input_tensor);
                neural_implicit_lod_1.inference(stream, neural_implicit_lod_0.input_tensor);

                IterateSphereTracing_kernel << < grid, block, sbytes, stream >> > (origins, directions, distances_lod_0, lod_0_delta, distances_lod_1, lod_1_delta, distances_lod_2, resolution, true, true, false, is_residual, LayerGemms::point_size);
            }


            for (int i = 0; i < lod_2_sphere_tracing_iters; ++i) {

                neural_implicit_lod_0.inference(stream, neural_implicit_lod_0.input_tensor);
                if (is_residual) {
                    neural_implicit_lod_1.inference(stream, neural_implicit_lod_0.input_tensor);
                }
                neural_implicit_lod_2.inference(stream, neural_implicit_lod_0.input_tensor);

                IterateSphereTracing_kernel << < grid, block, sbytes, stream >> > (origins, directions, distances_lod_0, lod_0_delta, distances_lod_1, lod_1_delta, distances_lod_2, resolution, true, true, true, is_residual, LayerGemms::point_size);
            }
        });

        // Calculate texture colors
        precision_t* tex_colors = nullptr;
        if (has_textures)
        {
            textures.inference(stream, neural_implicit_lod_0.input_tensor);
            tex_colors = textures.output_gemm.tensor_d.device_data();
        }

        // Calculate the distance to lod 0 to check whether the point is inside the lod 0 neighborhood in rendering.
        /*auto distances_lod_0 = neural_implicit_lod_0.output_gemm.tensor_d.device_data();

        if (lod_1_sphere_tracing_iters > 0 || lod_2_sphere_tracing_iters > 0)
        {
            neural_implicit_lod_0.inference(stream, neural_implicit_lod_0.input_tensor);
        }
        else
        {
            distances_lod_0 = nullptr;
        }*/

        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

        NormalCoord* normals_x;
        NormalCoord* normals_y;
        NormalCoord* normals_z;

        if (!is_residual) {
            switch (lod_to_show) {

            case 0: {
                normals_x = &normals_lod_0_x;
                normals_y = &normals_lod_0_y;
                normals_z = &normals_lod_0_z;
                break;
            }
            case 1: {
                normals_x = &normals_lod_1_x;
                normals_y = &normals_lod_1_y;
                normals_z = &normals_lod_1_z;
                break;
            }
            case 2: {
                normals_x = &normals_lod_2_x;
                normals_y = &normals_lod_2_y;
                normals_z = &normals_lod_2_z;
                break;
            }
            }

            auto& input_tensor = neural_implicit_lod_0.input_tensor;
            auto origins_ = input_tensor.device_data();
            normals_x->inference(input_tensor);
            normals_y->inference(input_tensor);
            normals_z->inference(input_tensor);

            CUDA_CHECK_THROW(cudaStreamSynchronize(normals_x->_stream));
            CUDA_CHECK_THROW(cudaStreamSynchronize(normals_y->_stream));
            CUDA_CHECK_THROW(cudaStreamSynchronize(normals_z->_stream));

            auto nx = normals_x->output_gemm_G.tensor_d.device_data();
            auto ny = normals_y->output_gemm_G.tensor_d.device_data();
            auto nz = normals_z->output_gemm_G.tensor_d.device_data();

            bool use_lod_0 = false;
            bool use_lod_1 = false;
            bool use_lod_2 = false;

            if (!is_residual) {
                if (lod_2_sphere_tracing_iters > 0)
                    use_lod_2 = true;
                else {
                    if (lod_1_sphere_tracing_iters > 0)
                        use_lod_1 = true;
                    else
                        use_lod_0 = true;
                }
            }
            else {
                if (lod_2_sphere_tracing_iters > 0) {
                    use_lod_1 = true;
                    use_lod_2 = true;
                }
                else if (lod_1_sphere_tracing_iters > 0)
                    use_lod_1 = true;
                use_lod_0 = true;
            }

            Shade_kernel << < grid, block, sbytes, stream >> > (origins_, directions, distances_lod_0, distances_lod_1, distances_lod_2, use_lod_0, use_lod_1, use_lod_2, is_residual, precision_t(distance_threshold), nx, ny, nz, tex_colors, out_img_data, resolution, shading, LayerGemms::point_size);
        }
        else {
            auto& input_tensor = neural_implicit_lod_0.input_tensor;
            auto origins_ = input_tensor.device_data();

            normals_lod_0_x.inference(input_tensor);
            normals_lod_0_y.inference(input_tensor);
            normals_lod_0_z.inference(input_tensor);

            switch (lod_to_show) {
            case 1: {
                normals_lod_1_x.inference(input_tensor);
                normals_lod_1_y.inference(input_tensor);
                normals_lod_1_z.inference(input_tensor);
                break;
            }
            case 2: {
                normals_lod_1_x.inference(input_tensor);
                normals_lod_1_y.inference(input_tensor);
                normals_lod_1_z.inference(input_tensor);
                
                normals_lod_2_x.inference(input_tensor);
                normals_lod_2_y.inference(input_tensor);
                normals_lod_2_z.inference(input_tensor);
                break;
            }
            }

            auto nx_l_0 = normals_lod_0_x.output_gemm_G.tensor_d.device_data();
            auto ny_l_0 = normals_lod_0_y.output_gemm_G.tensor_d.device_data();
            auto nz_l_0 = normals_lod_0_z.output_gemm_G.tensor_d.device_data();

            precision_t* nx_l_1 = nullptr;
            precision_t* ny_l_1 = nullptr;
            precision_t* nz_l_1 = nullptr;

            precision_t* nx_l_2 = nullptr;
            precision_t* ny_l_2 = nullptr;
            precision_t* nz_l_2 = nullptr;

            CUDA_CHECK_THROW(cudaStreamSynchronize(normals_lod_0_x._stream));
            CUDA_CHECK_THROW(cudaStreamSynchronize(normals_lod_0_y._stream));
            CUDA_CHECK_THROW(cudaStreamSynchronize(normals_lod_0_z._stream));

            if (lod_to_show == 1 || lod_to_show == 2) {
                nx_l_1 = normals_lod_1_x.output_gemm_G.tensor_d.device_data();
                ny_l_1 = normals_lod_1_y.output_gemm_G.tensor_d.device_data();
                nz_l_1 = normals_lod_1_z.output_gemm_G.tensor_d.device_data();

                CUDA_CHECK_THROW(cudaStreamSynchronize(normals_lod_1_x._stream));
                CUDA_CHECK_THROW(cudaStreamSynchronize(normals_lod_1_y._stream));
                CUDA_CHECK_THROW(cudaStreamSynchronize(normals_lod_1_z._stream));
            }
            
            if (lod_to_show == 2) {
                nx_l_2 = normals_lod_2_x.output_gemm_G.tensor_d.device_data();
                ny_l_2 = normals_lod_2_y.output_gemm_G.tensor_d.device_data();
                nz_l_2 = normals_lod_2_z.output_gemm_G.tensor_d.device_data();

                CUDA_CHECK_THROW(cudaStreamSynchronize(normals_lod_2_x._stream));
                CUDA_CHECK_THROW(cudaStreamSynchronize(normals_lod_2_y._stream));
                CUDA_CHECK_THROW(cudaStreamSynchronize(normals_lod_2_z._stream));
            }

            auto nx = normals_lod_2_x.output_gemm_G.tensor_d.device_data();
            auto ny = normals_lod_2_y.output_gemm_G.tensor_d.device_data();
            auto nz = normals_lod_2_z.output_gemm_G.tensor_d.device_data();

            sum_residual_normals << < grid, block, sbytes, stream >> > (nx_l_0, ny_l_0, nz_l_0, nx_l_1, ny_l_1, nz_l_1, nx_l_2, ny_l_2, nz_l_2, nx, ny, nz, resolution);

            bool use_lod_0 = false;
            bool use_lod_1 = false;
            bool use_lod_2 = false;

            if (lod_2_sphere_tracing_iters > 0) {
                use_lod_1 = true;
                use_lod_2 = true;
            }
            else if (lod_1_sphere_tracing_iters > 0)
                use_lod_1 = true;
            use_lod_0 = true;

            Shade_kernel << < grid, block, sbytes, stream >> > (origins_, directions, distances_lod_0, distances_lod_1, distances_lod_2, use_lod_0, use_lod_1, use_lod_2, is_residual, precision_t(distance_threshold), nx, ny, nz, tex_colors, out_img_data, resolution, shading, LayerGemms::point_size);
        }
    }

    void SdfInference() override {

        graph.capture_and_execute(stream, false, [&]() {
            neural_implicit_lod_0.inference(stream, neural_implicit_lod_0.input_tensor);
        });
    }

    void ImageFromInferenceGpu(float distance_threshold) override {

        // GPU version.
        uint* out_img_data = out_img.device_data();

        dim3 block(LayerGemms::block_x, LayerGemms::block_y, 1);
        dim3 grid(LayerGemms::resolution_x / block.x, LayerGemms::resolution_y / block.y, 1);
        int sbytes = 0;
        uint2 resolution = make_uint2(LayerGemms::resolution_x, LayerGemms::resolution_y);

        auto origins = neural_implicit_lod_0.input_tensor.device_data();
        auto distances = neural_implicit_lod_0.output_gemm.tensor_d.device_data();

        ImageFromInference_kernel << < grid, block, sbytes, stream >> > (origins, distances, precision_t(distance_threshold), out_img_data, resolution, LayerGemms::point_size);
        getLastCudaError("ImageFromInference_kernel");

        out_img.sync_host();
    }

    void ImageFromInference() override {

        // CPU version.
        auto distances = neural_implicit_lod_0.output_gemm.tensor_d;
        distances.sync_host();
        auto distances_host = distances.host_view();
        auto extent = out_img.extent();

        for (int i = 0; i < extent.row(); ++i) {

            for (int j = 0; j < extent.column(); ++j) {

                auto idx = Linearize(i, j, extent.row(), extent.column());
                float color = distances_host.at({ 0, idx });
                //out_img.host_view().at({ i, j }) = (abs(color) < 1e-6) ? RgbaFloatToInt(make_float4(0.f, 0.f, 1.f, 1.f)) : RgbaFloatToInt(make_float4(1.f, 0.f, 0.f, 1.f));
                out_img.host_view().at({ i, j }) = (color < 0.f) ? RgbaFloatToInt(make_float4(1.f, 0.f, 0.f, 1.f)) : RgbaFloatToInt(make_float4(0.f, 0.f, 1.f, 1.f));
                //out_img.host_view().at({ i, j }) = RgbaFloatToInt(make_float4(0.f, 0.f, color, 1.f));
            }
        }
    }

    void SaveInferenceImage(bool gpu_image = true) override {

        auto extent = out_img.extent();
        string filename = (gpu_image) ? "inference_gpu_image.ppm" : "inference_cpu_image.ppm";

        cout << "Saving inference image " << filename << endl;
        out_img.sync_host();
        sdkSavePPM4ub(filename.c_str(), (unsigned char*)out_img.host_data(), extent.row(), extent.column());
    }

private:
    cudaStream_t stream = nullptr;
    CudaGraph graph;

    Sdf neural_implicit_lod_0;
    Sdf neural_implicit_lod_1;
    Sdf neural_implicit_lod_2;

    NormalCoord normals_lod_0_x;
    NormalCoord normals_lod_0_y;
    NormalCoord normals_lod_0_z;

    NormalCoord normals_lod_1_x;
    NormalCoord normals_lod_1_y;
    NormalCoord normals_lod_1_z;

    NormalCoord normals_lod_2_x;
    NormalCoord normals_lod_2_y;
    NormalCoord normals_lod_2_z;

    Textures textures;

    float* inv_view_matrix = nullptr;
    float* inv_proj_matrix = nullptr;
    cutlass::HostTensor<precision_t, LayerGemms::LayoutInputA> ray_directions;
    cutlass::HostTensor<uint, cutlass::layout::ColumnMajor> out_img;

    int last_lod_0_sphere_tracing_iters = 0;
    int last_lod_1_sphere_tracing_iters = 0;
    int last_lod_2_sphere_tracing_iters = 0;
    bool last_skip_lod_0 = false;
    bool has_textures = false;
};

unique_ptr<MIPplicitBase> mip_plicit_3d;
unique_ptr<MIPplicitBase> mip_plicit_4d;
Experiment experiment;
int snapshot_counter = 0;

// ================
// State modifiers.
// ================

void InitRays() {

    mip_plicit_4d->InitRays();
}

void InitSdf(const Experiment& experiment_) {

    experiment = experiment_;

    switch(experiment.surface_W0)
    {
    #if RESOLUTION == 512
        case 30:
        {
            mip_plicit_3d = make_unique<MIPplicit<RTX30903DGemms_512x512<30>, RTX30903DNormalGemms_512x512, RTX30903DGemms_512x512<60>>>();
            mip_plicit_4d = make_unique<MIPplicit<RTX3090Gemms_256_hidden_512x512<30>, RTX3090NormalGemms_256_hidden_512x512, RTX3090Gemms_256_hidden_512x512<60>>>();
            break;
        }
        case 20:
        {
            mip_plicit_3d = make_unique<MIPplicit<RTX30903DGemms_512x512<20>, RTX30903DNormalGemms_512x512, RTX30903DGemms_512x512<60>>>();
            mip_plicit_4d = make_unique<MIPplicit<RTX3090Gemms_256_hidden_512x512<20>, RTX3090NormalGemms_256_hidden_512x512, RTX3090Gemms_256_hidden_512x512<60>>>();
            break;
        }
        case 16:
        {
            mip_plicit_3d = make_unique<MIPplicit<RTX30903DGemms_512x512<16>, RTX30903DNormalGemms_512x512, RTX30903DGemms_512x512<60>>>();
            mip_plicit_4d = make_unique<MIPplicit<RTX3090Gemms_256_hidden_512x512<16>, RTX3090NormalGemms_256_hidden_512x512, RTX3090Gemms_256_hidden_512x512<60>>>();
            break;
        }
        case 10:
        {
            mip_plicit_3d = make_unique<MIPplicit<RTX30903DGemms_512x512<10>, RTX30903DNormalGemms_512x512, RTX30903DGemms_512x512<60>>>();
            mip_plicit_4d = make_unique<MIPplicit<RTX3090Gemms_256_hidden_512x512<10>, RTX3090NormalGemms_256_hidden_512x512, RTX3090Gemms_256_hidden_512x512<60>>>();
            break;
        }
        case 1:
        {
            mip_plicit_3d = make_unique<MIPplicit<RTX30903DGemms_512x512<1>, RTX30903DNormalGemms_512x512, RTX30903DGemms_512x512<128>>>();
            mip_plicit_4d = make_unique<MIPplicit<RTX3090Gemms_256_hidden_512x512<1>, RTX3090NormalGemms_256_hidden_512x512, RTX3090Gemms_256_hidden_512x512<60>>>();
            break;
        }
    #else
        case 30:
        {
            mip_plicit_3d = make_unique<MIPplicit<RTX30903DGemms_1024x1024<30>, RTX30903DNormalGemms_1024x1024, RTX30903DGemms_1024x1024<60>>>();
            mip_plicit_4d = make_unique<MIPplicit<RTX3090Gemms_1024x1024<30>, RTX3090NormalGemms_1024x1024, RTX3090Gemms_1024x1024<60>>>();
            break;
        }
        case 20:
        {
            mip_plicit_3d = make_unique<MIPplicit<RTX30903DGemms_1024x1024<20>, RTX30903DNormalGemms_1024x1024, RTX30903DGemms_1024x1024<60>>>();
            mip_plicit_4d = make_unique<MIPplicit<RTX3090Gemms_1024x1024<20>, RTX3090NormalGemms_1024x1024, RTX3090Gemms_1024x1024<60>>>();
            break;
        }
        case 16:
        {
            mip_plicit_3d = make_unique<MIPplicit<RTX30903DGemms_1024x1024<16>, RTX30903DNormalGemms_1024x1024, RTX30903DGemms_1024x1024<60>>>();
            mip_plicit_4d = make_unique<MIPplicit<RTX3090Gemms_1024x1024<16>, RTX3090NormalGemms_1024x1024, RTX3090Gemms_1024x1024<60>>>();
            break;
        }
        case 10:
        {
            mip_plicit_3d = make_unique<MIPplicit<RTX30903DGemms_1024x1024<10>, RTX30903DNormalGemms_1024x1024, RTX30903DGemms_1024x1024<60>>>();
            mip_plicit_4d = make_unique<MIPplicit<RTX3090Gemms_1024x1024<10>, RTX3090NormalGemms_1024x1024, RTX3090Gemms_1024x1024<60>>>();
            break;
        }
        case 1:
        {
            mip_plicit_3d = make_unique<MIPplicit<RTX30903DGemms_1024x1024<1>, RTX30903DNormalGemms_1024x1024, RTX30903DGemms_1024x1024<60>>>();
            mip_plicit_4d = make_unique<MIPplicit<RTX3090Gemms_1024x1024<1>, RTX3090NormalGemms_1024x1024, RTX3090Gemms_1024x1024<60>>>();
            break;
        }
    #endif
        default:
        {
            throw runtime_error("Unexpected W0: " + std::to_string(experiment.surface_W0));
        }
    }

    if (experiment.is3D) {

        mip_plicit_3d->InitSdf(experiment);
    }
    else {

        mip_plicit_4d->InitSdf(experiment);
    }
}

void SdfInference(int lod_0_sphere_tracing_iters, int lod_1_sphere_tracing_iters, int lod_2_sphere_tracing_iters, float lod_0_delta,
    float lod_1_delta, float time, float cam_time, float distance_threshold, uint* out_img_data, int lod_to_show, bool skip_lod_0, bool is_residual, Shading shading, float* h_inv_view_matrix, float* h_inv_proj_matrix) {

    bool swap_y_and_z = experiment.swap_y_and_z;
    bool invert_z = experiment.invert_z;

    if (experiment.is3D) {

        mip_plicit_3d->SdfInference(lod_0_sphere_tracing_iters, lod_1_sphere_tracing_iters, lod_2_sphere_tracing_iters, lod_0_delta, lod_1_delta, time, cam_time, distance_threshold, out_img_data, lod_to_show, skip_lod_0, is_residual, shading,
            swap_y_and_z, invert_z, h_inv_view_matrix, h_inv_proj_matrix);
    }
    else {

        mip_plicit_4d->SdfInference(lod_0_sphere_tracing_iters, lod_1_sphere_tracing_iters, lod_2_sphere_tracing_iters, lod_0_delta, lod_1_delta, time, cam_time, distance_threshold, out_img_data, lod_to_show, skip_lod_0, is_residual, shading,
            swap_y_and_z, invert_z, h_inv_view_matrix, h_inv_proj_matrix);
    }
}

void SdfInference() {

    if (experiment.is3D) {

        mip_plicit_3d->SdfInference();
    }
    else {

        mip_plicit_4d->SdfInference();
    }
}

void ImageFromInferenceGpu(float distance_threshold) {

    if (experiment.is3D) {

        mip_plicit_3d->ImageFromInferenceGpu(distance_threshold);
    }
    else {

        mip_plicit_4d->ImageFromInferenceGpu(distance_threshold);
    }
}

void ImageFromInference() {

    if (experiment.is3D) {

        mip_plicit_3d->ImageFromInference();
    }
    else {

        mip_plicit_4d->ImageFromInference();
    }
}

void SaveInferenceImage(bool gpu_image = true) {

    if (experiment.is3D) {

        mip_plicit_3d->SaveInferenceImage(gpu_image);
    }
    else {

        mip_plicit_4d->SaveInferenceImage(gpu_image);
    }
}

void SaveInferenceImage(unsigned char* img, uint width, uint height) {

    stringstream ss;
    ss << "img" << snapshot_counter++ << ".ppm";

    cout << "Saving inference image " << ss.str() << endl;
    
    sdkSavePPM4ub(ss.str().c_str(), img, width, height);
}