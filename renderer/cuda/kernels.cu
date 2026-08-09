//===============================
// Kernels and device functions.
//===============================

enum Shading {

    NORMALS = 0,
    PHONG
};

// clamp x to range [a, b]
__device__ __host__ float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ __host__  int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

__device__ __host__ int Linearize(int i, int j, int width, int height) {

    // Input is considered col-major.
    return i + (width * j);
}

// convert floating point rgb color to 8-bit integer
__device__ __host__ unsigned int RgbaFloatToInt(float4 rgba) {

    rgba.x = clamp(rgba.x, 0.f, 1.f);   // clamp to [0.0, 1.0]
    rgba.y = clamp(rgba.y, 0.f, 1.f);
    rgba.z = clamp(rgba.z, 0.f, 1.f);
    rgba.w = clamp(rgba.w, 0.f, 1.f);
    return ((unsigned int)(rgba.w * 255.0f) << 24) |
        ((unsigned int)(rgba.z * 255.0f) << 16) |
        ((unsigned int)(rgba.y * 255.0f) << 8) |
        ((unsigned int)(rgba.x * 255.0f));
}

__global__ void ImageFromInference_kernel(precision_t* positions, precision_t* distances, precision_t distance_threshold, uint* out_img, uint2 resolution, int point_size) {

    //extern __shared__ uchar4 sdata[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    auto idx = Linearize(x, y, resolution.x, resolution.y);
    auto idx_pos = point_size * idx;
    float dist = distances[8 * idx];

    if (dist < distance_threshold &&
        abs(positions[idx_pos]) < precision_t(1.f) &&
        abs(positions[idx_pos + 1]) < precision_t(1.f) &&
        abs(positions[idx_pos + 2]) < precision_t(1.f)) {

        out_img[idx] = RgbaFloatToInt(make_float4(0.f, 0.f, 0.f, 1.f));
    }
    else {

        out_img[idx] = RgbaFloatToInt(make_float4(1.f, 1.f, 1.f, 1.f));
    }

}

__global__ void sum_residual_normals(precision_t* lod_0_x, precision_t* lod_0_y, precision_t* lod_0_z, precision_t* lod_1_x, precision_t* lod_1_y, precision_t* lod_1_z, precision_t* lod_2_x, precision_t* lod_2_y, precision_t* lod_2_z, precision_t* out_x, precision_t* out_y, precision_t* out_z, uint2 resolution) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    auto idx = Linearize(x, y, resolution.x, resolution.y);
    auto idx_normals = 8 * idx;

    precision_t nx = lod_0_x[idx_normals];
    precision_t ny = lod_0_y[idx_normals];
    precision_t nz = lod_0_z[idx_normals];

    //out_x[idx_normals] = lod_0_x[idx_normals];
    //out_y[idx_normals] = lod_0_y[idx_normals];
    //out_z[idx_normals] = lod_0_z[idx_normals];

    //if (lod_1_x != nullptr) {
    //    out_x[idx_normals] += lod_1_x[idx_normals];
    //    out_y[idx_normals] += lod_1_y[idx_normals];
    //    out_z[idx_normals] += lod_1_z[idx_normals];
    //}
    //
    //if (lod_2_x != nullptr) {
    //    out_x[idx_normals] += lod_2_x[idx_normals];
    //    out_y[idx_normals] += lod_2_y[idx_normals];
    //    out_z[idx_normals] += lod_2_z[idx_normals];
    //    // DEBUG
    //    /*precision_t epsilon = precision_t(1.e-6);
    //    if (abs(lod_2_x[idx_normals] - lod_1_x[idx_normals]) < epsilon && abs(lod_2_y[idx_normals] - lod_1_y[idx_normals]) < epsilon && abs(lod_2_z[idx_normals] - lod_1_z[idx_normals]) < epsilon) {
    //        out_x[idx_normals] = precision_t(1.0f);
    //        out_y[idx_normals] = precision_t(0.0f);
    //        out_z[idx_normals] = precision_t(0.0f);
    //    }
    //    else {
    //        out_x[idx_normals] = precision_t(0.0f);
    //        out_y[idx_normals] = precision_t(1.0f);
    //        out_z[idx_normals] = precision_t(0.0f);
    //    }*/
    //    //
    //}

    if (lod_1_x != nullptr) {
        nx += lod_1_x[idx_normals];
        ny += lod_1_y[idx_normals];
        nz += lod_1_z[idx_normals];
    }
    
    if (lod_2_x != nullptr) {
        nx += lod_2_x[idx_normals];
        ny += lod_2_y[idx_normals];
        nz += lod_2_z[idx_normals];
    }

    out_x[idx_normals] = nx;
    out_y[idx_normals] = ny;
    out_z[idx_normals] = nz;
}

__global__ void Shade_kernel(precision_t* positions, precision_t* directions, precision_t* distances_lod_0, precision_t* distances_lod_1, precision_t* distances_lod_2, bool use_lod_0, bool use_lod_1, bool use_lod_2, bool is_residual, precision_t distance_threshold, precision_t* nx,
    precision_t* ny, precision_t* nz, precision_t* tex_colors, uint* out_img, uint2 resolution, Shading shading, int point_size) {

    //extern __shared__ uchar4 sdata[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    auto idx = Linearize(x, y, resolution.x, resolution.y);
    auto idx_pos = point_size * idx;
    auto idx_dir = 3 * idx;
    auto idx_props = 8 * idx;
    precision_t dist = precision_t(0.0f);

    if (is_residual) {
        // DEBUG
        /*if(!use_lod_0) {
            out_img[idx] = RgbaFloatToInt(make_float4(1.f, 0.f, 0.f, 1.f));
            return;
        }
        if (use_lod_2) {
            if (use_lod_1) {
                out_img[idx] = RgbaFloatToInt(make_float4(0.f, 1.f, 0.f, 1.f));
            }
            else {
                out_img[idx] = RgbaFloatToInt(make_float4(1.f, 1.f, 0.f, 1.f));
            }
            return;
        }
        if (use_lod_1) {
            out_img[idx] = RgbaFloatToInt(make_float4(0.f, 0.f, 1.f, 1.f));
            return;
        }*/
        //
        
        if (use_lod_0) dist += distances_lod_0[idx_props];
        if (use_lod_1) dist += distances_lod_1[idx_props];
        if (use_lod_2) dist += distances_lod_2[idx_props];
    }
    else
    {
        // DEBUG
        /*if (use_lod_0) {
            if(use_lod_1 || use_lod_2)
                out_img[idx] = RgbaFloatToInt(make_float4(1.f, 0.f, 0.f, 1.f));
            else
                out_img[idx] = RgbaFloatToInt(make_float4(0.f, 0.f, 1.f, 1.f));
            return;
        }
        if (use_lod_1) {
            if (use_lod_0 || use_lod_2)
                out_img[idx] = RgbaFloatToInt(make_float4(0.f, 1.f, 0.f, 1.f));
            else
                out_img[idx] = RgbaFloatToInt(make_float4(1.f, 1.f, 0.f, 1.f));
            return;
        }
        if (use_lod_2) {
            if (use_lod_0 || use_lod_1)
                out_img[idx] = RgbaFloatToInt(make_float4(0.f, 0.f, 1.f, 1.f));
            else
                out_img[idx] = RgbaFloatToInt(make_float4(1.f, 1.f, 1.f, 1.f));
            return;
        }*/
        //

        if (use_lod_0) dist = distances_lod_0[idx_props];
        if (use_lod_1) dist = distances_lod_1[idx_props];
        if (use_lod_2) dist = distances_lod_2[idx_props];
    }

    if (abs(dist) < distance_threshold && 
        abs(positions[idx_pos]) < precision_t(1.f) &&
        abs(positions[idx_pos + 1]) < precision_t(1.f) &&
        abs(positions[idx_pos + 2]) < precision_t(1.f)) {

        precision_t dir_x = -directions[idx_dir];
        precision_t dir_y = -directions[idx_dir + 1];
        precision_t dir_z = -directions[idx_dir + 2];

        precision_t n_x = nx[idx_props];
        precision_t n_y = ny[idx_props];
        precision_t n_z = nz[idx_props];

        precision_t len = sqrt(n_x * n_x + n_y * n_y + n_z * n_z);
        n_x = n_x / len;
        n_y = n_y / len;
        n_z = n_z / len;

        if (shading == NORMALS) {

            out_img[idx] = RgbaFloatToInt(make_float4(abs(n_x), abs(n_z), abs(n_y), 1.f));
        }
        else {

            precision_t k_s(0.5f);
            precision_t k_d(1.f);
            precision_t k_a(1.f);
            precision_t shininess(35.f);

            precision_t ambient_r(0.2f);
            precision_t ambient_g(0.2f);
            precision_t ambient_b(0.2);
            precision_t specular_r(1.f);
            precision_t specular_g(1.f);
            precision_t specular_b(1.f);
            precision_t diffuse_r(0.54f);
            precision_t diffuse_g(0.54f);
            precision_t diffuse_b(0.54f);
            if (tex_colors != nullptr)
            {
                diffuse_r = tex_colors[idx_props] /*/ 255.f*/;
                diffuse_g = tex_colors[idx_props + 1] /*/ 255.f*/;
                diffuse_b = tex_colors[idx_props + 2] /*/ 255.f*/;
            }

            //precision_t light_dir_x(0.f);
            //precision_t light_dir_y(sin(0.1f));
            //precision_t light_dir_z(-cos(0.1f));

            precision_t light_dir_x = dir_x;
            precision_t light_dir_y = dir_y;
            precision_t light_dir_z = dir_z;

            precision_t color_r = k_a * ambient_r;
            precision_t color_g = k_a * ambient_g;
            precision_t color_b = k_a * ambient_b;

            for (int i = 0; i < 1; ++i) {
                precision_t dot_d = light_dir_x * n_x + light_dir_y * n_y + light_dir_z * n_z;
                    
                precision_t reflection_x = precision_t(2.f) * dot_d * n_x - light_dir_x;
                precision_t reflection_y = precision_t(2.f) * dot_d * n_y - light_dir_y;
                precision_t reflection_z = precision_t(2.f) * dot_d * n_z - light_dir_z;

                dot_d = precision_t(max(dot_d, 0.f));

                precision_t dot_s = precision_t(powf(dir_x * reflection_x + dir_y * reflection_y + dir_z * reflection_z, shininess));
                dot_s = precision_t(max(dot_s, 0.f));

                color_r += k_d * dot_d * diffuse_r + k_s * dot_s * specular_r;
                color_g += k_d * dot_d * diffuse_g + k_s * dot_s * specular_g;
                color_b += k_d * dot_d * diffuse_b + k_s * dot_s * specular_b;

                light_dir_z = precision_t(cos(0.1f));
            }
                
            out_img[idx] = RgbaFloatToInt(make_float4(color_r, color_g, color_b, 1.f));
        }
    }
    else {

        //out_img[idx] = RgbaFloatToInt(make_float4(0.f, 0.f, 0.f, 1.f));
        out_img[idx] = RgbaFloatToInt(make_float4(1.f, 1.f, 1.f, 1.f));
    }
}

template<int W0>
__global__ void CalculateG0(precision_t* Weights0, precision_t* a0, precision_t* G0, precision_t* p0, int coord, uint2 G0_size, int point_size) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < G0_size.x && j < G0_size.y) {

        int idx = Linearize(i, j, G0_size.x, G0_size.y);
        int idx_Weights0 = Linearize(i, coord, G0_size.x, point_size);

        precision_t a0_element = a0[idx];
        G0[idx] = W0 * Weights0[idx_Weights0] * cos(W0 * a0_element);
        p0[idx] = sin(W0 * a0_element);
    }
}

template<int W0>
__global__ void HadamardGi(precision_t* Gi, precision_t* ai, precision_t* pi, uint2 Gi_size) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < Gi_size.x && j < Gi_size.y) {

        int idx = Linearize(i, j, Gi_size.x, Gi_size.y);

        if (pi != nullptr) {
            precision_t ai_element = ai[idx];
            Gi[idx] = W0 * Gi[idx] * cos(W0 * ai_element);
            pi[idx] = sin(W0 * ai_element);
        }
        else {
            Gi[idx] = W0 * Gi[idx];
        }
    }
}

__global__ void IterateSphereTracing_kernel(precision_t* origins, precision_t* directions, precision_t* distances_lod_0, float delta_lod_0, precision_t* distances_lod_1, float delta_lod_1, precision_t* distances_lod_2, uint2 resolution, bool use_lod_0, bool use_lod_1, bool use_lod_2, bool is_residual, int point_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < resolution.x && j < resolution.y) {

        int idx_dist = Linearize(i, j, resolution.x, resolution.y);
        int idx_origins = point_size * idx_dist;
        int idx_directions = 3 * idx_dist;
        idx_dist = 8 * idx_dist; // 8 because the Tensor Core output size is 8.

        if (!is_residual) {
            if (use_lod_0 && !use_lod_1 && !use_lod_2)
            {
                origins[idx_origins] = origins[idx_origins] + directions[idx_directions] * (distances_lod_0[idx_dist] - delta_lod_0);
                origins[idx_origins + 1] = origins[idx_origins + 1] + directions[idx_directions + 1] * (distances_lod_0[idx_dist] - delta_lod_0);
                origins[idx_origins + 2] = origins[idx_origins + 2] + directions[idx_directions + 2] * (distances_lod_0[idx_dist] - delta_lod_0);
            }
            else {
                if (distances_lod_0[idx_dist] < delta_lod_0 * 1.4)
                {
                    if (use_lod_2)
                    {
                        origins[idx_origins] = origins[idx_origins] + directions[idx_directions] * (distances_lod_2[idx_dist]);
                        origins[idx_origins + 1] = origins[idx_origins + 1] + directions[idx_directions + 1] * (distances_lod_2[idx_dist]);
                        origins[idx_origins + 2] = origins[idx_origins + 2] + directions[idx_directions + 2] * (distances_lod_2[idx_dist]);
                    }
                    else
                    {
                        origins[idx_origins] = origins[idx_origins] + directions[idx_directions] * (distances_lod_1[idx_dist]);
                        origins[idx_origins + 1] = origins[idx_origins + 1] + directions[idx_directions + 1] * (distances_lod_1[idx_dist]);
                        origins[idx_origins + 2] = origins[idx_origins + 2] + directions[idx_directions + 2] * (distances_lod_1[idx_dist]);
                    }
                }
                else
                {
                    origins[idx_origins] = origins[idx_origins] + directions[idx_directions] * (distances_lod_0[idx_dist]);
                    origins[idx_origins + 1] = origins[idx_origins + 1] + directions[idx_directions + 1] * (distances_lod_0[idx_dist]);
                    origins[idx_origins + 2] = origins[idx_origins + 2] + directions[idx_directions + 2] * (distances_lod_0[idx_dist]);
                }
            }
        }
        else {

            precision_t dist_base = distances_lod_0[idx_dist];
            precision_t residual_1 = use_lod_1 ? distances_lod_1[idx_dist] : precision_t(0.0f);
            precision_t residual_2 = use_lod_2 ? distances_lod_2[idx_dist] : precision_t(0.0f);
            
            precision_t dist_res_1 = dist_base + residual_1;
            precision_t dist_res_2 = dist_base + residual_1 + residual_2;

            if (use_lod_0 && !use_lod_1 && !use_lod_2)
            {
                origins[idx_origins] = origins[idx_origins] + directions[idx_directions] * (dist_base - delta_lod_0);
                origins[idx_origins + 1] = origins[idx_origins + 1] + directions[idx_directions + 1] * (dist_base - delta_lod_0);
                origins[idx_origins + 2] = origins[idx_origins + 2] + directions[idx_directions + 2] * (dist_base - delta_lod_0);
            }
            else {
                if (distances_lod_0[idx_dist] < delta_lod_0 * 1.4)
                {
                    if (use_lod_2)
                    {
                        origins[idx_origins] = origins[idx_origins] + directions[idx_directions] * (dist_res_2);
                        origins[idx_origins + 1] = origins[idx_origins + 1] + directions[idx_directions + 1] * (dist_res_2);
                        origins[idx_origins + 2] = origins[idx_origins + 2] + directions[idx_directions + 2] * (dist_res_2);
                    }
                    else
                    {
                        origins[idx_origins] = origins[idx_origins] + directions[idx_directions] * (dist_res_1);
                        origins[idx_origins + 1] = origins[idx_origins + 1] + directions[idx_directions + 1] * (dist_res_1);
                        origins[idx_origins + 2] = origins[idx_origins + 2] + directions[idx_directions + 2] * (dist_res_1);
                    }
                }
                else
                {
                    origins[idx_origins] = origins[idx_origins] + directions[idx_directions] * (dist_base);
                    origins[idx_origins + 1] = origins[idx_origins + 1] + directions[idx_directions + 1] * (dist_base);
                    origins[idx_origins + 2] = origins[idx_origins + 2] + directions[idx_directions + 2] * (dist_base);
                }
            }
        }
    }
}

__global__ void InitializeRays_kernel(precision_t time, precision_t cam_time, precision_t* origins_time, precision_t* directions, uint2 resolution, int point_size, bool swap_y_and_z, bool invert_z, float* inv_view_matrix, float* inv_proj_matrix) {
    precision_t i(threadIdx.x + blockIdx.x * blockDim.x);
    precision_t j(threadIdx.y + blockIdx.y * blockDim.y);

    if (i < precision_t(resolution.x) && j < precision_t(resolution.y)) {
        int idx_origins = Linearize(i, j, resolution.x, resolution.y);
        int idx_directions = 3 * idx_origins;
        idx_origins *= point_size;

        // Compute normalized device coordinates (NDC)
        precision_t ndc_x = (precision_t(2.0f) * i) / precision_t(resolution.x) - precision_t(1.0f);
        precision_t ndc_y = (precision_t(2.0f) * j) / precision_t(resolution.y) - precision_t(1.0f);
        precision_t ndc_z = precision_t(-1.0f); // Assuming a near plane at z = -1.0

        // Apply the inverse projection matrix
        precision_t ray_clip[4] = { ndc_x, ndc_y, ndc_z, precision_t(1.0f) };
        precision_t ray_eye[4] = { precision_t(0.0f), precision_t(0.0f), precision_t(0.0f), precision_t(0.0f) };

        for (int k = 0; k < 4; ++k) {
            ray_eye[k] = inv_proj_matrix[0 * 4 + k] * ray_clip[0] +
                inv_proj_matrix[1 * 4 + k] * ray_clip[1] +
                inv_proj_matrix[2 * 4 + k] * ray_clip[2] +
                inv_proj_matrix[3 * 4 + k] * ray_clip[3];
        }

        // Apply the inverse view matrix
        precision_t ray_world[3] = { precision_t(0.0f), precision_t(0.0f), precision_t(0.0f) };

        for (int k = 0; k < 3; ++k) {
            ray_world[k] = inv_view_matrix[0 * 4 + k] * ray_eye[0] +
                inv_view_matrix[1 * 4 + k] * ray_eye[1] +
                inv_view_matrix[2 * 4 + k] * ray_eye[2] +
                inv_view_matrix[3 * 4 + k] * precision_t(1.0f);
        }

        // Set the ray origin
        precision_t ray_origin[3];
        ray_origin[0] = inv_view_matrix[12];
        ray_origin[1] = inv_view_matrix[13];
        ray_origin[2] = inv_view_matrix[14];

        // Compute the ray direction
        precision_t ray_direction[3];
        ray_direction[0] = ray_world[0] - ray_origin[0];
        ray_direction[1] = ray_world[1] - ray_origin[1];
        ray_direction[2] = ray_world[2] - ray_origin[2];

        // Normalize the ray direction
        precision_t ray_dir_length = sqrt(ray_direction[0] * ray_direction[0] + ray_direction[1] * ray_direction[1] + ray_direction[2] * ray_direction[2]);
        directions[idx_directions] = ray_direction[0] / ray_dir_length;
        directions[idx_directions + 1] = ray_direction[1] / ray_dir_length;
        directions[idx_directions + 2] = ray_direction[2] / ray_dir_length;

        // Set the ray origin in the output array
        origins_time[idx_origins] = ray_origin[0];
        origins_time[idx_origins + 1] = ray_origin[1];
        origins_time[idx_origins + 2] = ray_origin[2];
        if (point_size == 4) {
            origins_time[idx_origins + 3] = time;
        }
    }
}

__global__ void TestRays_kernel(precision_t* origins_time, uint* outImg, uint2 resolution)
{
    // Render an white box where the Siren model is. Red otherwise.
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < resolution.x && j < resolution.y) {
        int idx = Linearize(i, j, resolution.x, resolution.y);
        if (origins_time[idx] <= precision_t(1.f) && origins_time[idx] >= precision_t(-1.f) &&
            origins_time[idx + 1] <= precision_t(1.f) && origins_time[idx + 1] >= precision_t(-1.f) &&
            origins_time[idx + 2] < precision_t(-1.f))
        {
            outImg[j * resolution.y + i] = RgbaFloatToInt(make_float4(1.f, 1.f, 1.f, 1.f));
        }
        else
        {
            outImg[j * resolution.y + i] = RgbaFloatToInt(make_float4(1.f, 0.f, 0.f, 1.f));
        }
    }
}