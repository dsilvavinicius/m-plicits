#include <algorithm>
#include <iostream>
#include <chrono>
#include <string>

#include <helper_cuda.h>
#include "helper_image.h"
#include "helper.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"

using uint = unsigned int;
using namespace std;

using precision_t = typename cutlass::half_t;
//constexpr float W0 = 30.f;
//constexpr float W0 = 20.f;
//constexpr float W0 = 16.f;

#include "camera.cu"
#include "tools.cu"
#include "layer_gemms.cu"
#include "gemms.cu"
#include "kernels.cu"
#include "state.cu"