// ===============
// Sine epilogue.
// ===============

template <typename T, int W0>
struct sine {
    CUTLASS_HOST_DEVICE
        T operator()(T lhs) const {
        return (T)sinf((float)W0 * lhs);
    }
};

template <typename T, int N, int W0>
struct sine<cutlass::Array<T, N>, W0> {
    CUTLASS_HOST_DEVICE
        cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& lhs) const {
        cutlass::Array<T, N> result;
        sine<T, W0> scalar_op;

        CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < N; ++i) {
                result[i] = scalar_op(lhs[i]);
            }

        return result;
    }
};

template <
    typename ElementOutput_,                             ///< Data type used to load and store tensors
    int Count,                                           ///< Number of elements computed per operation
    int W0,
    typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
    typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
    cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest
>
class SineEpilogue {
public:

    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;

    static int const kCount = Count;

    using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
    using ComputeFragment = cutlass::Array<ElementCompute, kCount>;

    static cutlass::FloatRoundStyle const kRound = Round;

    /// Host-constructable parameters structure
    struct Params {

        ElementCompute alpha;                  ///< scales accumulators

        CUTLASS_HOST_DEVICE
            Params() :
            alpha(ElementCompute(1)) {}

        CUTLASS_HOST_DEVICE
            Params(
                ElementCompute alpha
            ) : alpha(alpha) {}
    };

private:
    ElementCompute alpha_;

public:

    /// Constructs the function object, possibly loading from pointers in host memory
    CUTLASS_HOST_DEVICE
        SineEpilogue(Params const& params) {
        alpha_ = params.alpha;
    }

    /// Returns true if source is needed
    CUTLASS_HOST_DEVICE
        bool is_source_needed() const {
        return true;
    }

    /// Functionally required for serial reduction in the epilogue
    CUTLASS_HOST_DEVICE
        void set_k_partition(int k_partition, int k_partition_count) {}

    CUTLASS_HOST_DEVICE
        FragmentOutput operator()(
            FragmentAccumulator const& accumulator,
            FragmentOutput const& source) const {

        // Convert source to interal compute numeric type
        cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

        ComputeFragment converted_source = source_converter(source);
        ComputeFragment converted_accumulator = accumulator_converter(accumulator);

        // Perform binary operations
        ComputeFragment intermediate;

        //cutlass::multiplies<ComputeFragment> mul_add_source;
        cutlass::multiply_add<ComputeFragment> mul_add_accumulator;

        //if (Scale == ScaleType::NoBetaScaling)
        intermediate = converted_source;
        //else
        //    intermediate = mul_add_source(beta_, converted_source);                             // X =  beta * C + uniform

        intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X

        // Compute threshold optionally
        sine<ComputeFragment, W0> sine;
        intermediate = (ComputeFragment)sine(intermediate);//relu(threshold_, intermediate);

        // Convert to destination numeric type
        cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

        return destination_converter(intermediate);
    }

    CUTLASS_HOST_DEVICE
        FragmentOutput operator()(
            FragmentAccumulator const& accumulator) const {

        // Convert source to interal compute numeric type
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

        ComputeFragment converted_accumulator = accumulator_converter(accumulator);

        // Perform binary operations
        ComputeFragment intermediate;

        cutlass::multiplies<ComputeFragment> mul_accumulator;

        intermediate = mul_accumulator(alpha_, converted_accumulator);    // D = alpha * Accum

        // Compute threshold optionally
        sine<ComputeFragment, W0> sine;
        intermediate = (ComputeFragment)sine(intermediate);//relu(threshold_, intermediate);

        // Convert to destination numeric type
        cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

        return destination_converter(intermediate);
    }
};

// =================
// Layer definitions
// =================
template<int W0>
struct RTX2070Gemms_512x512 {

    static constexpr int point_size = 4;

    static const int resolution_x = 512;
    static const int resolution_y = 512;

    static const int block_x = 16;
    static const int block_y = 16;

    static const int W0_ = W0;

    // =================
    // Output layer.
    // =================

    // Gemm operator cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8
    //using cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = cutlass::half_t;         // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and output matrices.
    using LayoutInputA = cutlass::layout::ColumnMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int OutputAlignmentA = 8;
    const static int OutputAlignmentB = 8;
    const static int OutputEpilogueComputedPerOp = 8;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using MMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using SmArch = cutlass::arch::Sm75;

    // This code section describes the tile size a thread block will compute
    using OutputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using OutputShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32 
    // This code section describes the size of MMA op
    using OutputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

    // Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
    //
    //    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
    //
    using OutputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        OutputEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    // Number of pipelines you want to use
    const static int NumStages = 2;

    using OutputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        MMAOp,
        SmArch,
        OutputShapeMMAThreadBlock,
        OutputShapeMMAWarp,
        OutputShapeMMAOp,
        OutputEpilogueOp,
        SwizzleThreadBlock,
        NumStages,
        OutputAlignmentA,
        OutputAlignmentB>;

    // ==============
    // Hidden Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8
    //using cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    const static int HiddenAlignmentA = 8;
    const static int HiddenAlignmentB = 8;
    const static int HiddenEpilogueComputedPerOp = 8;

    using HiddenEpilogueOp = SineEpilogue<
        ElementOutput,                                        // <- data type of output matrix
        HiddenEpilogueComputedPerOp,
        W0,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue
    >;

    // This code section describes the tile size a thread block will compute
    using HiddenShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 64, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using HiddenShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 32, N = 32, K = 32 
    // This code section describes the size of MMA op
    using HiddenShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    using HiddenGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        MMAOp,
        SmArch,
        HiddenShapeMMAThreadBlock,
        HiddenShapeMMAWarp,
        HiddenShapeMMAOp,
        HiddenEpilogueOp,
        SwizzleThreadBlock,
        NumStages,
        HiddenAlignmentA,
        HiddenAlignmentB>;

    // ==============
    // Input Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h884gemm_64x64_32x2_tt_align2
    //using cutlass_tensorop_h884gemm_64x64_32x2_tt_align2_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 2,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 2,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm70,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<32, 32, 32>,
    //    cutlass::gemm::GemmShape<8, 8, 4>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    2,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    using InputElementAccumulator = cutlass::half_t;
    using InputElementComputeEpilogue = InputElementAccumulator;
    using OutputArch = cutlass::arch::Sm70;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int InputAlignmentA = 2;
    const static int InputAlignmentB = 2;
    const static int InputEpilogueComputedPerOp = 2;

    // This code section describes the tile size a thread block will compute
    using InputShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using InputShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32 
    // This code section describes the size of MMA op
    using InputShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 16, N = 8, K = 8

    using InputEpilogueOp = SineEpilogue<
        ElementOutput,                                        // <- data type of output matrix
        InputEpilogueComputedPerOp,
        W0,
        InputElementAccumulator,                                   // <- data type of accumulator
        InputElementComputeEpilogue
    >;

    using InputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        InputElementAccumulator,
        MMAOp,
        OutputArch,
        InputShapeMMAThreadBlock,
        InputShapeMMAWarp,
        InputShapeMMAOp,
        InputEpilogueOp,
        SwizzleThreadBlock,
        NumStages,
        InputAlignmentA,
        InputAlignmentB>;
};

struct RTX2070NormalGemms_512x512 {

    static constexpr int point_size = 4;

    static const int resolution_x = 512;
    static const int resolution_y = 512;

    static const int block_x = 16;
    static const int block_y = 16;

    // =================
    // Output layer.
    // =================

    // Gemm operator cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8
    //using cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = cutlass::half_t;         // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and output matrices.
    using LayoutInputA = cutlass::layout::ColumnMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int OutputAlignmentA = 8;
    const static int OutputAlignmentB = 8;
    const static int OutputEpilogueComputedPerOp = 8;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using MMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using SmArch = cutlass::arch::Sm75;

    // This code section describes the tile size a thread block will compute
    using OutputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using OutputShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32 
    // This code section describes the size of MMA op
    using OutputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

    // Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
    //
    //    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
    //
    using OutputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        OutputEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    // Number of pipelines you want to use
    const static int NumStages = 2;

    using OutputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        MMAOp,
        SmArch,
        OutputShapeMMAThreadBlock,
        OutputShapeMMAWarp,
        OutputShapeMMAOp,
        OutputEpilogueOp,
        SwizzleThreadBlock,
        NumStages,
        OutputAlignmentA,
        OutputAlignmentB>;

    // ==============
    // Hidden Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8
    //using cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    const static int HiddenAlignmentA = 8;
    const static int HiddenAlignmentB = 8;
    const static int HiddenEpilogueComputedPerOp = 8;

    using HiddenEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        HiddenEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    // This code section describes the tile size a thread block will compute
    using HiddenShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 64, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using HiddenShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 32, N = 32, K = 32 
    // This code section describes the size of MMA op
    using HiddenShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    using HiddenGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        MMAOp,
        SmArch,
        HiddenShapeMMAThreadBlock,
        HiddenShapeMMAWarp,
        HiddenShapeMMAOp,
        HiddenEpilogueOp,
        SwizzleThreadBlock,
        NumStages,
        HiddenAlignmentA,
        HiddenAlignmentB>;

    // ==============
    // Input Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h884gemm_64x64_32x2_tt_align2
    //using cutlass_tensorop_h884gemm_64x64_32x2_tt_align2_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 2,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 2,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm70,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<32, 32, 32>,
    //    cutlass::gemm::GemmShape<8, 8, 4>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    2,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    using InputElementAccumulator = cutlass::half_t;
    using InputElementComputeEpilogue = InputElementAccumulator;
    using OutputArch = cutlass::arch::Sm70;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int InputAlignmentA = 2;
    const static int InputAlignmentB = 2;
    const static int InputEpilogueComputedPerOp = 2;

    // This code section describes the tile size a thread block will compute
    using InputShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using InputShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32 
    // This code section describes the size of MMA op
    using InputShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 16, N = 8, K = 8

    using InputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                          // <- data type of output matrix
        InputEpilogueComputedPerOp,
        InputElementAccumulator,                                // <- data type of accumulator
        InputElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using InputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        InputElementAccumulator,
        MMAOp,
        OutputArch,
        InputShapeMMAThreadBlock,
        InputShapeMMAWarp,
        InputShapeMMAOp,
        InputEpilogueOp,
        SwizzleThreadBlock,
        NumStages,
        InputAlignmentA,
        InputAlignmentB>;
};

template<int W0>
struct RTX3090Gemms_512x512 {

    static constexpr int point_size = 4;

    static const int resolution_x = 512;
    static const int resolution_y = 512;

    static const int block_x = 16;
    static const int block_y = 16;

    static const int W0_ = W0;

    // =================
    // Output layer.
    // =================

    // Gemm operator cutlass_simt_hgemm_128x32_8x2_tt_align1
    //using cutlass_tensorop_h16816gemm_128x128_32x3_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 128, 32>,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    3,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = cutlass::half_t;         // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and output matrices.
    using LayoutInputA = cutlass::layout::ColumnMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int OutputAlignmentA = 8;
    const static int OutputAlignmentB = 8;
    const static int OutputEpilogueComputedPerOp = 8;
    const static int OutputNumStages = 3;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using OutputMMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using OutputSmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using OutputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
    // This code section describes tile size a warp will compute
    using OutputShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
    // This code section describes the size of MMA op
    using OutputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

    // Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
    //
    //    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
    //
    using OutputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        OutputEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using OutputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        OutputMMAOp,
        OutputSmArch,
        OutputShapeMMAThreadBlock,
        OutputShapeMMAWarp,
        OutputShapeMMAOp,
        OutputEpilogueOp,
        SwizzleThreadBlock,
        OutputNumStages,
        OutputAlignmentA,
        OutputAlignmentB>;

    // ==============
    // Hidden Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h16816gemm_128x64_32x6_tt_align8_
    //using cutlass_tensorop_h16816gemm_128x64_32x6_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    6,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    const static int HiddenAlignmentA = 8;
    const static int HiddenAlignmentB = 8;
    const static int HiddenEpilogueComputedPerOp = 8;
    const static int HiddenNumStages = 6;

    using HiddenEpilogueOp = SineEpilogue<
        ElementOutput,                                        // <- data type of output matrix
        HiddenEpilogueComputedPerOp,
        W0,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue
    >;

    using HiddenMMAOp = cutlass::arch::OpClassTensorOp;
    using HiddenSmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using HiddenShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using HiddenShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32 
    // This code section describes the size of MMA op
    using HiddenShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    using HiddenGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        HiddenMMAOp,
        HiddenSmArch,
        HiddenShapeMMAThreadBlock,
        HiddenShapeMMAWarp,
        HiddenShapeMMAOp,
        HiddenEpilogueOp,
        SwizzleThreadBlock,
        HiddenNumStages,
        HiddenAlignmentA,
        HiddenAlignmentB>;

    // ==============
    // Input Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h16816gemm_256x64_32x4_tt_align4
    //using cutlass_tensorop_h16816gemm_256x64_32x4_tt_align4_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<256, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    4,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    4,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    using InputElementAccumulator = cutlass::half_t;
    using InputElementComputeEpilogue = InputElementAccumulator;
    using InputMMAOp = cutlass::arch::OpClassTensorOp;
    using InputSmArch = cutlass::arch::Sm80;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int InputAlignmentA = 4;
    const static int InputAlignmentB = 4;
    const static int InputEpilogueComputedPerOp = 4;
    const static int InputNumStages = 4;

    // This code section describes the tile size a thread block will compute
    using InputShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 64, 32>;  // <- threadblock tile M = 256, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using InputShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
    // This code section describes the size of MMA op
    using InputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    using InputEpilogueOp = SineEpilogue<
        ElementOutput,                                        // <- data type of output matrix
        InputEpilogueComputedPerOp,
        W0,
        InputElementAccumulator,                                   // <- data type of accumulator
        InputElementComputeEpilogue
    >;

    using InputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        InputElementAccumulator,
        InputMMAOp,
        InputSmArch,
        InputShapeMMAThreadBlock,
        InputShapeMMAWarp,
        InputShapeMMAOp,
        InputEpilogueOp,
        SwizzleThreadBlock,
        InputNumStages,
        InputAlignmentA,
        InputAlignmentB>;
};

template<int W0>
struct RTX3090Gemms_256_hidden_512x512 {

    static constexpr int point_size = 4;

    static const int resolution_x = 512;
    static const int resolution_y = 512;

    static const int block_x = 16;
    static const int block_y = 16;

    static const int W0_ = W0;

    // =================
    // Output layer.
    // =================

    // Gemm operator cutlass_tensorop_h16816gemm_128x64_64x3_tt_align8
    //using cutlass_tensorop_h16816gemm_128x64_64x3_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 64, 64>,
    //    cutlass::gemm::GemmShape<64, 32, 64>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    3,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = cutlass::half_t;         // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and output matrices.
    using LayoutInputA = cutlass::layout::ColumnMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int OutputAlignmentA = 8;
    const static int OutputAlignmentB = 8;
    const static int OutputEpilogueComputedPerOp = 8;
    const static int OutputNumStages = 3;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using OutputMMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using OutputSmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using OutputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 64>;  // <- threadblock tile M = 128, N = 64, K = 64
    // This code section describes tile size a warp will compute
    using OutputShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 64>;  // <- warp tile M = 64, N = 32, K = 64 
    // This code section describes the size of MMA op
    using OutputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

    // Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
    //
    //    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
    //
    using OutputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        OutputEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using OutputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        OutputMMAOp,
        OutputSmArch,
        OutputShapeMMAThreadBlock,
        OutputShapeMMAWarp,
        OutputShapeMMAOp,
        OutputEpilogueOp,
        SwizzleThreadBlock,
        OutputNumStages,
        OutputAlignmentA,
        OutputAlignmentB>;

    // ==============
    // Hidden Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h16816gemm_128x128_32x3_tt_align8
    //using cutlass_tensorop_h16816gemm_128x128_32x3_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 128, 32>,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    3,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    const static int HiddenAlignmentA = 8;
    const static int HiddenAlignmentB = 8;
    const static int HiddenEpilogueComputedPerOp = 8;
    const static int HiddenNumStages = 3;

    using HiddenEpilogueOp = SineEpilogue<
        ElementOutput,                                        // <- data type of output matrix
        HiddenEpilogueComputedPerOp,
        W0,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue
    >;

    using HiddenMMAOp = cutlass::arch::OpClassTensorOp;
    using HiddenSmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using HiddenShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
    // This code section describes tile size a warp will compute
    using HiddenShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
    // This code section describes the size of MMA op
    using HiddenShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    using HiddenGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        HiddenMMAOp,
        HiddenSmArch,
        HiddenShapeMMAThreadBlock,
        HiddenShapeMMAWarp,
        HiddenShapeMMAOp,
        HiddenEpilogueOp,
        SwizzleThreadBlock,
        HiddenNumStages,
        HiddenAlignmentA,
        HiddenAlignmentB>;

    // ==============
    // Input Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h16816gemm_128x128_32x5_tt_align4
    //using cutlass_tensorop_h16816gemm_128x128_32x5_tt_align4_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 128, 32>,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    4,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    5,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    using InputElementAccumulator = cutlass::half_t;
    using InputElementComputeEpilogue = InputElementAccumulator;
    using InputMMAOp = cutlass::arch::OpClassTensorOp;
    using InputSmArch = cutlass::arch::Sm80;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int InputAlignmentA = 4;
    const static int InputAlignmentB = 4;
    const static int InputEpilogueComputedPerOp = 4;
    const static int InputNumStages = 5;

    // This code section describes the tile size a thread block will compute
    using InputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
    // This code section describes tile size a warp will compute
    using InputShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
    // This code section describes the size of MMA op
    using InputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    using InputEpilogueOp = SineEpilogue<
        ElementOutput,                                        // <- data type of output matrix
        InputEpilogueComputedPerOp,
        W0,
        InputElementAccumulator,                                   // <- data type of accumulator
        InputElementComputeEpilogue
    >;

    using InputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        InputElementAccumulator,
        InputMMAOp,
        InputSmArch,
        InputShapeMMAThreadBlock,
        InputShapeMMAWarp,
        InputShapeMMAOp,
        InputEpilogueOp,
        SwizzleThreadBlock,
        InputNumStages,
        InputAlignmentA,
        InputAlignmentB>;
};

template<int W0>
struct RTX3090Gemms_1024x1024 {

    static constexpr int point_size = 4;

    static const int resolution_x = 1024;
    static const int resolution_y = 1024;

    static const int block_x = 8;
    static const int block_y = 32;

    static const int W0_ = W0;

    // =================
    // Output layer.
    // =================

    // Profiled operator: cutlass_tensorop_h1688gemm_64x64_32x2_tt_align8_base
    //using cutlass_tensorop_h1688gemm_64x64_32x2_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<32, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = cutlass::half_t;         // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and output matrices.
    using LayoutInputA = cutlass::layout::ColumnMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int OutputAlignmentA = 8;
    const static int OutputAlignmentB = 8;
    const static int OutputEpilogueComputedPerOp = 8;
    const static int OutputNumStages = 2;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using OutputMMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using OutputSmArch = cutlass::arch::Sm75;

    // This code section describes the tile size a thread block will compute
    using OutputShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 32>;  // <- threadblock tile M = 64, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using OutputShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;  // <- warp tile M = 32, N = 32, K = 32 
    // This code section describes the size of MMA op
    using OutputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

    // Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
    //
    //    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
    //
    using OutputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        OutputEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using OutputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        OutputMMAOp,
        OutputSmArch,
        OutputShapeMMAThreadBlock,
        OutputShapeMMAWarp,
        OutputShapeMMAOp,
        OutputEpilogueOp,
        SwizzleThreadBlock,
        OutputNumStages,
        OutputAlignmentA,
        OutputAlignmentB>;

    // ==============
    // Hidden Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h16816gemm_128x64_32x6_tt_align4
    //using cutlass_tensorop_h16816gemm_128x64_32x6_tt_align4_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    4,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    6,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    const static int HiddenAlignmentA = 4;
    const static int HiddenAlignmentB = 4;
    const static int HiddenEpilogueComputedPerOp = 4;
    const static int HiddenNumStages = 6;

    using HiddenEpilogueOp = SineEpilogue<
        ElementOutput,                                        // <- data type of output matrix
        HiddenEpilogueComputedPerOp,
        W0,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue
    >;

    using HiddenMMAOp = cutlass::arch::OpClassTensorOp;
    using HiddenSmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using HiddenShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using HiddenShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32 
    // This code section describes the size of MMA op
    using HiddenShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    using HiddenGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        HiddenMMAOp,
        HiddenSmArch,
        HiddenShapeMMAThreadBlock,
        HiddenShapeMMAWarp,
        HiddenShapeMMAOp,
        HiddenEpilogueOp,
        SwizzleThreadBlock,
        HiddenNumStages,
        HiddenAlignmentA,
        HiddenAlignmentB>;

    // ==============
    // Input Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h1688gemm_128x64_32x2_tt_align4
    //using cutlass_tensorop_h1688gemm_128x64_32x2_tt_align4_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    4,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    using InputElementAccumulator = cutlass::half_t;
    using InputElementComputeEpilogue = InputElementAccumulator;
    using InputMMAOp = cutlass::arch::OpClassTensorOp;
    using InputSmArch = cutlass::arch::Sm75;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int InputAlignmentA = 4;
    const static int InputAlignmentB = 4;
    const static int InputEpilogueComputedPerOp = 4;
    const static int InputNumStages = 2;

    // This code section describes the tile size a thread block will compute
    using InputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using InputShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32 
    // This code section describes the size of MMA op
    using InputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    using InputEpilogueOp = SineEpilogue<
        ElementOutput,                                        // <- data type of output matrix
        InputEpilogueComputedPerOp,
        W0,
        InputElementAccumulator,                                   // <- data type of accumulator
        InputElementComputeEpilogue
    >;

    using InputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        InputElementAccumulator,
        InputMMAOp,
        InputSmArch,
        InputShapeMMAThreadBlock,
        InputShapeMMAWarp,
        InputShapeMMAOp,
        InputEpilogueOp,
        SwizzleThreadBlock,
        InputNumStages,
        InputAlignmentA,
        InputAlignmentB>;
};

template<int W0>
struct RTX3090Gemms_1920x1080 {

    static constexpr int point_size = 4;

    static const int resolution_x = 1920;
    static const int resolution_y = 1080;

    static const int block_x = 8;
    static const int block_y = 32;

    static const int W0_ = W0;

    // =================
    // Output layer.
    // =================

    // Profiled operator: cutlass_tensorop_h1688gemm_64x64_32x2_tt_align8_base
    //using cutlass_tensorop_h1688gemm_64x64_32x2_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<32, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = cutlass::half_t;         // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and output matrices.
    using LayoutInputA = cutlass::layout::ColumnMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int OutputAlignmentA = 8;
    const static int OutputAlignmentB = 8;
    const static int OutputEpilogueComputedPerOp = 8;
    const static int OutputNumStages = 2;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using OutputMMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using OutputSmArch = cutlass::arch::Sm75;

    // This code section describes the tile size a thread block will compute
    using OutputShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 32>;  // <- threadblock tile M = 64, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using OutputShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;  // <- warp tile M = 32, N = 32, K = 32 
    // This code section describes the size of MMA op
    using OutputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

    // Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
    //
    //    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
    //
    using OutputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        OutputEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using OutputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        OutputMMAOp,
        OutputSmArch,
        OutputShapeMMAThreadBlock,
        OutputShapeMMAWarp,
        OutputShapeMMAOp,
        OutputEpilogueOp,
        SwizzleThreadBlock,
        OutputNumStages,
        OutputAlignmentA,
        OutputAlignmentB>;

    // ==============
    // Hidden Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h16816gemm_128x64_32x6_tt_align8_
    //using cutlass_tensorop_h16816gemm_128x64_64x3_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 64, 64>,
    //    cutlass::gemm::GemmShape<64, 32, 64>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    3,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    const static int HiddenAlignmentA = 8;
    const static int HiddenAlignmentB = 8;
    const static int HiddenEpilogueComputedPerOp = 8;
    const static int HiddenNumStages = 3;

    using HiddenEpilogueOp = SineEpilogue<
        ElementOutput,                                        // <- data type of output matrix
        HiddenEpilogueComputedPerOp,
        W0,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue
    >;

    using HiddenMMAOp = cutlass::arch::OpClassTensorOp;
    using HiddenSmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using HiddenShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 64>;  // <- threadblock tile M = 128, N = 64, K = 64
    // This code section describes tile size a warp will compute
    using HiddenShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 64>;  // <- warp tile M = 64, N = 32, K = 64 
    // This code section describes the size of MMA op
    using HiddenShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    using HiddenGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        HiddenMMAOp,
        HiddenSmArch,
        HiddenShapeMMAThreadBlock,
        HiddenShapeMMAWarp,
        HiddenShapeMMAOp,
        HiddenEpilogueOp,
        SwizzleThreadBlock,
        HiddenNumStages,
        HiddenAlignmentA,
        HiddenAlignmentB>;

    // ==============
    // Input Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h1688gemm_128x64_32x2_tt_align4
    //using cutlass_tensorop_h1688gemm_128x64_32x2_tt_align4_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    4,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    using InputElementAccumulator = cutlass::half_t;
    using InputElementComputeEpilogue = InputElementAccumulator;
    using InputMMAOp = cutlass::arch::OpClassTensorOp;
    using InputSmArch = cutlass::arch::Sm75;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int InputAlignmentA = 4;
    const static int InputAlignmentB = 4;
    const static int InputEpilogueComputedPerOp = 4;
    const static int InputNumStages = 2;

    // This code section describes the tile size a thread block will compute
    using InputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using InputShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32 
    // This code section describes the size of MMA op
    using InputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    using InputEpilogueOp = SineEpilogue<
        ElementOutput,                                        // <- data type of output matrix
        InputEpilogueComputedPerOp,
        W0,
        InputElementAccumulator,                                   // <- data type of accumulator
        InputElementComputeEpilogue
    >;

    using InputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        InputElementAccumulator,
        InputMMAOp,
        InputSmArch,
        InputShapeMMAThreadBlock,
        InputShapeMMAWarp,
        InputShapeMMAOp,
        InputEpilogueOp,
        SwizzleThreadBlock,
        InputNumStages,
        InputAlignmentA,
        InputAlignmentB>;
};

template<int W0>
struct RTX30903DGemms_512x512 {

    static constexpr int point_size = 3;

    static const int resolution_x = 512;
    static const int resolution_y = 512;

    static const int block_x = 16;
    static const int block_y = 16;

    static const int W0_ = W0;

    // =================
    // Output layer.
    // =================

    // Gemm operator cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8
    //using cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = cutlass::half_t;         // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and output matrices.
    using LayoutInputA = cutlass::layout::ColumnMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int OutputAlignmentA = 8;
    const static int OutputAlignmentB = 8;
    const static int OutputEpilogueComputedPerOp = 8;
    const static int OutputNumStages = 2;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using OutputMMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using OutputSmArch = cutlass::arch::Sm75;

    // This code section describes the tile size a thread block will compute
    using OutputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using OutputShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32
    // This code section describes the size of MMA op
    using OutputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

    // Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
    //
    //    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
    //
    using OutputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        OutputEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using OutputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        OutputMMAOp,
        OutputSmArch,
        OutputShapeMMAThreadBlock,
        OutputShapeMMAWarp,
        OutputShapeMMAOp,
        OutputEpilogueOp,
        SwizzleThreadBlock,
        OutputNumStages,
        OutputAlignmentA,
        OutputAlignmentB>;

    // =============
    // Hidden Layer
    // =============

    // Profiled operator: cutlass_tensorop_h16816gemm_128x128_32x3_tt_align8
    //using cutlass_tensorop_h16816gemm_128x128_32x3_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 128, 32>,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    3,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    const static int HiddenAlignmentA = 8;
    const static int HiddenAlignmentB = 8;
    const static int HiddenEpilogueComputedPerOp = 8;
    const static int HiddenNumStages = 3;

    using HiddenEpilogueOp = SineEpilogue<
        ElementOutput,                                        // <- data type of output matrix
        HiddenEpilogueComputedPerOp,
        W0,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue
    >;

    using HiddenMMAOp = cutlass::arch::OpClassTensorOp;
    using HiddenSmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using HiddenShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
    // This code section describes tile size a warp will compute
    using HiddenShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
    // This code section describes the size of MMA op
    using HiddenShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    using HiddenGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        HiddenMMAOp,
        HiddenSmArch,
        HiddenShapeMMAThreadBlock,
        HiddenShapeMMAWarp,
        HiddenShapeMMAOp,
        HiddenEpilogueOp,
        SwizzleThreadBlock,
        HiddenNumStages,
        HiddenAlignmentA,
        HiddenAlignmentB>;

    // ==============
    // Input Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h1688gemm_128x64_32x2_tt_align1
    //using cutlass_tensorop_h1688gemm_128x64_32x2_tt_align1_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    1,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    using InputElementAccumulator = cutlass::half_t;
    using InputElementComputeEpilogue = InputElementAccumulator;
    using InputMMAOp = cutlass::arch::OpClassTensorOp;
    using InputSmArch = cutlass::arch::Sm75;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int InputAlignmentA = 1;
    const static int InputAlignmentB = 1;
    const static int InputEpilogueComputedPerOp = 1;
    const static int InputNumStages = 2;

    // This code section describes the tile size a thread block will compute
    using InputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using InputShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32 
    // This code section describes the size of MMA op
    using InputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    using InputEpilogueOp = SineEpilogue<
        ElementOutput,                                        // <- data type of output matrix
        InputEpilogueComputedPerOp,
        W0,
        InputElementAccumulator,                                   // <- data type of accumulator
        InputElementComputeEpilogue
    >;

    using InputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        InputElementAccumulator,
        InputMMAOp,
        InputSmArch,
        InputShapeMMAThreadBlock,
        InputShapeMMAWarp,
        InputShapeMMAOp,
        InputEpilogueOp,
        SwizzleThreadBlock,
        InputNumStages,
        InputAlignmentA,
        InputAlignmentB>;
};

template<int W0>
struct RTX30903DGemms_1024x1024 {

    static constexpr int point_size = 3;

    static const int resolution_x = 1024;
    static const int resolution_y = 1024;

    static const int block_x = 16;
    static const int block_y = 16;

    static const int W0_ = W0;

    // =================
    // Output layer.
    // =================

    // Gemm operator cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8
    //using cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = cutlass::half_t;         // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and output matrices.
    using LayoutInputA = cutlass::layout::ColumnMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int OutputAlignmentA = 8;
    const static int OutputAlignmentB = 8;
    const static int OutputEpilogueComputedPerOp = 8;
    const static int OutputNumStages = 2;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using OutputMMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using OutputSmArch = cutlass::arch::Sm75;

    // This code section describes the tile size a thread block will compute
    using OutputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using OutputShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32
    // This code section describes the size of MMA op
    using OutputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

    // Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
    //
    //    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
    //
    using OutputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        OutputEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using OutputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        OutputMMAOp,
        OutputSmArch,
        OutputShapeMMAThreadBlock,
        OutputShapeMMAWarp,
        OutputShapeMMAOp,
        OutputEpilogueOp,
        SwizzleThreadBlock,
        OutputNumStages,
        OutputAlignmentA,
        OutputAlignmentB>;

    // =============
    // Hidden Layer
    // =============

    // Profiled operator: cutlass_tensorop_h16816gemm_128x128_32x3_tt_align8
    //using cutlass_tensorop_h16816gemm_128x128_32x3_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 128, 32>,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    3,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    const static int HiddenAlignmentA = 8;
    const static int HiddenAlignmentB = 8;
    const static int HiddenEpilogueComputedPerOp = 8;
    const static int HiddenNumStages = 3;

    using HiddenEpilogueOp = SineEpilogue<
        ElementOutput,                                        // <- data type of output matrix
        HiddenEpilogueComputedPerOp,
        W0,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue
    >;

    using HiddenMMAOp = cutlass::arch::OpClassTensorOp;
    using HiddenSmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using HiddenShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
    // This code section describes tile size a warp will compute
    using HiddenShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
    // This code section describes the size of MMA op
    using HiddenShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    using HiddenGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        HiddenMMAOp,
        HiddenSmArch,
        HiddenShapeMMAThreadBlock,
        HiddenShapeMMAWarp,
        HiddenShapeMMAOp,
        HiddenEpilogueOp,
        SwizzleThreadBlock,
        HiddenNumStages,
        HiddenAlignmentA,
        HiddenAlignmentB>;

    // ==============
    // Input Layer.
    // ==============

    // Gemm operator cutlass_tensorop_h1688gemm_64x128_32x2_tt_align1
    //using cutlass_tensorop_h1688gemm_64x128_32x2_tt_align1_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<64, 128, 32>,
    //    cutlass::gemm::GemmShape<32, 64, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    1,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    using InputElementAccumulator = cutlass::half_t;
    using InputElementComputeEpilogue = InputElementAccumulator;
    using InputMMAOp = cutlass::arch::OpClassTensorOp;
    using InputSmArch = cutlass::arch::Sm75;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int InputAlignmentA = 1;
    const static int InputAlignmentB = 1;
    const static int InputEpilogueComputedPerOp = 1;
    const static int InputNumStages = 2;

    // This code section describes the tile size a thread block will compute
    using InputShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 128, 32>;  // <- threadblock tile M = 64, N = 128, K = 32
    // This code section describes tile size a warp will compute
    using InputShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 32>;  // <- warp tile M = 32, N = 64, K = 32 
    // This code section describes the size of MMA op
    using InputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    using InputEpilogueOp = SineEpilogue<
        ElementOutput,                                        // <- data type of output matrix
        InputEpilogueComputedPerOp,
        W0,
        InputElementAccumulator,                                   // <- data type of accumulator
        InputElementComputeEpilogue
    >;

    using InputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        InputElementAccumulator,
        InputMMAOp,
        InputSmArch,
        InputShapeMMAThreadBlock,
        InputShapeMMAWarp,
        InputShapeMMAOp,
        InputEpilogueOp,
        SwizzleThreadBlock,
        InputNumStages,
        InputAlignmentA,
        InputAlignmentB>;
};

struct RTX3090NormalGemms_512x512 {

    static constexpr int point_size = 4;

    static const int resolution_x = 512;
    static const int resolution_y = 512;

    static const int block_x = 16;
    static const int block_y = 16;

    // =================
    // Output layer.
    // =================

    // Gemm operator cutlass_simt_hgemm_128x32_8x2_tt_align1
    //using cutlass_tensorop_h16816gemm_128x128_32x3_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 128, 32>,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    3,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = cutlass::half_t;         // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and output matrices.
    using LayoutInputA = cutlass::layout::ColumnMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int OutputAlignmentA = 8;
    const static int OutputAlignmentB = 8;
    const static int OutputEpilogueComputedPerOp = 8;
    const static int OutputNumStages = 3;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using OutputMMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using OutputSmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using OutputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
    // This code section describes tile size a warp will compute
    using OutputShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
    // This code section describes the size of MMA op
    using OutputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

    // Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
    //
    //    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
    //
    using OutputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        OutputEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using OutputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        OutputMMAOp,
        OutputSmArch,
        OutputShapeMMAThreadBlock,
        OutputShapeMMAWarp,
        OutputShapeMMAOp,
        OutputEpilogueOp,
        SwizzleThreadBlock,
        OutputNumStages,
        OutputAlignmentA,
        OutputAlignmentB>;

    // ==============
    // Hidden Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h16816gemm_128x64_32x6_tt_align8_
    //using cutlass_tensorop_h16816gemm_128x64_32x6_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    6,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    const static int HiddenAlignmentA = 8;
    const static int HiddenAlignmentB = 8;
    const static int HiddenEpilogueComputedPerOp = 8;
    const static int HiddenNumStages = 6;

    using HiddenEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        HiddenEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using HiddenMMAOp = cutlass::arch::OpClassTensorOp;
    using HiddenSmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using HiddenShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using HiddenShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32 
    // This code section describes the size of MMA op
    using HiddenShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    using HiddenGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        HiddenMMAOp,
        HiddenSmArch,
        HiddenShapeMMAThreadBlock,
        HiddenShapeMMAWarp,
        HiddenShapeMMAOp,
        HiddenEpilogueOp,
        SwizzleThreadBlock,
        HiddenNumStages,
        HiddenAlignmentA,
        HiddenAlignmentB>;

    // ==============
    // Input Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h16816gemm_256x64_32x4_tt_align4
    //using cutlass_tensorop_h16816gemm_256x64_32x4_tt_align4_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<256, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    4,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    4,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    using InputElementAccumulator = cutlass::half_t;
    using InputElementComputeEpilogue = InputElementAccumulator;
    using InputMMAOp = cutlass::arch::OpClassTensorOp;
    using InputSmArch = cutlass::arch::Sm80;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int InputAlignmentA = 4;
    const static int InputAlignmentB = 4;
    const static int InputEpilogueComputedPerOp = 4;
    const static int InputNumStages = 4;

    // This code section describes the tile size a thread block will compute
    using InputShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 64, 32>;  // <- threadblock tile M = 256, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using InputShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
    // This code section describes the size of MMA op
    using InputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    using InputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                          // <- data type of output matrix
        InputEpilogueComputedPerOp,
        InputElementAccumulator,                                // <- data type of accumulator
        InputElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using InputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        InputElementAccumulator,
        InputMMAOp,
        InputSmArch,
        InputShapeMMAThreadBlock,
        InputShapeMMAWarp,
        InputShapeMMAOp,
        InputEpilogueOp,
        SwizzleThreadBlock,
        InputNumStages,
        InputAlignmentA,
        InputAlignmentB>;
};

struct RTX3090NormalGemms_256_hidden_512x512 {

    static constexpr int point_size = 4;

    static const int resolution_x = 512;
    static const int resolution_y = 512;

    static const int block_x = 16;
    static const int block_y = 16;

    // =================
    // Output layer.
    // =================

    // Gemm operator cutlass_tensorop_h16816gemm_128x64_64x3_tt_align8
    //using cutlass_tensorop_h16816gemm_128x64_64x3_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 64, 64>,
    //    cutlass::gemm::GemmShape<64, 32, 64>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    3,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = cutlass::half_t;         // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and output matrices.
    using LayoutInputA = cutlass::layout::ColumnMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int OutputAlignmentA = 8;
    const static int OutputAlignmentB = 8;
    const static int OutputEpilogueComputedPerOp = 8;
    const static int OutputNumStages = 3;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using OutputMMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using OutputSmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using OutputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 64>;  // <- threadblock tile M = 128, N = 64, K = 64
    // This code section describes tile size a warp will compute
    using OutputShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 64>;  // <- warp tile M = 64, N = 32, K = 64 
    // This code section describes the size of MMA op
    using OutputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

    // Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
    //
    //    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
    //
    using OutputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        OutputEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using OutputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        OutputMMAOp,
        OutputSmArch,
        OutputShapeMMAThreadBlock,
        OutputShapeMMAWarp,
        OutputShapeMMAOp,
        OutputEpilogueOp,
        SwizzleThreadBlock,
        OutputNumStages,
        OutputAlignmentA,
        OutputAlignmentB>;

    // ==============
    // Hidden Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h16816gemm_128x128_32x3_tt_align8
    //using cutlass_tensorop_h16816gemm_128x128_32x3_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 128, 32>,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    3,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    const static int HiddenAlignmentA = 8;
    const static int HiddenAlignmentB = 8;
    const static int HiddenEpilogueComputedPerOp = 8;
    const static int HiddenNumStages = 3;

    using HiddenEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        HiddenEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using HiddenMMAOp = cutlass::arch::OpClassTensorOp;
    using HiddenSmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using HiddenShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
    // This code section describes tile size a warp will compute
    using HiddenShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
    // This code section describes the size of MMA op
    using HiddenShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    using HiddenGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        HiddenMMAOp,
        HiddenSmArch,
        HiddenShapeMMAThreadBlock,
        HiddenShapeMMAWarp,
        HiddenShapeMMAOp,
        HiddenEpilogueOp,
        SwizzleThreadBlock,
        HiddenNumStages,
        HiddenAlignmentA,
        HiddenAlignmentB>;

    // ==============
    // Input Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h16816gemm_128x128_32x5_tt_align4
    //using cutlass_tensorop_h16816gemm_128x128_32x5_tt_align4_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 128, 32>,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    4,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    5,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    using InputElementAccumulator = cutlass::half_t;
    using InputElementComputeEpilogue = InputElementAccumulator;
    using InputMMAOp = cutlass::arch::OpClassTensorOp;
    using InputSmArch = cutlass::arch::Sm80;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int InputAlignmentA = 4;
    const static int InputAlignmentB = 4;
    const static int InputEpilogueComputedPerOp = 4;
    const static int InputNumStages = 5;

    // This code section describes the tile size a thread block will compute
    using InputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
    // This code section describes tile size a warp will compute
    using InputShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
    // This code section describes the size of MMA op
    using InputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    using InputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                          // <- data type of output matrix
        InputEpilogueComputedPerOp,
        InputElementAccumulator,                                // <- data type of accumulator
        InputElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using InputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        InputElementAccumulator,
        InputMMAOp,
        InputSmArch,
        InputShapeMMAThreadBlock,
        InputShapeMMAWarp,
        InputShapeMMAOp,
        InputEpilogueOp,
        SwizzleThreadBlock,
        InputNumStages,
        InputAlignmentA,
        InputAlignmentB>;
};

struct RTX3090NormalGemms_1024x1024 {

    static constexpr int point_size = 4;

    static const int resolution_x = 1024;
    static const int resolution_y = 1024;

    static const int block_x = 8;
    static const int block_y = 32;

    // =================
    // Output layer.
    // =================

    // Profiled operator: cutlass_tensorop_h1688gemm_64x64_32x2_tt_align8_base
    //using cutlass_tensorop_h1688gemm_64x64_32x2_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<32, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = cutlass::half_t;         // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and output matrices.
    using LayoutInputA = cutlass::layout::ColumnMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int OutputAlignmentA = 8;
    const static int OutputAlignmentB = 8;
    const static int OutputEpilogueComputedPerOp = 8;
    const static int OutputNumStages = 2;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using OutputMMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using OutputSmArch = cutlass::arch::Sm75;

    // This code section describes the tile size a thread block will compute
    using OutputShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 32>;  // <- threadblock tile M = 64, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using OutputShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;  // <- warp tile M = 32, N = 32, K = 32 
    // This code section describes the size of MMA op
    using OutputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

    // Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
    //
    //    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
    //
    using OutputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        OutputEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using OutputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        OutputMMAOp,
        OutputSmArch,
        OutputShapeMMAThreadBlock,
        OutputShapeMMAWarp,
        OutputShapeMMAOp,
        OutputEpilogueOp,
        SwizzleThreadBlock,
        OutputNumStages,
        OutputAlignmentA,
        OutputAlignmentB>;

    // ==============
    // Hidden Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h16816gemm_128x64_32x6_tt_align4
    //using cutlass_tensorop_h16816gemm_128x64_32x6_tt_align4_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    4,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    6,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    const static int HiddenAlignmentA = 4;
    const static int HiddenAlignmentB = 4;
    const static int HiddenEpilogueComputedPerOp = 4;
    const static int HiddenNumStages = 6;

    using HiddenEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        HiddenEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using HiddenMMAOp = cutlass::arch::OpClassTensorOp;
    using HiddenSmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using HiddenShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using HiddenShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32 
    // This code section describes the size of MMA op
    using HiddenShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    using HiddenGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        HiddenMMAOp,
        HiddenSmArch,
        HiddenShapeMMAThreadBlock,
        HiddenShapeMMAWarp,
        HiddenShapeMMAOp,
        HiddenEpilogueOp,
        SwizzleThreadBlock,
        HiddenNumStages,
        HiddenAlignmentA,
        HiddenAlignmentB>;

    // ==============
    // Input Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h1688gemm_128x64_32x2_tt_align4
    //using cutlass_tensorop_h1688gemm_128x64_32x2_tt_align4_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    4,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    using InputElementAccumulator = cutlass::half_t;
    using InputElementComputeEpilogue = InputElementAccumulator;
    using InputMMAOp = cutlass::arch::OpClassTensorOp;
    using InputSmArch = cutlass::arch::Sm75;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int InputAlignmentA = 4;
    const static int InputAlignmentB = 4;
    const static int InputEpilogueComputedPerOp = 4;
    const static int InputNumStages = 2;

    // This code section describes the tile size a thread block will compute
    using InputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using InputShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32 
    // This code section describes the size of MMA op
    using InputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    using InputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                          // <- data type of output matrix
        InputEpilogueComputedPerOp,
        InputElementAccumulator,                                // <- data type of accumulator
        InputElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using InputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        InputElementAccumulator,
        InputMMAOp,
        InputSmArch,
        InputShapeMMAThreadBlock,
        InputShapeMMAWarp,
        InputShapeMMAOp,
        InputEpilogueOp,
        SwizzleThreadBlock,
        InputNumStages,
        InputAlignmentA,
        InputAlignmentB>;
};

struct RTX3090NormalGemms_1920x1080 {

    static constexpr int point_size = 4;

    static const int resolution_x = 1920;
    static const int resolution_y = 1080;

    static const int block_x = 8;
    static const int block_y = 32;

    // =================
    // Output layer.
    // =================

    // Profiled operator: cutlass_tensorop_h1688gemm_64x64_32x2_tt_align8_base
    //using cutlass_tensorop_h1688gemm_64x64_32x2_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<32, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = cutlass::half_t;         // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and output matrices.
    using LayoutInputA = cutlass::layout::ColumnMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int OutputAlignmentA = 8;
    const static int OutputAlignmentB = 8;
    const static int OutputEpilogueComputedPerOp = 8;
    const static int OutputNumStages = 2;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using OutputMMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using OutputSmArch = cutlass::arch::Sm75;

    // This code section describes the tile size a thread block will compute
    using OutputShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 32>;  // <- threadblock tile M = 64, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using OutputShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;  // <- warp tile M = 32, N = 32, K = 32 
    // This code section describes the size of MMA op
    using OutputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

    // Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
    //
    //    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
    //
    using OutputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        OutputEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using OutputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        OutputMMAOp,
        OutputSmArch,
        OutputShapeMMAThreadBlock,
        OutputShapeMMAWarp,
        OutputShapeMMAOp,
        OutputEpilogueOp,
        SwizzleThreadBlock,
        OutputNumStages,
        OutputAlignmentA,
        OutputAlignmentB>;

    // ==============
    // Hidden Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h16816gemm_128x64_32x6_tt_align8_
    //using cutlass_tensorop_h16816gemm_128x64_64x3_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 64, 64>,
    //    cutlass::gemm::GemmShape<64, 32, 64>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    3,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    const static int HiddenAlignmentA = 8;
    const static int HiddenAlignmentB = 8;
    const static int HiddenEpilogueComputedPerOp = 8;
    const static int HiddenNumStages = 3;

    using HiddenEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        HiddenEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using HiddenMMAOp = cutlass::arch::OpClassTensorOp;
    using HiddenSmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using HiddenShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 64>;  // <- threadblock tile M = 128, N = 64, K = 64
    // This code section describes tile size a warp will compute
    using HiddenShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 64>;  // <- warp tile M = 64, N = 32, K = 64 
    // This code section describes the size of MMA op
    using HiddenShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    using HiddenGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        HiddenMMAOp,
        HiddenSmArch,
        HiddenShapeMMAThreadBlock,
        HiddenShapeMMAWarp,
        HiddenShapeMMAOp,
        HiddenEpilogueOp,
        SwizzleThreadBlock,
        HiddenNumStages,
        HiddenAlignmentA,
        HiddenAlignmentB>;

    // ==============
    // Input Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h1688gemm_128x64_32x2_tt_align4
    //using cutlass_tensorop_h1688gemm_128x64_32x2_tt_align4_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    4,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    using InputElementAccumulator = cutlass::half_t;
    using InputElementComputeEpilogue = InputElementAccumulator;
    using InputMMAOp = cutlass::arch::OpClassTensorOp;
    using InputSmArch = cutlass::arch::Sm75;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int InputAlignmentA = 4;
    const static int InputAlignmentB = 4;
    const static int InputEpilogueComputedPerOp = 4;
    const static int InputNumStages = 2;

    // This code section describes the tile size a thread block will compute
    using InputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using InputShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32 
    // This code section describes the size of MMA op
    using InputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    using InputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                          // <- data type of output matrix
        InputEpilogueComputedPerOp,
        InputElementAccumulator,                                // <- data type of accumulator
        InputElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using InputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        InputElementAccumulator,
        InputMMAOp,
        InputSmArch,
        InputShapeMMAThreadBlock,
        InputShapeMMAWarp,
        InputShapeMMAOp,
        InputEpilogueOp,
        SwizzleThreadBlock,
        InputNumStages,
        InputAlignmentA,
        InputAlignmentB>;
};

struct RTX30903DNormalGemms_512x512 {

    static constexpr int point_size = 3;

    static const int resolution_x = 512;
    static const int resolution_y = 512;

    static const int block_x = 16;
    static const int block_y = 16;

    // =================
    // Output layer.
    // =================

    // Gemm operator cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8
    //using cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = cutlass::half_t;         // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and output matrices.
    using LayoutInputA = cutlass::layout::ColumnMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int OutputAlignmentA = 8;
    const static int OutputAlignmentB = 8;
    const static int OutputEpilogueComputedPerOp = 8;
    const static int OutputNumStages = 2;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using OutputMMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using OutputSmArch = cutlass::arch::Sm75;

    // This code section describes the tile size a thread block will compute
    using OutputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using OutputShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32
    // This code section describes the size of MMA op
    using OutputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

    // Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
    //
    //    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
    //
    using OutputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        OutputEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using OutputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        OutputMMAOp,
        OutputSmArch,
        OutputShapeMMAThreadBlock,
        OutputShapeMMAWarp,
        OutputShapeMMAOp,
        OutputEpilogueOp,
        SwizzleThreadBlock,
        OutputNumStages,
        OutputAlignmentA,
        OutputAlignmentB>;

    // =============
    // Hidden Layer
    // =============

    // Profiled operator: cutlass_tensorop_h16816gemm_128x128_32x3_tt_align8
    //using cutlass_tensorop_h16816gemm_128x128_32x3_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 128, 32>,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    3,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    const static int HiddenAlignmentA = 8;
    const static int HiddenAlignmentB = 8;
    const static int HiddenEpilogueComputedPerOp = 8;
    const static int HiddenNumStages = 3;

    using HiddenEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        HiddenEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using HiddenMMAOp = cutlass::arch::OpClassTensorOp;
    using HiddenSmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using HiddenShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
    // This code section describes tile size a warp will compute
    using HiddenShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
    // This code section describes the size of MMA op
    using HiddenShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    using HiddenGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        HiddenMMAOp,
        HiddenSmArch,
        HiddenShapeMMAThreadBlock,
        HiddenShapeMMAWarp,
        HiddenShapeMMAOp,
        HiddenEpilogueOp,
        SwizzleThreadBlock,
        HiddenNumStages,
        HiddenAlignmentA,
        HiddenAlignmentB>;

    // ==============
    // Input Layer.
    // ==============

    // Profiled operator: cutlass_tensorop_h1688gemm_128x64_32x2_tt_align1
    //using cutlass_tensorop_h1688gemm_128x64_32x2_tt_align1_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    1,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    using InputElementAccumulator = cutlass::half_t;
    using InputElementComputeEpilogue = InputElementAccumulator;
    using InputMMAOp = cutlass::arch::OpClassTensorOp;
    using InputSmArch = cutlass::arch::Sm75;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int InputAlignmentA = 1;
    const static int InputAlignmentB = 1;
    const static int InputEpilogueComputedPerOp = 1;
    const static int InputNumStages = 2;

    // This code section describes the tile size a thread block will compute
    using InputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using InputShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32 
    // This code section describes the size of MMA op
    using InputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    using InputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        InputEpilogueComputedPerOp,
        InputElementAccumulator,                                   // <- data type of accumulator
        InputElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using InputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        InputElementAccumulator,
        InputMMAOp,
        InputSmArch,
        InputShapeMMAThreadBlock,
        InputShapeMMAWarp,
        InputShapeMMAOp,
        InputEpilogueOp,
        SwizzleThreadBlock,
        InputNumStages,
        InputAlignmentA,
        InputAlignmentB>;
};

struct RTX30903DNormalGemms_1024x1024 {

    static constexpr int point_size = 3;

    static const int resolution_x = 1024;
    static const int resolution_y = 1024;

    static const int block_x = 8;
    static const int block_y = 32;

    // =================
    // Output layer.
    // =================

    // Gemm operator cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8
    //using cutlass_tensorop_h1688gemm_128x64_32x2_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<128, 64, 32>,
    //    cutlass::gemm::GemmShape<64, 32, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = cutlass::half_t;         // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and output matrices.
    using LayoutInputA = cutlass::layout::ColumnMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int OutputAlignmentA = 8;
    const static int OutputAlignmentB = 8;
    const static int OutputEpilogueComputedPerOp = 8;
    const static int OutputNumStages = 2;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using OutputMMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using OutputSmArch = cutlass::arch::Sm75;

    // This code section describes the tile size a thread block will compute
    using OutputShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 64, K = 32
    // This code section describes tile size a warp will compute
    using OutputShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 32, K = 32
    // This code section describes the size of MMA op
    using OutputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

    // Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
    //
    //    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
    //
    using OutputEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                        // <- data type of output matrix
        OutputEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using OutputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        OutputMMAOp,
        OutputSmArch,
        OutputShapeMMAThreadBlock,
        OutputShapeMMAWarp,
        OutputShapeMMAOp,
        OutputEpilogueOp,
        SwizzleThreadBlock,
        OutputNumStages,
        OutputAlignmentA,
        OutputAlignmentB>;

    // =============
    // Hidden Layer
    // =============

    // Profiled operator: cutlass_tensorop_h16816gemm_128x128_32x3_tt_align8
    //using cutlass_tensorop_h16816gemm_128x128_32x3_tt_align8_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm80,
    //    cutlass::gemm::GemmShape<128, 128, 32>,
    //    cutlass::gemm::GemmShape<64, 64, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 16>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    8,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    3,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    const static int HiddenAlignmentA = 8;
    const static int HiddenAlignmentB = 8;
    const static int HiddenEpilogueComputedPerOp = 8;
    const static int HiddenNumStages = 3;

    using HiddenEpilogueOp = cutlass::epilogue::thread::LinearCombination <
        ElementOutput,                                        // <- data type of output matrix
        HiddenEpilogueComputedPerOp,
        ElementAccumulator,                                   // <- data type of accumulator
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using HiddenMMAOp = cutlass::arch::OpClassTensorOp;
    using HiddenSmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using HiddenShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
    // This code section describes tile size a warp will compute
    using HiddenShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
    // This code section describes the size of MMA op
    using HiddenShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16

    using HiddenGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        HiddenMMAOp,
        HiddenSmArch,
        HiddenShapeMMAThreadBlock,
        HiddenShapeMMAWarp,
        HiddenShapeMMAOp,
        HiddenEpilogueOp,
        SwizzleThreadBlock,
        HiddenNumStages,
        HiddenAlignmentA,
        HiddenAlignmentB>;

    // ==============
    // Input Layer.
    // ==============

    // Gemm operator cutlass_tensorop_h1688gemm_64x128_32x2_tt_align1
    //using cutlass_tensorop_h1688gemm_64x128_32x2_tt_align1_base =
    //    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,    // transposed B operand
    //    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,    // transposed A operand
    //    cutlass::half_t, cutlass::layout::RowMajor,
    //    cutlass::half_t,
    //    cutlass::arch::OpClassTensorOp,
    //    cutlass::arch::Sm75,
    //    cutlass::gemm::GemmShape<64, 128, 32>,
    //    cutlass::gemm::GemmShape<32, 64, 32>,
    //    cutlass::gemm::GemmShape<16, 8, 8>,
    //    cutlass::epilogue::thread::LinearCombination<
    //    cutlass::half_t,
    //    1,
    //    cutlass::half_t,
    //    cutlass::half_t
    //    >,
    //    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    //    2,
    //    cutlass::arch::OpMultiplyAdd
    //    >::GemmKernel;

    using InputElementAccumulator = cutlass::half_t;
    using InputElementComputeEpilogue = InputElementAccumulator;
    using InputMMAOp = cutlass::arch::OpClassTensorOp;
    using InputSmArch = cutlass::arch::Sm75;

    // Matrices alignment and number of elements computed per epilogue operation.
    const static int InputAlignmentA = 1;
    const static int InputAlignmentB = 1;
    const static int InputEpilogueComputedPerOp = 1;
    const static int InputNumStages = 2;

    // This code section describes the tile size a thread block will compute
    using InputShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 128, 32>;  // <- threadblock tile M = 64, N = 128, K = 32
    // This code section describes tile size a warp will compute
    using InputShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 32>;  // <- warp tile M = 32, N = 64, K = 32 
    // This code section describes the size of MMA op
    using InputShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

    using InputEpilogueOp = cutlass::epilogue::thread::LinearCombination <
        ElementOutput,                                        // <- data type of output matrix
        InputEpilogueComputedPerOp,
        InputElementAccumulator,                                   // <- data type of accumulator
        InputElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

    using InputGemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        InputElementAccumulator,
        InputMMAOp,
        InputSmArch,
        InputShapeMMAThreadBlock,
        InputShapeMMAWarp,
        InputShapeMMAOp,
        InputEpilogueOp,
        SwizzleThreadBlock,
        InputNumStages,
        InputAlignmentA,
        InputAlignmentB>;
};

// Instantiations:

// RTX2070Gemms_512x512<30> RTX_2070_512_30;
// RTX2070Gemms_512x512<20> RTX_2070_512_20;
// RTX2070Gemms_512x512<16> RTX_2070_512_16;

// RTX3090Gemms_512x512<30> RTX_3090_512_30;
// RTX3090Gemms_512x512<20> RTX_3090_512_20;
// RTX3090Gemms_512x512<16> RTX_3090_512_16;

// RTX3090Gemms_256_hidden_512x512<30> RTX_3090_256_30;
// RTX3090Gemms_256_hidden_512x512<20> RTX_3090_256_20;
// RTX3090Gemms_256_hidden_512x512<16> RTX_3090_256_16;

// RTX3090Gemms_1024x1024<30> RTX_3090_1024_30;
// RTX3090Gemms_1024x1024<20> RTX_3090_1024_20;
// RTX3090Gemms_1024x1024<16> RTX_3090_1024_16;

// RTX3090Gemms_1920x1080<30> RTX_3090_1920_30;
// RTX3090Gemms_1920x1080<20> RTX_3090_1920_20;
// RTX3090Gemms_1920x1080<16> RTX_3090_1920_16;

// RTX30903DGemms_512x512<30> RTX_3090_3D_512_30;
// RTX30903DGemms_512x512<20> RTX_3090_3D_512_20;
// RTX30903DGemms_512x512<16> RTX_3090_3D_512_16;