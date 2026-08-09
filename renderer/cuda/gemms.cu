// =======================================================
// Layer GEMM operation which composes a Neural Implicit.
// =======================================================

template<typename Gemm>
class LayerGemm
{
public:
    LayerGemm() {}

    LayerGemm(const cutlass::gemm::GemmCoord& problem_size_) {
        init(problem_size_);
    }

    void init(const cutlass::gemm::GemmCoord& problem_size_) {

        problem_size = problem_size_;
        tensor_a.reset(problem_size.mk());
        tensor_c_bias.reset({ problem_size.m(), 1 });
        tensor_d.reset(problem_size.mn());

        cutlass::reference::host::TensorFill(
            tensor_a.host_view());
        cutlass::reference::host::TensorFill(
            tensor_c_bias.host_view());
        cutlass::reference::host::TensorFill(
            tensor_d.host_view());  // <- fill matrix D on host with zeros

        // Copy data from host to GPU
        sync_device();
    }

    void sync_device() {

        tensor_a.sync_device();
        tensor_c_bias.sync_device();
        tensor_d.sync_device();
    }

    void sync_host() {

        tensor_a.sync_host();
        tensor_c_bias.sync_host();
        tensor_d.sync_host();
    }

    void inference(cudaStream_t stream, cutlass::HostTensor<typename Gemm::ElementB, typename Gemm::LayoutB>& tensor_b) {

        // Initialize alpha for dot product computation
        auto alpha = typename Gemm::EpilogueOutputOp::ElementCompute(1);

        // Split K dimension into 1 partitions
        int split_k_slices = 1;

        // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
        // instantiated CUTLASS kernel
        typename Gemm::Arguments arguments{
          problem_size,                       // <- problem size of matrix multiplication
          tensor_a.device_ref(),              // <- reference to matrix A on device
          tensor_b.device_ref(),              // <- reference to matrix B on device

          {tensor_c_bias.device_data(), 0},   // <- the C matrix is treated as the bias vector. We can enable the GEMM
                                              //    to project away the N dimension by setting the stride to zero.

          tensor_d.device_ref(),              // <- reference to matrix D on device
          {alpha},                              // <- alpha
          split_k_slices };                    // <- k-dimension split factor

        // Using the arguments, query for extra workspace required for matrix multiplication computation
        size_t workspace_size = Gemm::get_workspace_size(arguments);

        // Allocate workspace memory
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        // Instantiate CUTLASS kernel depending on templates
        Gemm gemm_op;

        // Check the problem size is supported or not 
        cutlass::Status status = gemm_op.can_implement(arguments);
        CUTLASS_ASSERT(status == cutlass::Status::kSuccess);
        //EXPECT_EQ(status, cutlass::Status::kSuccess);

        // Initialize CUTLASS kernel with arguments and workspace pointer
        status = gemm_op.initialize(arguments, workspace.get());
        CUTLASS_ASSERT(status == cutlass::Status::kSuccess);
        //EXPECT_EQ(status, cutlass::Status::kSuccess);

        // Launch initialized CUTLASS kernel
        {
            status = gemm_op(stream);
            CUTLASS_ASSERT(status == cutlass::Status::kSuccess);
            //EXPECT_EQ(status, cutlass::Status::kSuccess);
        }
    }

    cutlass::gemm::GemmCoord problem_size;
    cutlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_a;
    cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_c_bias;
    cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_d;
};

// ============================================================
// Neural Implicit for an image with resolution 512x512 pixels.
// ============================================================

template<typename LayerGemms>
class NeuralImplicit
{
public:
    /** The files are expected to be binary row-major linearizations of the network matrices, with the same dimensions as the tensors in NeuralImplicit512x512. */
    void init(int n_hidden_layers, int hidden_layer_size, int output_size, const string& weightsFilename, const string& biasesFilename) {

        allocate_buffers(n_hidden_layers, hidden_layer_size);

        auto weights = ReadBinaryFile(weightsFilename);

        auto biases = ReadBinaryFile(biasesFilename);

        int weights_offset = 0;
        int biases_offset = 0;

        size_t input_size = input_gemm.problem_size.k();

        FillLayer(weights, biases, weights_offset, biases_offset, true, input_gemm, hidden_layer_size, input_size);             // Input layer.

        for (auto& hidden_gemm : hidden_gemms) {

            FillLayer(weights, biases, weights_offset, biases_offset, true, hidden_gemm, hidden_layer_size, hidden_layer_size);    // Hidden layer 0.
        }

        FillLayer(weights, biases, weights_offset, biases_offset, true, output_gemm, output_size, hidden_layer_size); // Output layer (file has out size 1 instead of 8).

        sync_device();
    }

    void sync_device() {

        input_tensor.sync_device();
        input_gemm.sync_device();
        for (auto& hidden_gemm : hidden_gemms) {

            hidden_gemm.sync_device();
        }
        output_gemm.sync_device();
    }

    void inference(cudaStream_t stream,
        cutlass::HostTensor<typename LayerGemms::InputGemm::ElementB, typename LayerGemms::InputGemm::LayoutB>& input_tensor_) {

        input_gemm.inference(stream, input_tensor_);
        auto* input = &input_gemm.tensor_d;
        for (auto& hidden_gemm : hidden_gemms) {

            hidden_gemm.inference(stream, *input);
            input = &hidden_gemm.tensor_d;
        }
        output_gemm.inference(stream, *input);
    }

private:
    void allocate_buffers(int n_hidden_layers, int hidden_layer_size) {

        hidden_gemms.resize(n_hidden_layers);

        const int n_inputs = LayerGemms::resolution_x * LayerGemms::resolution_y;
        const int input_size = LayerGemms::point_size;
        const int output_size = 8;

        cutlass::gemm::GemmCoord input_problem(hidden_layer_size, n_inputs, input_size);
        cutlass::gemm::GemmCoord hidden_problem(hidden_layer_size, n_inputs, hidden_layer_size);
        cutlass::gemm::GemmCoord output_problem(output_size, n_inputs, hidden_layer_size);

        input_gemm.init(input_problem);     // Gemm: input and first hidden layers.
        for (auto& hidden_gemm : hidden_gemms) {

            hidden_gemm.init(hidden_problem);
        }
        output_gemm.init(output_problem);   // Gemm: third hidden and output layers.

        // Input tensor.
        input_tensor.reset(input_problem.kn());
        cutlass::reference::host::TensorFill(
            input_tensor.host_view());  // <- Fill matrix A on host with uniform-distribution random data
        input_tensor.sync_device();
    }

public:
    cutlass::HostTensor<typename LayerGemms::InputGemm::ElementB, typename LayerGemms::InputGemm::LayoutB> input_tensor;
    LayerGemm<typename LayerGemms::InputGemm> input_gemm;               // Gemm: input and first hidden layers.
    vector<LayerGemm<typename LayerGemms::HiddenGemm>> hidden_gemms;    // Gemm: hidden layers.
    LayerGemm<typename LayerGemms::OutputGemm> output_gemm;             // Gemm: third hidden and output layers.
};

template<int W0>
__global__ void CalculateG0(precision_t* Weights0, precision_t* a0, precision_t* G0, precision_t* p0, int coord, uint2 G0_size, int point_size);

template<int W0>
__global__ void HadamardGi(precision_t* Gi, precision_t* ai, precision_t* pi, uint2 Gi_size);

// ===============================================================================================================================================================
// Fast Neural Implicit normal coordinate calculation, using at max two GEMMs and one kernel per layer. One GEMM calculates a_i (SIREN layer without activation)
// and another G_i (gradient). The kernel applies the Hadamard multiplication on G_i and the activation function on a_i to generate p_i. Check the paper for more
// details.
// ===============================================================================================================================================================

template<typename LayerGemms, int SirenW0>
class Normals
{
public:

    ~Normals() {

        if (_stream) {

            CUDA_CHECK_THROW(cudaStreamDestroy(_stream));
        }
    }

    /** The files are expected to be binary row-major linearizations of the network matrices, with the same dimensions as the tensors in NeuralImplicit512x512. */
    void init(int n_hidden_layers, int hidden_layer_size, int output_size, const string& weightsFilename, const string& biasesFilename, const int coord_) {

        allocate_buffers(n_hidden_layers, hidden_layer_size);

        CUDA_CHECK_THROW(cudaStreamCreate(&_stream));

        coord = coord_;
        auto weights = ReadBinaryFile(weightsFilename);
        auto biases = ReadBinaryFile(biasesFilename);

        int weights_offset = 0;
        int biases_offset = 0;

        size_t input_size = input_gemm_a.problem_size.k();

        FillLayer(weights, biases, weights_offset, biases_offset, true, input_gemm_a, hidden_layer_size, input_size);           // Input layer.

        for (int i = 0; i < hidden_gemms_a.size(); ++i) {

            FillLayer(weights, biases, weights_offset, biases_offset, true, hidden_gemms_a[i], hidden_layer_size, hidden_layer_size);  // Hidden layer.
            CopyLayer(hidden_gemms_a[i], hidden_gemms_G[i]);
        }

        FillLayer(weights, biases, weights_offset, biases_offset, true, output_gemm_a, output_size, hidden_layer_size);                   // Output layer (file has out size 1 instead of 8).
        CopyLayer(output_gemm_a, output_gemm_G);

        sync_device();
    }

    void sync_device() {

        input_gemm_a.sync_device();

        for (int i = 0; i < hidden_gemms_a.size(); ++i) {

            hidden_gemms_a[i].sync_device();
            hidden_gemms_G[i].sync_device();
        }
        
        output_gemm_a.sync_device();
        output_gemm_G.sync_device();
    }

    void inference(cutlass::HostTensor<typename LayerGemms::InputGemm::ElementB, typename LayerGemms::InputGemm::LayoutB>& input_tensor) {

        _graph.capture_and_execute(_stream, false, [&]() {

            dim3 block(LayerGemms::block_x, LayerGemms::block_y, 1);
            int sbytes = 0;

            /*layer 0:
            for all p in parallel
                using a GEMM
                    a_0 = W_0(p) + b_0
                using a kernel,
                    G_0 = WS0 * W_0[col j] . cos(WS0 * a_0)
                    p_0 = sin(WS0 * a_0)*/

            {
                // Input Layer
                input_gemm_a.inference(_stream, input_tensor);
                auto W0 = input_gemm_a.tensor_a.device_data();
                auto a0 = input_gemm_a.tensor_d.device_data();
                auto G0_data = G0.device_data();
                uint2 G0_size; G0_size.x = G0.extent()[0]; G0_size.y = G0.extent()[1];
                dim3 grid(G0_size.x / block.x, G0_size.y / block.y, 1);
                CalculateG0<SirenW0> << < grid, block, sbytes, _stream >> > (W0, a0, G0_data, a0, coord, G0_size, LayerGemms::point_size);
                getLastCudaError("Calculate G0.");
            }

            /*layer 1 (2 is also the same):
            for all p in parallel
                using a GEMM
                    a_1 = W_1(p_0) + b_1
                    G_1 = (W_1 * G_0)
                using a kernel,
                    G_1 = WS0 * G_1 . cos(WS0 * a_1)
                    p_1 = sin(WS0 * a_1)*/

            auto* input_a = &input_gemm_a.tensor_d;
            auto* input_G = &G0;
            for(int i = 0; i < hidden_gemms_a.size(); ++i) {
                
                // Hidden Layer 0
                hidden_gemms_a[i].inference(_stream, *input_a);
                hidden_gemms_G[i].inference(_stream, *input_G);
                auto& Gi = hidden_gemms_G[i].tensor_d;
                auto Gi_data = Gi.device_data();
                auto ai = hidden_gemms_a[i].tensor_d.device_data();
                uint2 Gi_size; Gi_size.x = Gi.extent()[0]; Gi_size.y = Gi.extent()[1];
                dim3 grid(Gi_size.x / block.x, Gi_size.y / block.y, 1);
                HadamardGi<SirenW0> << < grid, block, sbytes, _stream >> > (Gi_data, ai, ai, Gi_size);
                getLastCudaError("Calculate hidden layer 0 Gi.");

                input_a = &hidden_gemms_a[i].tensor_d;
                input_G = &hidden_gemms_G[i].tensor_d;
            }

            /*layer 3:
            for all p in parallel
                using a GEMM
                    G_3 = (W_3 * G_2)
                using a kernel,
                    G_3 = WS0 * G_3*/

            {
                // Output Layer
                block = dim3(8, 32, 1);
                output_gemm_G.inference(_stream, *input_G);
                auto& Gi = output_gemm_G.tensor_d;
                auto Gi_data = Gi.device_data();
                uint2 Gi_size; Gi_size.x = Gi.extent()[0]; Gi_size.y = Gi.extent()[1];
                dim3 grid(Gi_size.x / block.x, Gi_size.y / block.y, 1);
                HadamardGi<SirenW0> << < grid, block, sbytes, _stream >> > (Gi_data, nullptr, nullptr, Gi_size);
                getLastCudaError("Calculate output layer Gi.");
            }
        });
    }

private:
    void allocate_buffers(int n_hidden_layers, int hidden_layer_size) {

        hidden_gemms_a.resize(n_hidden_layers);
        hidden_gemms_G.resize(n_hidden_layers);

        const int n_inputs = LayerGemms::resolution_x * LayerGemms::resolution_y;
        const int input_size = LayerGemms::point_size;
        const int output_size = 8;

        cutlass::gemm::GemmCoord input_problem(hidden_layer_size, n_inputs, input_size);
        cutlass::gemm::GemmCoord hidden_problem(hidden_layer_size, n_inputs, hidden_layer_size);
        cutlass::gemm::GemmCoord output_problem(output_size, n_inputs, hidden_layer_size);

        // Gemm: input and first hidden layers.
        input_gemm_a.init(input_problem);

        for (int i = 0; i < hidden_gemms_a.size(); ++i) {

            hidden_gemms_a[i].init(hidden_problem);
            hidden_gemms_G[i].init(hidden_problem);
        }

        // Gemm: third hidden and output layers.
        output_gemm_a.init(output_problem);
        output_gemm_G.init(output_problem);

        G0.reset(hidden_problem.kn());
        cutlass::reference::host::TensorFill(
            G0.host_view());
        G0.sync_device();
    }

public:
    int coord;

    cudaStream_t _stream = nullptr;
    CudaGraph _graph;

    //cutlass::HostTensor<typename InputGemm::ElementB, typename InputGemm::LayoutB> input_tensor;

    // Gemm: input and first hidden layers.
    LayerGemm<LayerGemms::InputGemm> input_gemm_a;     // Calculates a_0.
    cutlass::HostTensor<typename LayerGemms::InputGemm::ElementB, typename LayerGemms::InputGemm::LayoutB> G0;

    // Gemm: hidden layers.
    vector<LayerGemm<LayerGemms::HiddenGemm>> hidden_gemms_a; // Calculates a_i.
    vector<LayerGemm<LayerGemms::HiddenGemm>> hidden_gemms_G; // Calculates G_i.

    // Gemm: third hidden and output layers.
    LayerGemm<LayerGemms::OutputGemm> output_gemm_a;   // Calculates a_n.
    LayerGemm<LayerGemms::OutputGemm> output_gemm_G;   // Calculates G_n.
};