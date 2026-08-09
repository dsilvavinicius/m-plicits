/** Cuda Graph. */
class CudaGraph {
public:
    ~CudaGraph();

    template <typename F>
    void capture_and_execute(cudaStream_t stream, bool skip_capture, F fun) {
        cudaStreamCaptureStatus captureStatus;
        CUDA_CHECK_THROW(cudaStreamIsCapturing(stream, &captureStatus));
        skip_capture |= captureStatus == cudaStreamCaptureStatusActive; // If the caller is already capturing, no need for a nested capture.
        if (!skip_capture) {
            CUDA_CHECK_THROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        }

        fun();

        if (!skip_capture) {
            if (m_graph) {
                CUDA_CHECK_THROW(cudaGraphDestroy(m_graph));
                m_graph = nullptr;
            }
            CUDA_CHECK_THROW(cudaStreamEndCapture(stream, &m_graph));

            cudaGraphExecUpdateResult update_result;
            cudaGraphNode_t error_node;
            if (m_graph_instance) {
                CUDA_CHECK_THROW(cudaGraphExecUpdate(m_graph_instance, m_graph, &error_node, &update_result));
            }

            if (!m_graph_instance || update_result != cudaGraphExecUpdateSuccess) {
                if (m_graph_instance) {
                    CUDA_CHECK_THROW(cudaGraphExecDestroy(m_graph_instance));
                }
                CUDA_CHECK_THROW(cudaGraphInstantiate(&m_graph_instance, m_graph, NULL, NULL, 0));
            }

            CUDA_CHECK_THROW(cudaGraphLaunch(m_graph_instance, stream));
        }
    }

    void reset();

private:
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_graph_instance = nullptr;
};

CudaGraph::~CudaGraph() {
    reset();
}

void CudaGraph::reset() {
    if (m_graph) {
        CUDA_CHECK_PRINT(cudaGraphDestroy(m_graph));
        m_graph = nullptr;
    }

    if (m_graph_instance) {
        CUDA_CHECK_PRINT(cudaGraphExecDestroy(m_graph_instance));
        m_graph_instance = nullptr;
    }
}

// Data root prepended to every checkpoint path (-data_root=<dir>); defaults to
// the executable's working directory, where CMake copies data/ post-build.
std::string g_data_root = "";

vector<precision_t> ReadBinaryFile(const string& file)
{
    const string path = g_data_root.empty() ? file : g_data_root + "/" + file;
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "FATAL: cannot open checkpoint file '%s'"
                        " (use -data_root=<dir> to point at the data folder)\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    auto charVec = vector<char>(istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    return vector<precision_t>((float*)charVec.data(), (float*)charVec.data() + charVec.size() / sizeof(float));
}

template<typename Layer>
void FillLayer(const vector<precision_t>& weights, const vector<precision_t>& biases, int& weights_offset, int& biases_offset, bool ptr_row_major,
    Layer& layer, size_t n_rows, size_t n_cols, bool skip_bias = false) {

    auto weight_tensor = layer.tensor_a.host_view();
    auto bias_tensor = layer.tensor_c_bias.host_view();

    for (int i = 0; i < n_rows; ++i) {

        for (int j = 0; j < n_cols; ++j) {

            if (ptr_row_major) {
                weight_tensor.at({ i, j }) = weights[weights_offset + i * n_cols + j]; // row major.
            }
            else {

                weight_tensor.at({ i, j }) = biases[weights_offset + j * n_rows + i]; // col major.
            }
        }

        if (!skip_bias) {
            bias_tensor.at({ i, 0 }) = biases[biases_offset + i];
        }
    }
    weights_offset += n_rows * n_cols;

    if (!skip_bias) {

        biases_offset += n_rows;
    }
}

template<typename Layer>
static void CopyLayer(Layer& from, Layer& to, bool skip_bias = true) {

    auto from_a = from.tensor_a.host_data();
    to.tensor_a.copy_in_host_to_host(from_a);

    if (!skip_bias) {

        auto from_c = from.tensor_c_bias.host_data();
        to.tensor_c_bias.copy_in_host_to_host(from_c);
    }
}