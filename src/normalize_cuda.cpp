#include <torch/extension.h>

torch::Tensor normalize_cuda(torch::Tensor x);
torch::Tensor normalize_cuda_optimized(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("normalize", &normalize_cuda, "Normalize packets (CUDA)");
    m.def("normalize_optimized", &normalize_cuda_optimized, "Optimized normalize (CUDA)");
}

