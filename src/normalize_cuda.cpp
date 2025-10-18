#include <torch/extension.h>

torch::Tensor normalize_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("normalize", &normalize_cuda, "Normalize packets (CUDA)");
}

