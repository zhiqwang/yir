#include <torch/script.h>

namespace pnnx {

void shape_inference(
    const torch::jit::Module& mod,
    std::shared_ptr<torch::jit::Graph>& graph,
    const std::vector<at::Tensor>& input_tensors,
    const std::vector<at::Tensor>& input_tensors2);

} // namespace pnnx
