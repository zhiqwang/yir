#include "pass.h"

#include "relay/constant_unpooling.h"
#include "relay/inline_block.h"
#include "relay/shape_inference.h"

namespace pnnx {

void pass_level0(
    const torch::jit::Module& mod,
    std::shared_ptr<torch::jit::Graph>& g,
    const std::vector<at::Tensor>& input_tensors,
    const std::vector<at::Tensor>& input_tensors2,
    const std::vector<std::string>& module_operators) {
  inline_block(g, module_operators);

  constant_unpooling(g);

  if (!input_tensors.empty()) {
    shape_inference(mod, g, input_tensors, input_tensors2);
  }
}

} // namespace pnnx
