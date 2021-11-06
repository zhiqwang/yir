#ifndef PNNX_PASS_LEVEL1_H
#define PNNX_PASS_LEVEL1_H

#include <torch/csrc/jit/api/module.h>
#include <torch/script.h>
#include "ir.h"

namespace pnnx {

class FuseModulePass {
 public:
  virtual const char* match_type_str() const = 0;

  virtual const char* type_str() const = 0;

  virtual void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const;

  virtual void write(
      Operator* op,
      const std::shared_ptr<torch::jit::Graph>& graph,
      const torch::jit::Module& mod) const;
};

class FuseModulePassRegister {
 public:
  FuseModulePassRegister(const FuseModulePass* pass);
};

const std::vector<const FuseModulePass*>& get_global_pnnx_fuse_module_passes();

#define REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(CLASS) \
  static FuseModulePassRegister g_global_pnnx_fusemodulepass_##CLASS##_register(new CLASS);

void pass_level1(
    const torch::jit::Module& mod,
    const std::shared_ptr<torch::jit::Graph>& g,
    Graph& pg);

} // namespace pnnx

#endif // PNNX_PASS_LEVEL1_H
