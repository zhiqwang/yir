#include <torch/csrc/jit/passes/quantization/helper.h>

#include "trace.h"

namespace pnnx {

void FuseModulePass::write(Operator*, const std::shared_ptr<torch::jit::Graph>&) const {}

void FuseModulePass::write(
    Operator* op,
    const std::shared_ptr<torch::jit::Graph>& graph,
    const torch::jit::Module&) const {
  write(op, graph);
}

static std::vector<const FuseModulePass*> g_global_pnnx_fuse_module_passes;

const std::vector<const FuseModulePass*>& get_global_pnnx_fuse_module_passes() {
  return g_global_pnnx_fuse_module_passes;
}

FuseModulePassRegister::FuseModulePassRegister(const FuseModulePass* pass) {
  g_global_pnnx_fuse_module_passes.push_back(pass);
}

void pass_level1(
    const torch::jit::Module& mod,
    const std::shared_ptr<torch::jit::Graph>& g,
    Graph& pg) {
  for (int i = 1; i < (int)g->inputs().size(); i++) {
    const auto& in = g->inputs()[i];

    char name[32];
    sprintf(name, "pnnx_input_%d", i - 1);

    Operator* op = pg.new_operator("pnnx.Input", name);
    Operand* r = pg.new_operand(in);
    r->producer = op;
    op->outputs.push_back(r);
  }

  std::map<std::string, std::string> class_type_to_names;
  int pnnx_unknown_index = 0;

  for (const auto& n : g->block()->nodes()) {
    if (n->kind() == c10::prim::GetAttr) {
      // pass
      std::string name = n->s(torch::jit::attr::name);

      auto class_type = n->output(0)->type()->cast<torch::jit::ClassType>();

      if (class_type) {
        std::string class_type_str = class_type->str();
        class_type_to_names[class_type_str] = name;
      } else {
        Operator* op = pg.new_operator("pnnx.Attribute", name);

        for (int i = 0; i < (int)n->outputs().size(); i++) {
          const auto& on = n->output(i);
          Operand* r = pg.new_operand(on);
          r->producer = op;
          op->outputs.push_back(r);
        }

        std::deque<std::string> module_names;
        {
          auto np = n->input(0)->node();
          while (np->hasAttribute(torch::jit::attr::name)) {
            module_names.push_front(np->s(torch::jit::attr::name));
            np = np->input(0)->node();
          }
        }

        std::string wrapped_name;
        auto sub_mod = mod;
        for (auto module_name : module_names) {
          if (wrapped_name.size() > 0)
            wrapped_name = wrapped_name + "." + module_name;
          else
            wrapped_name = module_name;
          sub_mod = sub_mod.attr(module_name).toModule();
        }

        op->name = wrapped_name;

        op->attrs[name] = sub_mod.attr(name).toTensor();
      }

      continue;
    } else if (n->kind() == c10::prim::Constant) {
      char name[32];
      sprintf(name, "pnnx_%d", pnnx_unknown_index++);

      Operator* op = pg.new_operator(n->kind().toDisplayString(), name);

      for (int i = 0; i < (int)n->inputs().size(); i++) {
        const auto& in = n->input(i);
        Operand* r = pg.get_operand(in->debugName());
        r->consumers.push_back(op);
        op->inputs.push_back(r);
      }

      for (int i = 0; i < (int)n->outputs().size(); i++) {
        const auto& on = n->output(i);
        Operand* r = pg.new_operand(on);
        r->producer = op;
        op->outputs.push_back(r);
      }

      op->params["value"] = n;

      if (op->params["value"].type == 8) {
        op->type = "pnnx.Attribute";

        op->params.erase("value");

        op->attrs[name] = n->t(torch::jit::attr::value);
      }

      continue;
    }

    switch (n->kind()) {
      case c10::prim::CallMethod: {
        auto class_type = n->input(0)->type()->cast<torch::jit::ClassType>();

        std::string name = class_type_to_names[class_type->str()];

        std::string class_type_str = torch::jit::removeTorchMangle(class_type->str());

        std::string optypename = class_type_str;
        for (const auto& ow : get_global_pnnx_fuse_module_passes()) {
          if (class_type_str != ow->match_type_str())
            continue;

          optypename = ow->type_str();
          break;
        }

        if (optypename == class_type_str) {
          optypename = class_type_str.substr(10);
        }

        Operator* op = pg.new_operator(optypename, name);

        for (int i = 1; i < (int)n->inputs().size(); i++) {
          const auto& in = n->input(i);
          Operand* r = pg.get_operand(in->debugName());
          r->consumers.push_back(op);
          op->inputs.push_back(r);
        }

        for (int i = 0; i < (int)n->outputs().size(); i++) {
          const auto& on = n->output(i);
          Operand* r = pg.new_operand(on);
          r->producer = op;
          op->outputs.push_back(r);
        }

        for (const auto& ow : get_global_pnnx_fuse_module_passes()) {
          if (class_type_str != ow->match_type_str())
            continue;

          auto class_type = n->input(0)->type()->cast<torch::jit::ClassType>();
          torch::jit::Function& function = class_type->getMethod(n->s(torch::jit::attr::name));

          std::deque<std::string> module_names;
          {
            auto np = n->input(0)->node();
            while (np->hasAttribute(torch::jit::attr::name)) {
              module_names.push_front(np->s(torch::jit::attr::name));
              np = np->input(0)->node();
            }
          }

          std::string wrapped_name;
          auto sub_mod = mod;
          for (auto module_name : module_names) {
            if (wrapped_name.size() > 0)
              wrapped_name = wrapped_name + "." + module_name;
            else
              wrapped_name = module_name;
            sub_mod = sub_mod.attr(module_name).toModule();
          }

          op->name = wrapped_name;

          ow->write(op, function.graph(), sub_mod);

          break;
        }

        break;
      }
      default: {
        char name[32];
        sprintf(name, "pnnx_%d", pnnx_unknown_index++);

        Operator* op = pg.new_operator(n->kind().toDisplayString(), name);

        for (int i = 0; i < (int)n->inputs().size(); i++) {
          const auto& in = n->input(i);
          Operand* r = pg.get_operand(in->debugName());
          r->consumers.push_back(op);
          op->inputs.push_back(r);
        }

        for (int i = 0; i < (int)n->outputs().size(); i++) {
          const auto& on = n->output(i);
          Operand* r = pg.new_operand(on);
          r->producer = op;
          op->outputs.push_back(r);
        }

        break;
      }
    }
  }

  for (int i = 0; i < (int)g->outputs().size(); i++) {
    const auto& in = g->outputs()[i];

    char name[32];
    sprintf(name, "pnnx_output_%d", i);
    Operator* op = pg.new_operator("pnnx.Output", name);
    Operand* r = pg.get_operand(in->debugName());
    r->consumers.push_back(op);
    op->inputs.push_back(r);
  }
}

} // namespace pnnx
