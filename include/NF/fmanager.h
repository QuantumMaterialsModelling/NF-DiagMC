#pragma once

#include <torch/nn/module.h>
#include <type_traits>
#include <vector>

#include <NF/Archetypes/distribution.h>
#include <NF/Archetypes/flow.h>

//****************//
//---DEFINITION---//
//****************//

namespace LLNF {

struct FlowManagerImpl : public torch::nn::Module {
public:
  template <typename BASE, typename TARGET>
  FlowManagerImpl(BASE base, std::vector<std::shared_ptr<Flow>> flows,
                  TARGET target);
  template <typename BASE>
  FlowManagerImpl(BASE base, std::vector<std::shared_ptr<Flow>> flows);

  torch::Tensor forward(torch::Tensor z);
  torch::Tensor sample(int num_sample = 1);
  torch::Tensor log_prob(torch::Tensor z);
  torch::Tensor inverse_KL(int num_sample);
  torch::Tensor forward_KL(const torch::Tensor &z);
  torch::Tensor cond_sample(const torch::Tensor &cond);

  std::tuple<torch::Tensor, torch::Tensor> sample_log_prob(int num_sample = 1);
  std::tuple<torch::Tensor, torch::Tensor>
  cond_sample_log_prob(const torch::Tensor &cond);

private:
  template <typename CHILD, typename PARENT>
  std::shared_ptr<PARENT> make_parent(CHILD kid);

private:
  std::shared_ptr<Distribution> base_, target_;
  std::vector<std::shared_ptr<Flow>> flows_;
};

TORCH_MODULE(FlowManager);

//*****************************//
//---TEMPLATE IMPLEMENTATION---//
//*****************************//
template <typename CHILD, typename PARENT>
std::shared_ptr<PARENT> FlowManagerImpl::make_parent(CHILD kid) {
  if constexpr (std::is_base_of_v<Distribution, CHILD>)
    return std::shared_ptr<Distribution>(new CHILD(kid));
  else
    return std::shared_ptr<Distribution>(new
                                         typename CHILD::ContainedType(*kid));
}

template <typename BASE, typename TARGET>
FlowManagerImpl::FlowManagerImpl(BASE base,
                                 std::vector<std::shared_ptr<Flow>> flows,
                                 TARGET target) {
  base_ = make_parent<BASE, Distribution>(base);
  target_ = make_parent<TARGET, Distribution>(target);

  register_module("base", base_);
  register_module("target", target_);

  for (int i = 0; i != (int)flows.size(); i++) {
    auto name = std::string("flow") + std::to_string(i);
    register_module(name, flows[i]);

    flows_.push_back(flows[i]);
  }
}

template <typename BASE>
FlowManagerImpl::FlowManagerImpl(BASE base,
                                 std::vector<std::shared_ptr<Flow>> flows) {
  base_ = make_parent<BASE, Distribution>(base);
  register_module("base", base_);

  for (int i = 0; i != (int)flows.size(); i++) {
    auto name = std::string("flow") + std::to_string(i);
    register_module(name, flows[i]);

    flows_.push_back(flows[i]);
  }
}

} // namespace LLNF
