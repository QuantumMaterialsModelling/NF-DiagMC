#pragma once

#include <NF/Archetypes/distribution.h>
#include <NF/Archetypes/flow.h>
#include <NF/Nets/MLP.h>

#include "Option.h"

namespace MLHol {

//---DISTIBUTIONS---//

class BaseImpl : public Distribution {
public:
  BaseImpl(int max_order, torch::TensorOptions options = torch::kF64);

  std::tuple<torch::Tensor, torch::Tensor> forward(int num_sample = 1) override;
  torch::Tensor sample(uint num_sample = 1) override;
  torch::Tensor log_prob(const torch::Tensor &z) override;
  torch::Tensor cond_sample(const torch::Tensor &cond) override;

private:
  torch::Tensor mean;
};

TORCH_MODULE(Base);

class TargetImpl : public TargetDistribution {
public:
  TargetImpl(int max_order, double c = 1E2);

  torch::Tensor log_prob(const torch::Tensor &z) override;

private:
  double c;
};

TORCH_MODULE(Target);

//---MODEL---//

std::vector<std::shared_ptr<Flow>> create_flow_list(Option opt);
void print_vector(const std::vector<double> &vx, const char *path,
                  bool over = false);
std::vector<double> read_vector(const char *path);

} // namespace MLHol
