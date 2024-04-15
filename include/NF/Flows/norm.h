#pragma once

#include <NF/Archetypes/flow.h>
#include <torch/nn/pimpl.h>

//*************************//
//---BATCH NORMALIZATION---//
//*************************//

class BatchNormImpl : public Flow {
public:
  BatchNormImpl(int n_dim, double eps = 1E-5,
                torch::TensorOptions options = torch::kF64);

  std::tuple<torch::Tensor, torch::Tensor>
  forward(const torch::Tensor &z) override;
  std::tuple<torch::Tensor, torch::Tensor>
  inverse(const torch::Tensor &z) override;

private:
  double eps_;
  torch::Tensor Uweight_, bias_;
  torch::Tensor mean_, var_;
};

TORCH_MODULE(BatchNorm);
