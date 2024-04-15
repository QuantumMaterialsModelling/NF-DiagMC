#pragma once

#include <NF/Archetypes/distribution.h>

//***********************//
//---DIAGONAL GAUSSIAN---//
//***********************//

class GaussianImpl : public Distribution {
public:
  GaussianImpl(int n_dim, bool trainable = false,
               torch::TensorOptions option = torch::kFloat64);

  torch::Tensor sample(uint num_sample = 1) override;
  torch::Tensor log_prob(const torch::Tensor &z) override;

private:
  torch::Tensor mean, std;
};

TORCH_MODULE(Gaussian);
