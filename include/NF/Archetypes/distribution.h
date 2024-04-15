#pragma once

#include <ATen/ops/zeros_like.h>
#include <torch/nn/module.h>
#include <tuple>

class Distribution : public torch::nn::Module {
public:
  Distribution(int n_dim = 1): n_dim(n_dim){}

	virtual std::tuple<torch::Tensor, torch::Tensor> forward(int num_sample = 1){
		torch::Tensor samples = sample(num_sample);
		return {samples, log_prob(samples)};
	}

	virtual torch::Tensor sample(uint num_sample = 1) = 0;
	virtual torch::Tensor log_prob(const torch::Tensor& z) = 0;
  virtual torch::Tensor cond_sample(const torch::Tensor &cond) {
    printf("Conditional sampling not implemented!\n");
    exit(1);

    return torch::zeros_like(cond);
  }

  const int & get_dim() const{return n_dim;}

protected:
  int n_dim;
};


class TargetDistribution : public Distribution {
public:
  TargetDistribution(int n_dim, torch::TensorOptions options = torch::kF64);
  TargetDistribution(int n_dim, torch::Tensor scale, torch::Tensor shift);

  torch::Tensor sample(uint num_sample = 1) override;
  virtual torch::Tensor log_prob(const torch::Tensor &z) override = 0;
  
private:
  torch::Tensor scale_, shift_;
};
