#pragma once

#include <torch/nn/functional/linear.h>
#include <torch/nn/module.h>

class MLinearImpl : public torch::nn::Module {
public:
  MLinearImpl(torch::Tensor mask, bool bias = true);

	inline torch::Tensor forward(const torch::Tensor &z){
		return torch::nn::functional::linear(z, W * mask_, b);
	}

private:
  torch::Tensor W, b;
  torch::Tensor mask_;
};

TORCH_MODULE(MLinear);
