#pragma once

#include <torch/nn/module.h>
#include <torch/nn/modules/container/sequential.h>

class MLPImpl : public torch::nn::Module {
public:
  MLPImpl() = default;
  MLPImpl(std::vector<int> sizes, bool init_zero = false);

	inline torch::Tensor forward(const torch::Tensor &z){
		return net->forward(z);
	}

private:
	torch::nn::Sequential net;
};


TORCH_MODULE(MLP);
