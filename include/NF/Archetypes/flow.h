#pragma once

#include <torch/nn/module.h>
#include <tuple>

struct Flow : public torch::nn::Module {
	virtual std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& z) = 0;
	virtual std::tuple<torch::Tensor, torch::Tensor> inverse(const torch::Tensor& z) = 0;
};
