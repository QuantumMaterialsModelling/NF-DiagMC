#include <NF/Nets/MLP.h>
#include <torch/nn/init.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/linear.h>

MLPImpl::MLPImpl(std::vector<int> sizes, bool init_zero) {
  for (uint i = 0; i != sizes.size() - 2; i++) {
    net->push_back(torch::nn::Linear(sizes[i], sizes[i + 1]));
    net->push_back(torch::nn::ReLU());
  }
  net->push_back(
      torch::nn::Linear(sizes[sizes.size() - 2], sizes[sizes.size() - 1]));

  if (init_zero) {
    torch::nn::init::zeros_(*(net->parameters().end() - 2));
    torch::nn::init::zeros_(*(net->parameters().end() - 1));
  }
  net = register_module("net", net);
}
