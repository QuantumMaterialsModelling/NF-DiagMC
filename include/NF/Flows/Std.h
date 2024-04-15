#pragma once

#include <NF/Archetypes/flow.h>
#include <torch/nn/modules/container/sequential.h>

////////////////////////
//---AUTOREGRESSIVE---//
////////////////////////

class MaskedConditionerImpl : public torch::nn::Module {
public:
  MaskedConditionerImpl() = default;
  MaskedConditionerImpl(int in_dim, int n_par, int hidden = 1, int cond = 0,
                        bool init_zero = true);

  inline torch::Tensor forward(const torch::Tensor &z) {
    return net->forward(z);
  }

private:
  torch::nn::Sequential net;

}; // MaskedConditionerImpl

TORCH_MODULE(MaskedConditioner);

//-----------------------------

class ARQSImpl : public Flow {
public:
  ARQSImpl(int ninp, int nbin = 5, double bound = 1., int hidden = 50,
           int ncond = 0);

  std::tuple<torch::Tensor, torch::Tensor>
  forward(const torch::Tensor &z) override;
  std::tuple<torch::Tensor, torch::Tensor>
  inverse(const torch::Tensor &z) override;

private:
  MaskedConditioner cond;
  int n_bin, n_inp, n_cond;
  double bound;

}; // ARQSImpl

TORCH_MODULE(ARQS);

//-----------------------------

class AAffImpl : public Flow {
public:
  AAffImpl(int ninp, int hidden = 50, int ncond = 0);

  std::tuple<torch::Tensor, torch::Tensor>
  forward(const torch::Tensor &z) override;
  std::tuple<torch::Tensor, torch::Tensor>
  inverse(const torch::Tensor &z) override;

private:
  MaskedConditioner cond;
  int n_inp, n_cond;

}; // ARQSImpl

TORCH_MODULE(AAff);
