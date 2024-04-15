#include <NF/Flows/norm.h>
#include <cstdio>
#include <torch/nn/functional/activation.h>

#define F torch::nn::functional
using torch::Tensor;

//******************************//
//---BATCH NORMALIZATION IMPL---//
//******************************//

BatchNormImpl::BatchNormImpl(int n_dim, double eps,
                             torch::TensorOptions options)
    : eps_(eps) {
  Uweight_ = register_parameter("Uweight", torch::zeros(n_dim, options));
  bias_ = register_parameter("bias", torch::zeros_like(Uweight_));

  mean_ = register_buffer("mean", torch::empty(n_dim, options));
  var_ = register_buffer("var", torch::empty(n_dim, options));
}

std::tuple<Tensor, Tensor> BatchNormImpl::forward(const Tensor &z) {
  // if(is_training())
  mean_ = z.mean(0), var_ = z.var(0);

  // printf("Dove sei?");
  auto weight = F::softplus(Uweight_) + eps_;

  auto res = bias_ + weight * (z - mean_) / (var_ + eps_).sqrt();
  auto log_det = weight.log() - 0.5 * (var_ + eps_).log();

  // printf("Sono qui!");
  return {res, torch::ones(z.size(0), z.options()) * log_det.sum()};
}

std::tuple<Tensor, Tensor> BatchNormImpl::inverse(const Tensor &z) {
  auto weight = F::softplus(Uweight_) + eps_;

  auto res = mean_ + (z - bias_) * (var_ + eps_).sqrt() / weight;
  auto log_det = -weight.log() + 0.5 * (var_ + eps_).log();

  return {res, torch::ones(z.size(0), z.options()) * log_det.sum()};
}
