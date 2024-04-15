#include <NF/Archetypes/distribution.h>
#include <NF/Distributions/base.h>

using torch::Tensor;

GaussianImpl::GaussianImpl(int n_dim, bool trainable,
                           torch::TensorOptions option)
    : Distribution(n_dim) {
  if (trainable) {
    mean = register_parameter("mean", torch::zeros(n_dim, option));
    std = register_parameter("std", torch::ones_like(mean));
  } else {
    mean = register_buffer("mean", torch::zeros(n_dim, option));
    std = register_buffer("std", torch::ones_like(mean));
  }
}

Tensor GaussianImpl::sample(uint num_sample) {
  auto samples = torch::randn({num_sample, get_dim()}, mean.options());
  return samples * std + mean;
}

Tensor GaussianImpl::log_prob(const Tensor &z) {
  return -get_dim() * 0.9189385332046727 -
         (std.log() + 0.5 * ((z - mean) / std).pow(2)).sum(1);
}
