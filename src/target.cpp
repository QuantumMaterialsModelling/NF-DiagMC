#include <NF/Distributions/target.h>
#include <cmath>

using namespace torch;

//*************************//
//---TARGET DISTRIBUTION---//
//*************************//

TargetDistribution::TargetDistribution(int n_dim, torch::TensorOptions options)
    : Distribution(n_dim) {
  scale_ = register_buffer("scale", torch::ones(n_dim, options));
  shift_ = register_buffer("shift", torch::zeros_like(scale_));
}

TargetDistribution::TargetDistribution(int n_dim, torch::Tensor scale,
                                       torch::Tensor shift)
    : Distribution(n_dim) {
  scale_ = register_buffer("scale", scale);
  shift_ = register_buffer("shift", shift);
}

torch::Tensor TargetDistribution::sample(uint num_sample) {
  torch::Tensor res;
  int tot = 0;

  while (tot < num_sample) {
    torch::Tensor samples =
        torch::rand({num_sample, get_dim()}, scale_.options()) * scale_ +
        shift_;
    torch::Tensor prob = log_prob(samples).exp();

    torch::Tensor acc = torch::rand_like(prob) < prob;

    res = torch::cat({res, samples[acc]}, 1);
    tot += acc.to(torch::kI32).sum().item<int>();
  }

  return res.slice(0, 0, num_sample);
}

//***************************//
//---EXPONENTIAL EXPANSION---//
//***************************//

ExpExpansionImpl::ExpExpansionImpl(double E, double V)
    : E_V_l_(std::log(E - V)), E_(E), V_(V), TargetDistribution(2) {}

Tensor ExpExpansionImpl::log_prob(const torch::Tensor &z) {
  auto order = z.slice(1, 0, 1).floor();
  auto tm_fl = z.slice(1, 1);

  return E_V_l_ + order * (V_ * tm_fl).log() - E_ * tm_fl -
         (order + 1).lgamma();
}

//*********************//
//---ON SITE POLARON---//
//*********************//

OnSitePolaronImpl::OnSitePolaronImpl(int max_order, double G, double O)
    : G_(G), O_(O), TargetDistribution(max_order + 2) {}

Tensor OnSitePolaronImpl::log_prob(const torch::Tensor &z) {
  return torch::zeros_like(z.slice(1, 1));
}

Tensor OnSitePolaronImpl::sample(uint num_sample) {
  return torch::ones({num_sample, n_dim});
}
