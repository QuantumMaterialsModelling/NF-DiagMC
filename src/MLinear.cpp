#include <ATen/ops/zeros_like.h>
#include <NF/Nets/MLinear.h>
#include <cmath>

using namespace torch;

MLinearImpl::MLinearImpl(Tensor mask, bool bias) {
  double k = 1;

  W = register_parameter("W", k * torch::rand_like(mask) - 0.5 * k);

  if (bias)
    b = register_parameter(
        "b",
        k * torch::rand_like(mask.slice(1, 0, 1)).transpose(0, 1) - 0.5 * k);
  else
    b = register_buffer("b",
                        torch::zeros_like(mask.slice(1, 0, 1)).transpose(0, 1));

  mask_ = register_buffer("mask", mask);
}
