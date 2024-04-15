#include "../Model.h"
#include <ATen/TensorIndexing.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>
#include <NF/Flows/Std.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <cstdio>
#include <math.h>
#include <torch/nn/functional/activation.h>

using namespace torch;
using namespace MLHol;

#define F torch::nn::functional

/*******
 *  The vectors here will be represented as
 *
 *  [tau, tau_1, tau_2, ...]
 * */

BaseImpl::BaseImpl(int max_order, torch::TensorOptions options)
    : Distribution(max_order) {
  mean = register_buffer("mean", torch::full({max_order + 1}, 10, options));
}

std::tuple<torch::Tensor, torch::Tensor> BaseImpl::forward(int num_sample) {
  auto tau = torch::linspace(0.05, 20, num_sample / 10, mean.options())
                 .repeat(10)
                 .unsqueeze(-1);
  // auto tau     = 10 * torch::ones({num_sample, 1}, mean.options());
  auto samples = torch::randn({num_sample, get_dim()}, mean.options()) + 2.;

  auto res = torch::cat({tau, samples}, 1);
  auto log_prob =
      -get_dim() * 0.9189385332046727 - (0.5 * (samples - 2.).pow(2)).sum(1);

  return {res, log_prob};
}

Tensor BaseImpl::sample(uint num_sample) {
  auto tau = 20. * torch::rand({num_sample, 1}, mean.options());
  // auto tau     = 10 * torch::ones({num_sample, 1}, mean.options());
  auto samples = torch::randn({num_sample, get_dim()}, mean.options()) + 2.;

  return torch::cat({tau, samples}, 1);
}

Tensor BaseImpl::log_prob(const Tensor &z) {
  auto tau = z.slice(1, 0, 1);
  return -get_dim() * 0.9189385332046727 -
         (0.5 * (z.slice(1, 1) - 2.).pow(2)).sum(1);
}

Tensor BaseImpl::cond_sample(const Tensor &cond) {
  auto samples = torch::randn({cond.size(0), get_dim()}, cond.options()) + 2.;

  return torch::cat({cond, samples}, 1);
}

TargetImpl::TargetImpl(int max_order, double c)
    : TargetDistribution(max_order, torch::kF64), c(c) {}

Tensor TargetImpl::log_prob(const torch::Tensor &z) {
  auto t = z.slice(1, 0, 1);
  auto ti = z.slice(1, 1);
  auto res = torch::zeros_like(ti);

  // Esponentials as in the on site model
  res += ti.slice(1, 0, indexing::None, 2);
  res -= ti.slice(1, 1, indexing::None, 2);

  // Using a plateu function to select stuff
  res -= F::softplus(c * (ti.slice(1, -1) - t)); // Makes so t_i < t

  auto tr = ti.roll(-1, 1);
  tr.slice(1, 0, 1) = 0;
  res -= F::softplus(-c * (ti - tr)); // Makes so t_i > t_{i-1}

  return res.sum(1);
}

std::vector<std::shared_ptr<Flow>> MLHol::create_flow_list(Option opt) {
  std::vector<std::shared_ptr<Flow>> flows;

  for (int j = 0; j != opt.NAffine; j++) {
    flows.push_back(std::shared_ptr<Flow>(
        new ARQSImpl(opt.MOrder, opt.n_bin, opt.bound, opt.Hidden, 1)));
    // flows.push_back(std::shared_ptr<Flow>(new AAffImpl(opt.MOrder,
    // opt.Hidden, 1)));
  }

  return flows;
}

void MLHol::print_vector(const std::vector<double> &vx, const char *path,
                         bool over) {
  FILE *file;

  if (over) {
    file = fopen(path, "a");
  } else {
    file = fopen(path, "w");
  }

  if (file == NULL) {
    printf("Unable to open file!\n");
    exit(1);
  }

  for (auto &x : vx)
    fprintf(file, "%f\n", x);

  fclose(file);
}

std::vector<double> MLHol::read_vector(const char *path) {
  FILE *file = fopen(path, "r");

  if (file == NULL) {
    printf("Unable to open file!\n");
    exit(1);
  }

  std::vector<double> vx;
  double temp;

  while (fscanf(file, "%lf", &temp) == 1)
    vx.push_back(temp);

  if (feof(file))
    printf("Vector readed correctly!\n");
  else {
    printf("Problem reading the vector!\n");
    exit(1);
  }
  fclose(file);

  return vx;
}
