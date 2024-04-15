#pragma once

#include "Diagram.cpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace Holstein {

class chg_t : public DMC::Update<Diagram> {
public:
  chg_t() : Update("chg_t") {}

  double atempt() override { return 1.1; }

  void accept() override {
    double max = dia->n == 0 ? 0
                             : *std::max_element(dia->Pe.begin(),
                                                 dia->Pe.begin() + dia->n);

    dia->t =
        max - std::log(std::uniform_real_distribution<double>()(rng)) / dia->e;
  }
};

class add_n : public DMC::Update<Diagram> {
public:
  add_n() : Update("add_n", "rem_n") {}

  double atempt() override {
    new_b = std::uniform_real_distribution<double>(0, dia->t)(rng);
    new_e = std::uniform_real_distribution<double>(0, dia->t)(rng);

    if (new_e < new_b)
      std::swap(new_e, new_b);

    return 0.5 * std::pow(dia->g * dia->t, 2) *
           std::exp(-dia->o * (new_e - new_b)) / (dia->n + 1);
  }

  void accept() override { dia->add_p(new_b, new_e); }

private:
  double new_b, new_e;
};

class rem_n : public DMC::Update<Diagram> {
public:
  rem_n() : Update("rem_n", "add_n") {}

  double atempt() override {
    if (dia->n == 0)
      return 0;

    which = rng() % dia->n;

    return 2. * std::pow(dia->g * dia->t, -2) *
           std::exp(dia->o * (dia->Pe[which] - dia->Pb[which])) * dia->n;
  }

  void accept() override { dia->rem_p(which); }

private:
  int which;
};

class nnchg_n : public DMC::Update<Diagram> {
public:
  nnchg_n() : Update("nnchg_n") {}

  double atempt() override {
    double mean = dia->t - 1 + std::exp(-dia->t);
    int new_n = std::poisson_distribution<int>(dia->g * dia->g * mean)(rng);

    // Check for detailed balance with local updates
    if (std::abs(dia->n - new_n) == 1)
      return 0;

    // If both are zero is always accepted
    if (new_n == 0 && dia->n == 0) {
      new_P = torch::empty(0, torch::kF64);
      return 1;
    }

    // Sampling a new diagram of order new_n with same time of flight
    auto cond = torch::full({new_n, 1}, dia->t, torch::kF64);
    auto [cnew_P, new_prob] = dia->NNsample->cond_sample_log_prob(cond);
    new_P = cnew_P.clone();

    // CHeck for invalid generation (hopefully small)
    if (*(new_P.slice(1, 1) > dia->t).any().to(torch::kCPU).data_ptr<bool>() ||
        *(new_P.slice(1, 1) < 0).any().to(torch::kCPU).data_ptr<bool>() ||
        *(new_P.slice(1, 1, 2) > new_P.slice(1, 2))
             .any()
             .to(torch::kCPU)
             .data_ptr<bool>())
      return -1;

    double prob =
        *(dia->NNsample->log_prob(dia->get_tensor()).sum() - new_prob.sum())
             .exp()
             .to(torch::kCPU)
             .data_ptr<double>();
    double new_w =
        -*new_P.slice(1, 1).diff(1).sum().to(torch::kCPU).data_ptr<double>();

    for (int i = 0; i != dia->n; i++)
      new_w += dia->Pe[i] - dia->Pb[i];

    return std::pow(mean, dia->n - new_n) * prob * std::exp(new_w);
  }

  void accept() override { dia->set_tensor(new_P); }

private:
  torch::Tensor new_P;
};

} // namespace Holstein
