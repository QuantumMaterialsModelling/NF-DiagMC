#include <ATen/core/interned_strings.h>
#include <ATen/ops/ones_like.h>
#include <NF/fmanager.h>
#include <iostream>

using namespace torch;
using LLNF::FlowManagerImpl;

Tensor FlowManagerImpl::forward(torch::Tensor z) {
  for (auto &flow : flows_) {
    auto &&result = flow->forward(z);
    z = std::get<0>(result);
  }

  return z;
}

Tensor FlowManagerImpl::sample(int num_sample) {
  return forward(base_->sample(num_sample));
}

Tensor FlowManagerImpl::log_prob(torch::Tensor z) {
  Tensor log_prob = torch::zeros(z.size(0), z.options());

  for (int i = flows_.size() - 1; i != -1; i--) {
    auto [x, log_det] = flows_[i]->inverse(z);

    log_prob += log_det, z = x;
  }

  return log_prob + base_->log_prob(z);
}

std::tuple<torch::Tensor, torch::Tensor>
FlowManagerImpl::sample_log_prob(int num_sample) {
  auto [z, log_prob] = base_->forward(num_sample);

  for (auto &flow : flows_) {
    auto [x, log_det] = flow->forward(z);

    log_prob -= log_det, z = x;
  }

  return {z, log_prob};
}

std::tuple<torch::Tensor, torch::Tensor>
FlowManagerImpl::cond_sample_log_prob(const torch::Tensor &cond) {
  auto z = base_->cond_sample(cond);
  auto log_prob = base_->log_prob(z);

  for (auto &flow : flows_) {
    auto [x, log_det] = flow->forward(z);

    log_prob -= log_det, z = x;
  }

  return {z, log_prob};
}

Tensor FlowManagerImpl::inverse_KL(int num_sample) {
  auto [z, mlog_prob] = sample_log_prob(num_sample);
  return mlog_prob.mean() - target_->log_prob(z).mean();
}

Tensor FlowManagerImpl::forward_KL(const Tensor &z) {
  return -log_prob(z).mean();
}

Tensor FlowManagerImpl::cond_sample(const Tensor &cond) {
  auto sample = base_->cond_sample(cond);
  return forward(sample);
}
