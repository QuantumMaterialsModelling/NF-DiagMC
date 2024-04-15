#include <NF/Flows/Std.h>
#include <NF/Nets/MLinear.h>

#include <ATen/TensorIndexing.h>
#include <NF/Utils/RQS.h>
#include <cstdio>
#include <torch/torch.h>

using namespace torch;
using torch::indexing::Slice;

// Helper function to create the mask
torch::Tensor autoregressive_mask(int in_dim, int out_dim, int cond = 1) {
  auto mask = torch::zeros({out_dim, in_dim});
  int stride = out_dim > in_dim ? out_dim / in_dim : cond;

  for (int i = 0; i < in_dim; i++)
    if (out_dim > in_dim)
      mask.index_put_({Slice(i * stride, (i + 1) * stride),
                       Slice(indexing::None, i + cond)},
                      1);
    else
      mask.index_put_({Slice(i * stride, (i + 1) * stride),
                       Slice(i * in_dim / (out_dim / cond),
                             (i + 1) * in_dim / (out_dim / cond))},
                      1);

  return mask;
}

MaskedConditionerImpl::MaskedConditionerImpl(int in_dim, int n_par, int hidden,
                                             int cond, bool init_zero) {
  net->push_back(MLinear(autoregressive_mask(in_dim, in_dim * hidden, cond)));
  net->push_back(torch::nn::ReLU());
  net->push_back(
      MLinear(autoregressive_mask(in_dim * hidden, in_dim * n_par, n_par)));

  if (init_zero) {
    torch::nn::init::zeros_(*(net->parameters().end() - 2));
    torch::nn::init::zeros_(*(net->parameters().end() - 1));
  }
  net = register_module("net", net);
}

ARQSImpl::ARQSImpl(int ninp, int nbin, double bound, int hidden, int ncond)
    : n_bin(nbin), n_inp(ninp), n_cond(ncond), bound(bound) {
  cond = register_module("cond", MaskedConditioner(n_inp + ncond - 1, 3 * n_bin,
                                                   hidden, ncond, true));

  cond->to(kF64);
}

std::tuple<Tensor, Tensor> ARQSImpl::forward(const torch::Tensor &z) {
  auto h = cond->forward(z.clone().slice(1, 0, -1))
               .reshape({z.size(0), 3 * n_inp, n_bin});

  auto unconst_width = h.slice(1, 0, indexing::None, 3);
  auto unconst_height = h.slice(1, 1, indexing::None, 3);
  auto unconst_deriva = h.slice(1, 2, indexing::None, 3);

  auto [res, log_det] = RQS(z.slice(1, n_cond), unconst_width, unconst_height,
                            unconst_deriva, bound, false);

  return {torch::cat({z.slice(1, 0, n_cond), res}, 1), log_det.sum(1)};
}

std::tuple<Tensor, Tensor> ARQSImpl::inverse(const torch::Tensor &z) {
  auto res = z.clone();

  for (int i = n_cond; i != z.size(1); i++) {
    auto h = cond->forward(res.clone().slice(1, 0, -1))
                 .reshape({z.size(0), 3 * n_inp, n_bin});

    auto unconst_width = h.slice(1, 0, indexing::None, 3);
    auto unconst_height = h.slice(1, 1, indexing::None, 3);
    auto unconst_deriva = h.slice(1, 2, indexing::None, 3);

    auto [qrs, qrs_det] = RQS(z.slice(1, 1), unconst_width, unconst_height,
                              unconst_deriva, bound, true);

    res.slice(1, i, i + 1) = qrs.slice(1, i - 1, i);
  }
  auto h =
      cond->forward(res.slice(1, 0, -1)).reshape({z.size(0), 3 * n_inp, n_bin});

  auto unconst_width = h.slice(1, 0, indexing::None, 3);
  auto unconst_height = h.slice(1, 1, indexing::None, 3);
  auto unconst_deriva = h.slice(1, 2, indexing::None, 3);

  auto [qrs, qrs_det] = RQS(z.slice(1, n_cond), unconst_width, unconst_height,
                            unconst_deriva, bound, true);

  return {res, qrs_det.sum(1)};
}

AAffImpl::AAffImpl(int ninp, int hidden, int ncond)
    : n_inp(ninp), n_cond(ncond) {
  cond = register_module(
      "cond", MaskedConditioner(n_inp + ncond - 1, 2, hidden, ncond, true));

  cond->to(kF64);
}

std::tuple<Tensor, Tensor> AAffImpl::forward(const torch::Tensor &z) {
  auto h = cond->forward(z.clone().slice(1, 0, -1));

  auto a = h.slice(1, 0, indexing::None, 2);
  auto b = h.slice(1, 1, indexing::None, 2);
  auto res = z.slice(1, 1) * a.exp() + b;

  return {torch::cat({z.slice(1, 0, n_cond), res}, 1), a.sum(1)};
}

std::tuple<Tensor, Tensor> AAffImpl::inverse(const torch::Tensor &z) {
  auto res = z.clone();

  for (int i = n_cond; i != z.size(1); i++) {
    auto h = cond->forward(res.clone().slice(1, 0, -1));

    auto a = h.slice(1, 0, indexing::None, 2);
    auto b = h.slice(1, 1, indexing::None, 2);
    auto t = (res.slice(1, 1) - b) * (-a).exp();

    res.slice(1, i, i + 1) = t.slice(1, i - 1, i);
  }
  auto h = cond->forward(z.clone().slice(1, 0, -1));

  auto a = h.slice(1, 0, indexing::None, 2);
  auto b = h.slice(1, 1, indexing::None, 2);

  return {res, -a.sum(1)};
}
