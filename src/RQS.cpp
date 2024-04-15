#include <ATen/TensorIndexing.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros_like.h>
#include <NF/Utils/RQS.h>
#include <algorithm>
#include <cstdio>
#include <torch/nn/options/padding.h>
#include <torch/torch.h>

using namespace torch::indexing;
#define F torch::nn::functional

std::tuple<at::Tensor, at::Tensor>
cosRQS(at::Tensor inputs, at::Tensor unnormalized_widths,
       at::Tensor unnormalized_heights, at::Tensor unnormalized_derivatives,
       at::Tensor left, at::Tensor right, at::Tensor bottom, at::Tensor top,
       bool inverse, double min_bin_width, double min_bin_height,
       double min_derivative) {
  int num_bins = unnormalized_widths.size(-1);

  // Whidths computations
  auto widths = F::softmax(unnormalized_widths, -1);
  widths = min_bin_width + (1 - min_bin_width * num_bins) * widths;

  auto cumwidths = torch::cumsum(widths, -1);
  cumwidths = torch::constant_pad_nd(cumwidths, {1, 0}, 0);
  cumwidths = (right - left) * cumwidths + left;
  widths = cumwidths.index({"...", Slice(1)}) -
           cumwidths.index({"...", Slice(None, -1)});

  // Derivatives computations
  auto derivatives = min_derivative + F::softplus(unnormalized_derivatives);

  // Heights computations
  auto heights = F::softmax(unnormalized_heights, -1);
  heights = min_bin_height + (1 - min_bin_height * num_bins) * heights;

  auto cumheights = torch::cumsum(heights, -1);
  cumheights = torch::constant_pad_nd(cumheights, {1, 0}, 0);
  cumheights = (top - bottom) * cumheights + bottom;
  heights = cumheights.index({"...", Slice(1)}) -
            cumheights.index({"...", Slice(None, -1)});

  // Search the bins of the input
  auto bin_idx = torch::empty(0, inputs.options());
  if (inverse)
    bin_idx = searchsorted(cumheights, inputs.unsqueeze(-1)) - 1;
  else
    bin_idx = searchsorted(cumwidths, inputs.unsqueeze(-1)) - 1;

  // Collect informations about the bin inputs
  auto input_cumwidths = cumwidths.gather(-1, bin_idx).index({"...", 0});
  auto input_bin_widths = widths.gather(-1, bin_idx).index({"...", 0});
  auto input_cumheights = cumheights.gather(-1, bin_idx).index({"...", 0});

  auto delta = heights / widths;
  auto input_delta = delta.gather(-1, bin_idx).index({"...", 0});

  auto input_derivatives = derivatives.gather(-1, bin_idx).index({"...", 0});
  auto input_derivatives_plus_one = derivatives.index({"...", Slice(1)})
                                        .gather(-1, bin_idx)
                                        .index({"...", 0});

  auto input_heights = heights.gather(-1, bin_idx).index({"...", 0});

  if (inverse) {
    auto a = ((
        (inputs - input_cumheights) *
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) +
        input_heights * (input_delta - input_derivatives)));
    auto b = (input_heights * input_derivatives -
              (inputs - input_cumheights) *
                  (input_derivatives + input_derivatives_plus_one -
                   2 * input_delta));
    auto c = -input_delta * (inputs - input_cumheights);

    auto discriminant = b.pow(2) - 4 * a * c;
    // TODO: Se il discriminante <0 errore

    auto root = (2 * c) / (-b - torch::sqrt(discriminant));
    auto outputs = root * input_bin_widths + input_cumwidths;

    auto theta_one_minus_theta = root * (1 - root);
    auto denominator =
        input_delta +
        ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) *
         theta_one_minus_theta);
    auto derivative_numerator =
        input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2) +
                              2 * input_delta * theta_one_minus_theta +
                              input_derivatives * (1 - root).pow(2));
    auto logabsdet =
        torch::log(derivative_numerator) - 2 * torch::log(denominator);
    return {outputs, -logabsdet};
  } else {
    auto theta = (inputs - input_cumwidths) / input_bin_widths;
    auto theta_one_minus_theta = theta * (1 - theta);

    auto numerator =
        input_heights * (input_delta * theta.pow(2) +
                         input_derivatives * theta_one_minus_theta);
    auto denominator =
        input_delta +
        ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) *
         theta_one_minus_theta);
    auto outputs = input_cumheights + numerator / denominator;

    auto derivative_numerator =
        input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2) +
                              2 * input_delta * theta_one_minus_theta +
                              input_derivatives * (1 - theta).pow(2));
    auto logabsdet =
        torch::log(derivative_numerator) - 2 * torch::log(denominator);
    return {outputs, logabsdet};
  }
}

std::tuple<at::Tensor, at::Tensor>
cosRQS(at::Tensor inputs, at::Tensor unnormalized_widths,
       at::Tensor unnormalized_heights, at::Tensor unnormalized_derivatives,
       double left, double right, double bottom, double top, bool inverse,
       double min_bin_width, double min_bin_height, double min_derivative) {
  int num_bins = unnormalized_widths.size(-1);

  // Whidths computations
  auto widths = F::softmax(unnormalized_widths, -1);
  widths = min_bin_width + (1 - min_bin_width * num_bins) * widths;

  auto cumwidths = torch::cumsum(widths, -1);
  cumwidths = torch::constant_pad_nd(cumwidths, {1, 0}, 0);
  cumwidths = (right - left) * cumwidths + left;
  widths = cumwidths.index({"...", Slice(1)}) -
           cumwidths.index({"...", Slice(None, -1)});

  // Derivatives computations
  auto derivatives = min_derivative + F::softplus(unnormalized_derivatives);

  // Heights computations
  auto heights = F::softmax(unnormalized_heights, -1);
  heights = min_bin_height + (1 - min_bin_height * num_bins) * heights;

  auto cumheights = torch::cumsum(heights, -1);
  cumheights = torch::constant_pad_nd(cumheights, {1, 0}, 0);
  cumheights = (top - bottom) * cumheights + bottom;
  heights = cumheights.index({"...", Slice(1)}) -
            cumheights.index({"...", Slice(None, -1)});

  // Search the bins of the input
  auto bin_idx = torch::empty(0, inputs.options());
  if (inverse)
    bin_idx = searchsorted(cumheights, inputs.unsqueeze(-1)) - 1;
  else
    bin_idx = searchsorted(cumwidths, inputs.unsqueeze(-1)) - 1;

  // Collect informations about the bin inputs
  auto input_cumwidths = cumwidths.gather(-1, bin_idx).index({"...", 0});
  auto input_bin_widths = widths.gather(-1, bin_idx).index({"...", 0});
  auto input_cumheights = cumheights.gather(-1, bin_idx).index({"...", 0});

  auto delta = heights / widths;
  auto input_delta = delta.gather(-1, bin_idx).index({"...", 0});

  auto input_derivatives = derivatives.gather(-1, bin_idx).index({"...", 0});
  auto input_derivatives_plus_one = derivatives.index({"...", Slice(1)})
                                        .gather(-1, bin_idx)
                                        .index({"...", 0});

  auto input_heights = heights.gather(-1, bin_idx).index({"...", 0});

  if (inverse) {
    auto a = ((
        (inputs - input_cumheights) *
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) +
        input_heights * (input_delta - input_derivatives)));
    auto b = (input_heights * input_derivatives -
              (inputs - input_cumheights) *
                  (input_derivatives + input_derivatives_plus_one -
                   2 * input_delta));
    auto c = -input_delta * (inputs - input_cumheights);

    auto discriminant = b.pow(2) - 4 * a * c;
    // TODO: Se il discriminante <0 errore

    auto root = (2 * c) / (-b - torch::sqrt(discriminant));
    auto outputs = root * input_bin_widths + input_cumwidths;

    auto theta_one_minus_theta = root * (1 - root);
    auto denominator =
        input_delta +
        ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) *
         theta_one_minus_theta);
    auto derivative_numerator =
        input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2) +
                              2 * input_delta * theta_one_minus_theta +
                              input_derivatives * (1 - root).pow(2));
    auto logabsdet =
        torch::log(derivative_numerator) - 2 * torch::log(denominator);
    return {outputs, -logabsdet};
  } else {
    auto theta = (inputs - input_cumwidths) / input_bin_widths;
    auto theta_one_minus_theta = theta * (1 - theta);

    auto numerator =
        input_heights * (input_delta * theta.pow(2) +
                         input_derivatives * theta_one_minus_theta);
    auto denominator =
        input_delta +
        ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) *
         theta_one_minus_theta);
    auto outputs = input_cumheights + numerator / denominator;

    auto derivative_numerator =
        input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2) +
                              2 * input_delta * theta_one_minus_theta +
                              input_derivatives * (1 - theta).pow(2));
    auto logabsdet =
        torch::log(derivative_numerator) - 2 * torch::log(denominator);
    return {outputs, logabsdet};
  }
}

std::tuple<at::Tensor, at::Tensor>
RQS(at::Tensor inputs, at::Tensor unnormalized_widths,
    at::Tensor unnormalized_heights, at::Tensor unnormalized_derivatives,
    at::Tensor tail, bool inverse, double min_bin_width, double min_bin_height,
    double min_derivative) {

  auto inside_intvl_mask = (inputs >= -tail) & (inputs <= tail);
  auto outside_intvl_mask = ~inside_intvl_mask;

  auto outputs = torch::zeros_like(inputs);
  auto logabsdet = torch::zeros_like(inputs);

  unnormalized_derivatives =
      torch::constant_pad_nd(unnormalized_derivatives, {1, 1},
                             std::log(std::exp(1 - min_derivative) - 1.));

  outputs = outputs.index_put({outside_intvl_mask},
                              inputs.index({outside_intvl_mask}));

  tail = tail.index({inside_intvl_mask}).unsqueeze(-1);

  auto [val, lad] = cosRQS(
      inputs.index({inside_intvl_mask}),
      unnormalized_widths.index({inside_intvl_mask, Slice()}),
      unnormalized_heights.index({inside_intvl_mask, Slice()}),
      unnormalized_derivatives.index({inside_intvl_mask, Slice()}), -tail, tail,
      -tail, tail, inverse, min_bin_width, min_bin_height, min_derivative);

  outputs = outputs.index_put({inside_intvl_mask}, val);
  logabsdet = logabsdet.index_put({inside_intvl_mask}, lad);

  return {outputs, logabsdet};
}

std::tuple<at::Tensor, at::Tensor>
RQS(at::Tensor inputs, at::Tensor unnormalized_widths,
    at::Tensor unnormalized_heights, at::Tensor unnormalized_derivatives,
    double tail, bool inverse, double min_bin_width, double min_bin_height,
    double min_derivative) {

  auto inside_intvl_mask = (inputs >= -tail) & (inputs <= tail);
  auto outside_intvl_mask = ~inside_intvl_mask;

  auto outputs = torch::zeros_like(inputs);
  auto logabsdet = torch::zeros_like(inputs);

  unnormalized_derivatives =
      torch::constant_pad_nd(unnormalized_derivatives, {1, 1},
                             std::log(std::exp(1 - min_derivative) - 1.));

  outputs = outputs.index_put({outside_intvl_mask},
                              inputs.index({outside_intvl_mask}));

  auto [val, lad] = cosRQS(
      inputs.index({inside_intvl_mask}),
      unnormalized_widths.index({inside_intvl_mask, Slice()}),
      unnormalized_heights.index({inside_intvl_mask, Slice()}),
      unnormalized_derivatives.index({inside_intvl_mask, Slice()}), -tail, tail,
      -tail, tail, inverse, min_bin_width, min_bin_height, min_derivative);

  outputs = outputs.index_put({inside_intvl_mask}, val);
  logabsdet = logabsdet.index_put({inside_intvl_mask}, lad);

  return {outputs, logabsdet};
}
