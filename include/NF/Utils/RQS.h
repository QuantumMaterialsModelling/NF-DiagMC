#pragma once

#include <ATen/core/TensorBody.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> cosRQS(at::Tensor inputs,
                                       at::Tensor unnormalized_widths,
                                       at::Tensor unnormalized_heights,
                                       at::Tensor unnormalized_derivatives,
                                       at::Tensor left,
                                       at::Tensor right,
                                       at::Tensor bottom,
                                       at::Tensor top,
                                       bool inverse = false,
                                       double min_bin_width = 1e-3,
                                       double min_bin_height = 1e-3,
                                       double min_derivative = 1e-3
                                       );


std::tuple<at::Tensor, at::Tensor> RQS(at::Tensor inputs,
                                       at::Tensor unnormalized_widths,
                                       at::Tensor unnormalized_heights,
                                       at::Tensor unnormalized_derivatives,
                                       at::Tensor tail,
                                       bool inverse = false,
                                       double min_bin_width = 1e-3,
                                       double min_bin_height = 1e-3,
                                       double min_derivative = 1e-3
                                       );




std::tuple<at::Tensor, at::Tensor> cosRQS(at::Tensor inputs,
                                       at::Tensor unnormalized_widths,
                                       at::Tensor unnormalized_heights,
                                       at::Tensor unnormalized_derivatives,
                                       double left,
                                       double right,
                                       double bottom,
                                       double top,
                                       bool inverse = false,
                                       double min_bin_width = 1e-3,
                                       double min_bin_height = 1e-3,
                                       double min_derivative = 1e-3
                                       );


std::tuple<at::Tensor, at::Tensor> RQS(at::Tensor inputs,
                                       at::Tensor unnormalized_widths,
                                       at::Tensor unnormalized_heights,
                                       at::Tensor unnormalized_derivatives,
                                       double tail,
                                       bool inverse = false,
                                       double min_bin_width = 1e-3,
                                       double min_bin_height = 1e-3,
                                       double min_derivative = 1e-3
                                       );
