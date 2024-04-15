#pragma once

#include <ATen/core/TensorBody.h>

class Transformer{
public:
  Transformer(int n_features): n_features_(n_features){}

  virtual at::Tensor forward(const at::Tensor& z, const at::Tensor& h) = 0;
  virtual at::Tensor inverse(const at::Tensor& z, const at::Tensor& h) = 0;
  virtual at::Tensor log_det(const at::Tensor& z, const at::Tensor& h) = 0;

  inline const int & get_num_features() const{return n_features_;}

private:
  int n_features_;
};
