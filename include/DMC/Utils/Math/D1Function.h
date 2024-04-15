#pragma once

#include <functional>

namespace DMC {

class D1Function {
private:
  typedef double (*fun_t)(double);

public:
  D1Function() {}
  D1Function(fun_t fun) : fun_(fun) {}

  virtual double operator()(double x) const { return fun_(x); };

  operator fun_t() { return fun_; }

private:
  fun_t fun_;

}; // D1Function

} // namespace DMC
