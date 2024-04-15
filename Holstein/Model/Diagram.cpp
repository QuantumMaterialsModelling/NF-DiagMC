#pragma once

#include <DMC/Archetypes.h>
#include <Model.h>
#include <NF/fmanager.h>
#include <Option.h>
#include <array>
#include <torch/torch.h>

namespace Holstein {

class Diagram : public DMC::Configuration {
public:
  Diagram() : Configuration("Holstein") {
    MLHol::Option opt =
        MLHol::Option::read("./Holstein/ML/Results/RQS/2_5_20/sim.ini");
    auto base = MLHol::Base(opt.MOrder, torch::kF64);
    auto targ = MLHol::Target(opt.MOrder, opt.CPlateu);

    NNsample = LLNF::FlowManager(base, MLHol::create_flow_list(opt), targ);
    torch::load(NNsample, "./Holstein/ML/Results/RQS/2_5_20/Mod.pt");
    NNsample->eval();
    NNsample->to(torch::kCPU);
  }

  void add_p(double b, double e) {
    Pb[n] = b, Pe[n] = e;

    n++;
  }

  void rem_p(int which) {
    std::swap(Pb[which], Pb[n - 1]);
    std::swap(Pe[which], Pe[n - 1]);

    n--;
  }

  void set_param(std::map<std::string, double> param) override {
    g = param["g"], o = param["o"], e = param["e"];
  }

  torch::Tensor get_tensor() {
    auto res = torch::zeros({n, 3}, torch::kF64);

    for (int i = 0; i != n; i++)
      res.index_put_({i, 0}, t), res.index_put_({i, 1}, Pb[i]),
          res.index_put_({i, 2}, Pe[i]);

    return res;
  }

  void set_tensor(const torch::Tensor &z) {
    if (z.size(0) == 0) {
      n = 0;
      return;
    }

    n = z.size(0);

    for (int i = 0; i != n; i++) {
      Pb[i] = *z.index({i, 1}).to(torch::kCPU).data_ptr<double>();
      Pe[i] = *z.index({i, 2}).to(torch::kCPU).data_ptr<double>();
    }
  }

public:
  int n{0};
  std::array<double, 1000> Pb, Pe;
  double t{1};

  double g, o, e;

  LLNF::FlowManager NNsample{nullptr};
};

} // namespace Holstein
