#include "Model/Model.h"
#include <ATen/ops/meshgrid.h>
#include <NF/Utils/histogram.h>
#include <NF/fmanager.h>

#include <cstdio>
#include <torch/torch.h>
#include <torch/utils.h>

using namespace MLHol;

int main(int argc, char *argv[]) {
  //---TECNICAL STUFF---//
  torch::Device device(torch::kCPU);

  //---READ OPTIONS---//
  Option opt;

  if (argc > 1)
    opt = Option::read(std::string(argv[1]) + "sim.ini");
  else
    opt = Option::read();

  //---CREATE THE FLOW---//
  auto base = Base(opt.MOrder, torch::kF64);
  auto targ = Target(opt.MOrder, opt.CPlateu);

  LLNF::FlowManager flow(base, create_flow_list(opt), targ);
  torch::load(flow, std::string(argv[1]) + "Mod.pt");
  flow->to(device);
  flow->eval();

  //---SAMPLING TRIAL---//
  printf("--------------------EVALUATION--------------------\n\nCollection "
         "statistic...\n");
  Histogram t1(100, 0, opt.TEval);
  Histogram t2(100, 0, opt.TEval);
  auto cond = torch::tensor({{opt.TEval}}, torch::kF64).expand({opt.Batch, 1});

  FILE *file = fopen((std::string(argv[1]) + "tt.dat").data(), "w");

  for (int i = 0; i != 100; i++) {
    auto samples = flow->cond_sample(cond);

    for (int j = 0; j != opt.Batch; j++) {
      t1.add(*samples.index({j, 1}).data_ptr<double>());
      t2.add(*samples.index({j, 2}).data_ptr<double>());
    }

    for (int i = 0; i != opt.Batch; i++)
      fprintf(file, "%lf %lf\n", *samples.index({i, 1}).data_ptr<double>(),
              *samples.index({i, 2}).data_ptr<double>());
  }

  fclose(file);
  t1.print((std::string(argv[1]) + "t1.dat").data());
  t2.print((std::string(argv[1]) + "t2.dat").data());

  // Clean stuff
  cond = torch::empty(0);

  printf("...end!\n");

  //---2D SAMPLING---//
  printf("\nStaring Density estimation...\n");

  //---LOG PROBABILITY PLOT---//
  auto x = torch::linspace(-0.1, opt.TEval + 0.1, opt.Dsize, torch::kF64);
  auto y = torch::linspace(-0.1, opt.TEval + 0.1, opt.Dsize, torch::kF64);

  auto grid = torch::meshgrid({x, y}, "xy");
  auto prob = torch::cat({torch::tensor({{opt.TEval}}, torch::kF64)
                              .expand({opt.Dsize * opt.Dsize, 1}),
                          grid[0].flatten().unsqueeze(-1),
                          grid[1].flatten().unsqueeze(-1),
                          torch::tensor({opt.TEval - 0.1})
                              .expand({opt.Dsize * opt.Dsize, opt.MOrder - 2})},
                         1);

  prob = flow->log_prob(prob).exp();
  prob.index_put_({prob.isnan()}, 0);
  prob.index_put_({prob.isinf()}, 0);

  FILE *pfile = fopen((std::string(argv[1]) + "prob.dat").data(), "w");

  for (int i = 0; i != prob.size(0); i++)
    fprintf(pfile, "%lf\n", *prob[i].data_ptr<double>());

  fclose(pfile);

  printf("...end!\n");

  return 0;
}
