#include "Model/Model.h"
#include <NF/fmanager.h>

#include <cstdio>
#include <torch/optim.h>
#include <torch/optim/sgd.h>
#include <torch/torch.h>
#include <torch/utils.h>

using namespace MLHol;

int main(int argc, char *argv[]) {
  //---TECNICAL STUFF---//
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                   : torch::kCPU);

  //---READ OPTIONS---//
  Option opt;

  if (argc > 1)
    opt = Option::read(std::string(argv[1]) + "sim.ini");
  else
    opt = Option::read();

  torch::manual_seed(opt.RNGseed);

  //---CREATE THE FLOW---//
  Base base(opt.MOrder, torch::kF64);
  Target targ(opt.MOrder, opt.CPlateu);

  LLNF::FlowManager flow(base, create_flow_list(opt), targ);
  if (opt.ResTrain)
    torch::load(flow, std::string(argv[1]) + "Mod.pt");
  flow->to(device);

  //---OPTIMIZER AND LOSS SETUP---//
  torch::optim::SGD optim(flow->parameters(), torch::optim::SGDOptions(opt.lr));

  torch::Tensor loss;
  std::vector<double> loss_hist;

  //---TRAINING LOOP---//
  std::cout << "Starting training:\n" << std::endl;
  for (int i = 1; i != opt.NEpoch + 1; i++) {
    for (int j = 0; j != 1E2; j++) {
      flow->zero_grad();

      loss = flow->inverse_KL(opt.Batch);

      if (loss.isinf().item<bool>() || loss.isnan().item<bool>()) {
        std::cerr << "Loss exploded!" << std::endl;
        goto end;
      }

      loss.backward();
      optim.step();

      loss_hist.push_back(loss.detach().item<double>());
    }

    printf("End of epoch %3d the loss is: %.4f\n", i, *(loss_hist.end() - 1));
    torch::save(flow, (std::string(argv[1]) + "Mod.pt").data());
  }

  std::cout << "\nEnded training saving the model!" << std::endl;

  torch::save(flow, (std::string(argv[1]) + "Mod.pt").data());
  print_vector(loss_hist, (std::string(argv[1]) + "loss.dat").data(),
               opt.ResTrain);

end:
  return 0;
}
