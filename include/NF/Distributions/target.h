#pragma once

#include <NF/Archetypes/distribution.h>
#include <torch/nn/pimpl.h>

//**************************//
//---EXONENTIAL EXPANSION---//
//**************************//

class ExpExpansionImpl : public TargetDistribution {
public:
  ExpExpansionImpl(double E, double V);

  torch::Tensor log_prob(const torch::Tensor &z) override;

private:
  double E_V_l_, E_, V_;
};

TORCH_MODULE(ExpExpansion);

//*********************//
//---ON SITE POLARON---//
//*********************//

class OnSitePolaronImpl : public TargetDistribution {
public:
  OnSitePolaronImpl(int max_order, double G, double O);

  torch::Tensor log_prob(const torch::Tensor &z) override;
  torch::Tensor sample(uint num_sample = 1) override;

private:
  double G_, O_;
};

TORCH_MODULE(OnSitePolaron);
