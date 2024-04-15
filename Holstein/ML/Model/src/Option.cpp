#include "../Option.h"
#include "INI.h"
#include <cstdio>

using namespace MLHol;

Option Option::read(const std::string &path){
  Option opt;

  // Create datastructure
  mINI::INIStructure ini;                                                                                                            
  // Read from the file
  mINI::INIFile(path).read(ini);

  if (!ini.has("Main")){
    printf("Ini file without Main!\n");
    exit(1);
  }

  if (ini["Main"].has("lr"))
    opt.lr = std::stod(ini["Main"]["lr"]);

  if (ini["Main"].has("cp"))
    opt.CPlateu = std::stod(ini["Main"]["cp"]);

  if (ini["Main"].has("na"))
    opt.NAffine = std::stoi(ini["Main"]["na"]);
  
  if (ini["Main"].has("ne"))
    opt.NEpoch = std::stoi(ini["Main"]["ne"]);

  if (ini["Main"].has("mo"))
    opt.MOrder = std::stoi(ini["Main"]["mo"]);

  if (ini["Main"].has("hi"))
    opt.Hidden = std::stoi(ini["Main"]["hi"]);

  if (ini["Main"].has("nb"))
    opt.Batch = std::stoi(ini["Main"]["nb"]);

  if (ini["Main"].has("nl"))
    opt.NLayers = std::stoi(ini["Main"]["nl"]);

  if (ini["Main"].has("ds"))
    opt.Dsize = std::stoi(ini["Main"]["ds"]);

  if (ini["Main"].has("te"))
    opt.TEval = std::stod(ini["Main"]["te"]);

  if (ini["Main"].has("bo"))
    opt.bound = std::stod(ini["Main"]["bo"]);

  if (ini["Main"].has("nbin"))
    opt.n_bin = std::stod(ini["Main"]["nbin"]);

  if (ini["Main"].has("Mpath"))
    opt.Mpath = ini["Main"]["Mpath"];

  if (ini["Main"].has("rt"))
    opt.ResTrain = std::stoi(ini["Main"]["rt"]);

  if (ini["Main"].has("seed"))
    opt.RNGseed = std::stoi(ini["Main"]["seed"]);

  return opt;
}
