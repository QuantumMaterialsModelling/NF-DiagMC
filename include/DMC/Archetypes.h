#pragma once

//******************//
//---DEPENDENCIES---//
//******************//
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>

namespace DMC {

//*****************************//
//---PREEMPTIVE DECLARATIONS---//
//*****************************//
template <class DIA> class Manager;

//*******************//
//---CONFIGURATION---//
//*******************//

class Configuration {
public:
  Configuration(const char *name) : name_(name) {}

  virtual void set_param(std::map<std::string, double> param) = 0;

  const char *name() const { return name_.data(); }

private:
  std::string name_;

}; // Configuration

//*******************//
//---UPDATE STATUS---//
//*******************//

enum UpStatus { ACCEPTED = 0, REJECTED = 1, INVALID = 2 };

//************//
//---UPDATE---//
//************//

template <class DIA> class Update {
public:
  friend Manager<DIA>;

  std::shared_ptr<DIA> dia;
  std::mt19937_64 rng;

  Update(const char *name) : name_(name), inverse_(name) {}
  Update(const char *name, const char *inverse)
      : name_(name), inverse_(inverse) {}

  const char *name() const { return name_.data(); }
  const char *inverse() const { return inverse_.data(); }

  virtual double atempt() = 0;
  virtual void accept() = 0;

private:
  inline virtual void set_rng(std::mt19937_64 &new_rng) { rng.seed(new_rng()); }
  inline virtual void set_dia(std::shared_ptr<DIA> &new_dia) { dia = new_dia; }
  inline void set_inverse(std::string inverse) { inverse_ = inverse; }

private:
  std::string name_;
  std::string inverse_;

}; // Update

//****************//
//---OBSERVABLE---//
//****************//

template <class DIA> class Observable {
public:
  friend Manager<DIA>;

  std::shared_ptr<DIA> dia;

  Observable(const char *name) : name_(name) {}

  const char *name() const { return name_.data(); }

  virtual void eval() = 0;
  virtual void print(const std::string &path) = 0;

  virtual void conv() {}
  bool is_converged() const { return converged; }

private:
  inline virtual void set_dia(std::shared_ptr<DIA> &new_dia) { dia = new_dia; }

protected:
  bool converged{false};

private:
  std::string name_;
}; // Observable

} // namespace DMC
