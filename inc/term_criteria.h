#pragma once

#include <cstdint>

#include "constants.h"

class TermCriteria {
 public:
  enum class Criteria : int8_t {
    kEpsilon = 1 << 0,
    kMaxIterations = 1 << 1,
    kAny = kEpsilon | kMaxIterations
  };

 public:
  TermCriteria(Criteria criteria,
               int max_iter = kDefaultMaxIterations,
               double epsilon = kDefaultMinEpsilon)
      : max_iter_(max_iter), epsilon_(epsilon), criteria_(criteria){};

  Criteria getCriteria() const { return criteria_; }

  int getMaxIter() const { return max_iter_; }

  double getEspilon() const { return epsilon_; }

 private:
  int max_iter_;
  double epsilon_;
  Criteria criteria_;
};
