#pragma once

#include <opencv2/core/mat.hpp>

#include "constants.h"

class Clustering final {
 public:
  Clustering();

 public:
  ~Clustering(){};
  cv::Mat clusterImage(cv::Mat image,
                       int clusters,
                       bool visualize = false,
                       int repetitions = kDefaultAttempts,
                       double eps = kDefaultMinEpsilon,
                       int max_iter = kDefaultMaxIterations);
};
