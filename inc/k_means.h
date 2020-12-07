#pragma once

#include <cstdint>

#include <opencv2/core/mat.hpp>

#include "term_criteria.h"

/*
KMeans for clustering on cv::Mat structure.
*/
class KMeans {
 public:
  KMeans(int clusters, TermCriteria term_criteria, int attempts = 5);
  int run(cv::Mat input,
          cv::Mat& labels,
          cv::Mat& centers,
          bool visualize = false) const;
  cv::Mat applyLabels(const cv::Mat input,
                      const cv::Mat labels,
                      const cv::Mat centers) const;

 private:
  void randomInitialize(const cv::Mat image, cv::Mat& centers) const;
  int getClosestCenter(const cv::Vec3b pixel, const cv::Mat centers) const;
  void recalculateCenters(const cv::Mat image,
                          const cv::Mat labels,
                          cv::Mat& centers) const;
  double calculateError(const cv::Mat image,
                        const cv::Mat labels,
                        const cv::Mat centers) const;

 private:
  int clusters_;
  int attempts_;
  TermCriteria term_criteria_;
};
