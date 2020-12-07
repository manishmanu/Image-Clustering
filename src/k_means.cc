#include "k_means.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <limits>

#include <opencv2/videoio.hpp>

KMeans::KMeans(int clusters, TermCriteria term_criteria, int attempts)
    : clusters_(clusters), attempts_(attempts), term_criteria_(term_criteria) {
  attempts_ = std::max(attempts_, 1);
  srand(time(NULL));
}

/*
Currently only supports 3-channel images and only 2D Mat objects.
*/
int KMeans::run(cv::Mat image,
                cv::Mat& labels,
                cv::Mat& centers,
                bool visualize) const {
  if (image.channels() != 3 || image.dims != 2 || clusters_ < 1) {
    printf("Currently only 2D 3 channel image patterns are supported\n");
    return -1;
  }

  centers = cv::Mat(clusters_, 1, CV_32FC3);
  labels = cv::Mat(image.size(), CV_32S);

  double min_error = std::numeric_limits<double>::max();
  int best_attempt;

  for (int a = 0; a < attempts_; a++) {
    cv::Mat current_labels = cv::Mat(image.size(), CV_32S);
    cv::Mat current_centers = cv::Mat(clusters_, 1, CV_32FC3);

    double error = 0.0;
    int iter = 0;

    cv::VideoWriter video;
    if (visualize) {
      std::string gif_name =
          "visualize_kmeans_attempt_" + std::to_string(a) + ".avi";
      video =
          cv::VideoWriter(gif_name, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          10, image.size());
    }

    randomInitialize(image, current_centers);
    while (true) {
      iter++;

      // Assign clusters to pixels
      for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
          current_labels.at<int>(i, j) =
              getClosestCenter(image.at<cv::Vec3b>(i, j), current_centers);
        }
      }

      if (visualize) {
        cv::Mat current_clusters =
            applyLabels(image, current_labels, current_centers);
        video.write(current_clusters);
      }

      // Recalculate cluster centers
      recalculateCenters(image, current_labels, current_centers);

      // check termination criteria
      if ((term_criteria_.getCriteria() ==
               TermCriteria::Criteria::kMaxIterations ||
           term_criteria_.getCriteria() == TermCriteria::Criteria::kAny) &&
          iter == term_criteria_.getMaxIter()) {
        break;
      } else if ((term_criteria_.getCriteria() ==
                      TermCriteria::Criteria::kEpsilon ||
                  term_criteria_.getCriteria() ==
                      TermCriteria::Criteria::kAny)) {
        double new_error =
            calculateError(image, current_labels, current_centers);
        double diff = std::abs(new_error - error);
        error = new_error;
        if (diff <= term_criteria_.getEspilon()) {
          break;
        }
      }
    }

    if (error < min_error) {
      best_attempt = a;
      min_error = error;
      centers = current_centers;
      labels = current_labels;
    }

    video.release();
  }

  if (visualize) {
    for (int a = 0; a < attempts_; a++) {
      std::string gif_name =
          "visualize_kmeans_attempt_" + std::to_string(a) + ".avi";
      if (a != best_attempt) {
        remove(gif_name.c_str());
      } else {
        rename(gif_name.c_str(), "visualize_kmeans.avi");
      }
    }
  }

  return 0;
}

void KMeans::randomInitialize(const cv::Mat image, cv::Mat& centers) const {
  int pixels = image.total();
  int K = centers.rows;

  for (int k = 0; k < K; k++) {
    int pixel = rand() % pixels;
    int r = pixel / image.cols;
    int c = pixel % image.rows;

    cv::Vec3f center;
    center[0] = (float)image.at<cv::Vec3b>(r, c)[0];
    center[1] = (float)image.at<cv::Vec3b>(r, c)[1];
    center[2] = (float)image.at<cv::Vec3b>(r, c)[2];

    centers.at<cv::Vec3f>(k) = center;
  }
}

inline double norm(cv::Vec3b a, cv::Vec3f b) {
  return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2));
}

int KMeans::getClosestCenter(const cv::Vec3b pixel,
                             const cv::Mat centers) const {
  double min_distance = std::numeric_limits<double>::max();
  int closest_center;
  for (int i = 0; i < centers.rows; i++) {
    double distance = norm(pixel, centers.at<cv::Vec3f>(i, 0));
    if (distance < min_distance) {
      min_distance = distance;
      closest_center = i;
    }
  }
  return closest_center;
}

void KMeans::recalculateCenters(const cv::Mat image,
                                const cv::Mat labels,
                                cv::Mat& centers) const {
  float sum[centers.rows][3];
  int count[centers.rows];
  for (int i = 0; i < centers.rows; i++) {
    count[i] = 0;
    for (int j = 0; j < 3; j++) {
      sum[i][j] = 0.f;
    }
  }

  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
      int cluster_ind = labels.at<int>(i, j);

      count[cluster_ind]++;

      sum[cluster_ind][0] += pixel[0];
      sum[cluster_ind][1] += pixel[1];
      sum[cluster_ind][2] += pixel[2];
    }
  }

  for (int i = 0; i < centers.rows; i++) {
    for (int j = 0; j < 3; j++) {
      sum[i][j] /= count[i];
    }
    centers.at<cv::Vec3f>(i, 0) = cv::Vec3f(sum[i][0], sum[i][1], sum[i][2]);
  }
}

double KMeans::calculateError(const cv::Mat image,
                              const cv::Mat labels,
                              const cv::Mat centers) const {
  double error = 0.0;
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      error += pow(norm(image.at<cv::Vec3b>(i, j),
                        centers.at<cv::Vec3f>(labels.at<int>(i, j), 0)),
                   2);
    }
  }
  return error;
}

cv::Mat KMeans::applyLabels(const cv::Mat input,
                            const cv::Mat labels,
                            const cv::Mat centers) const {
  cv::Mat new_image = cv::Mat::zeros(input.size(), CV_8UC3);
  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      cv::Vec3f cluster = centers.at<cv::Vec3f>(labels.at<int>(i, j));
      new_image.at<cv::Vec3b>(i, j)[0] = (uint8_t)cluster[0];
      new_image.at<cv::Vec3b>(i, j)[1] = (uint8_t)cluster[1];
      new_image.at<cv::Vec3b>(i, j)[2] = (uint8_t)cluster[2];
    }
  }
  return new_image;
}
