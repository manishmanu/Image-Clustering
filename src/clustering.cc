#include "clustering.h"

#include "k_means.h"

Clustering::Clustering() {}

cv::Mat Clustering::clusterImage(cv::Mat image,
                                 int clusters,
                                 bool visualize,
                                 int repetitions,
                                 double eps,
                                 int max_iter) {
  std::shared_ptr<KMeans> kmeans = std::make_shared<KMeans>(
      clusters, TermCriteria(TermCriteria::Criteria::kAny, max_iter, eps),
      repetitions);

  cv::Mat labels, centers;
  int ret = kmeans->run(image, labels, centers, visualize);
  if (ret == 0) {
    return kmeans->applyLabels(image, labels, centers);
  } else {
    printf("kmeans failed\n");
    return cv::Mat();
  }
}
