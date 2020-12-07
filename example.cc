#include <iostream>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "clustering.h"

void createClustersGIF(cv::Mat img, int min_clusters, int max_clusters) {
  std::shared_ptr<Clustering> clustering = std::make_shared<Clustering>();

  cv::VideoWriter video("image_clustering.avi",
                        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 2,
                        img.size());
  for (int c = min_clusters; c <= max_clusters; c++) {
    printf("running with clusters : %d\n", c);
    cv::Mat clustered_img = clustering->clusterImage(img, c);
    if (!clustered_img.empty()) {
      video.write(clustered_img);
    } else {
      printf("Image clustering failed\n");
    }
  }
  video.release();
}

int main(int argc, char** argv) {
  cv::Mat img = cv::imread(argv[1]);
  bool gif_mode = false;

  if (gif_mode) {
    createClustersGIF(img, 2, 15);
  } else {
    int K = std::stoi(argv[2]);
    std::string output_name = argv[1];
    std::shared_ptr<Clustering> clustering = std::make_shared<Clustering>();
    cv::Mat clustered_img = clustering->clusterImage(img, K, true);
    if (!clustered_img.empty()) {
      cv::imwrite("output_" + std::to_string(K) + ".png", clustered_img);
    } else {
      printf("Image clustering failed\n");
    }
  }
}
