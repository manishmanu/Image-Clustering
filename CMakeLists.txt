cmake_minimum_required(VERSION 2.8)
project(ImageClustering)

find_package(OpenCV REQUIRED)

add_library(clustering STATIC ${CMAKE_SOURCE_DIR}/src/k_means.cc ${CMAKE_SOURCE_DIR}/src/clustering.cc)
target_link_libraries(clustering PUBLIC ${OpenCV_LIBS})
target_include_directories(clustering PUBLIC ${CMAKE_SOURCE_DIR}/inc)

add_executable(example ${CMAKE_SOURCE_DIR}/example.cc)
target_link_libraries(example PUBLIC clustering ${OpenCV_LIBS})
target_include_directories(example PUBLIC ${CMAKE_SOURCE_DIR}/inc)
