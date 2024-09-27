//cuda.h
#pragma once
#include <opencv2/opencv.hpp>

void applyGaussianBlurDirect(const cv::Mat& input, cv::Mat& output, int kernelSize, float sigma);
void applyGaussianBlurSeparable(const cv::Mat& input, cv::Mat& output, int kernelSize, float sigma);

