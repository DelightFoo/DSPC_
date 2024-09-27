#ifndef DIRECTCONVOLUTION_H
#define DIRECTCONVOLUTION_H

#include <opencv2/opencv.hpp> // Include OpenCV for image processing
#include <vector>


using namespace std;
using namespace cv;

vector<float> generateGaussianKernel(int size, float sigma);

void applyGaussianBlurToChannel(const Mat& channel, Mat& output, const vector<float>& kernel, int kernelSize);


#endif // DIRECTCONVOLUTION_H