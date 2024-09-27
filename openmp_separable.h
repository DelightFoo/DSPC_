#ifndef SEPARABLECONVOLUTION_H
#define SEPARABLECONVOLUTION_H

#include <opencv2/opencv.hpp> // Include OpenCV for image processing
#include <vector>


using namespace std;
using namespace cv;

// Declaration of the Direct Convolution function
vector<float> createGaussianKernel1D(int ksize, double sigma);
void apply1DConvolution(const Mat& src, Mat& dst, const vector<float>& kernel, bool horizontal);
void applyGaussianBlurSeparable_openmp(const Mat& src, Mat& dst, int ksize, double sigma);

#endif // SEPARABLECONVOLUTION_H