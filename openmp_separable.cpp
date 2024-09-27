#include "openmp_separable.h"
#include <cmath>
#include <vector>
const float PI = 3.14159265358979323846;

// Function to create a 1D Gaussian Kernel
vector<float> createGaussianKernel1D(int ksize, double sigma) {
    vector<float> kernel(ksize);
    int halfSize = ksize / 2;
    float sum = 0.0;

#pragma omp parallel for reduction(+:sum)
    for (int i = -halfSize; i <= halfSize; i++) {
        float value = exp(-(i * i) / (2 * sigma * sigma)) / (sqrt(2 * PI) * sigma);
        kernel[i + halfSize] = value;
        sum += value;
    }

    // Normalize the kernel to make sure the sum of the elements is 1
#pragma omp parallel for
    for (int i = 0; i < ksize; i++) {
        kernel[i] /= sum;
    }

    return kernel;
}

// Function to apply 1D convolution on a row or column
void apply1DConvolution(const Mat& src, Mat& dst, const vector<float>& kernel, bool horizontal) {
    int ksize = kernel.size();
    int halfSize = ksize / 2;
    dst = Mat::zeros(src.size(), src.type());

    int rows = src.rows;
    int cols = src.cols;


#pragma omp parallel for
    for (int y = 0; y < rows; y++) {
        const float* srcRow = src.ptr<float>(y);
        float* dstRow = dst.ptr<float>(y);
        for (int x = 0; x < cols; x++) {
            float sum = 0.0f;
            if (horizontal) {
                // Horizontal convolution
                int start = max(x - halfSize, 0);
                int end = min(x + halfSize, cols - 1);
                for (int k = start; k <= end; k++) {
                    sum += kernel[k - (x - halfSize)] * srcRow[k];
                }
            }
            else {
                // Vertical convolution
                for (int k = -halfSize; k <= halfSize; k++) {
                    int yy = y + k;
                    if (yy >= 0 && yy < rows) {
                        sum += kernel[k + halfSize] * src.at<float>(yy, x);
                    }
                }
            }
            dstRow[x] = sum;
        }
    }
}

// Function to apply Gaussian blur using separable convolution
void applyGaussianBlurSeparable_openmp(const Mat& src, Mat& dst, int ksize, double sigma) {
    // Create 1D Gaussian kernel
    vector<float> kernel = createGaussianKernel1D(ksize, sigma);

    // Temporary matrix for the result after the first convolution
    Mat temp;

    // Apply 1D convolution horizontally
    apply1DConvolution(src, temp, kernel, true);

    // Apply 1D convolution vertically
    apply1DConvolution(temp, dst, kernel, false);
}