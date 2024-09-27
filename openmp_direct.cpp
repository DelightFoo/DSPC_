#include "openmp_direct.h"
#include <cmath>
const float PI = 3.14159265358979323846;

// Function to generate Gaussian kernel (2D) as a flat array
vector<float> generateGaussianKernel(int size, float sigma) {
    vector<float> kernel(size * size, 0.0f);
    float sum = 0.0f;
    int halfSize = size / 2;
    float twoSigmaSquare = 2.0f * sigma * sigma;

    // Parallelize the kernel generation
#pragma omp parallel for reduction(+:sum)
    for (int y = -halfSize; y <= halfSize; ++y) {
        for (int x = -halfSize; x <= halfSize; ++x) {
            int idx = (y + halfSize) * size + (x + halfSize);
            float exponent = -(x * x + y * y) / twoSigmaSquare;
            kernel[idx] = expf(exponent) / (PI * twoSigmaSquare);
            sum += kernel[idx];
        }
    }

    // Normalize the kernel
#pragma omp parallel for
    for (int i = 0; i < size * size; ++i) {
        kernel[i] /= sum;
    }

    return kernel;
}

// Function to apply Gaussian blur to a single channel using direct convolution
void applyGaussianBlurToChannel(const Mat& channel, Mat& output, const vector<float>& kernel, int kernelSize) {
    int imgHeight = channel.rows;
    int imgWidth = channel.cols;
    int halfKernel = kernelSize / 2;

    // Pad the image to handle borders
    Mat paddedImage;
    copyMakeBorder(channel, paddedImage, halfKernel, halfKernel, halfKernel, halfKernel, BORDER_REPLICATE);

    // Initialize output
    output = Mat::zeros(channel.size(), CV_32F);

    // Access the data directly for efficiency
    const float* paddedData = (const float*)paddedImage.data;
    float* outputData = (float*)output.data;

    // Perform convolution
#pragma omp parallel for
    for (int y = 0; y < imgHeight; ++y) {
        for (int x = 0; x < imgWidth; ++x) {
            float sum = 0.0f;
            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    int ix = x + kx;
                    int iy = y + ky;
                    sum += kernel[ky * kernelSize + kx] * paddedData[iy * paddedImage.cols + ix];
                }
            }
            outputData[y * imgWidth + x] = sum;
        }
    }
}

