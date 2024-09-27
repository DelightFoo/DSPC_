//cuda
#include "CUDA.h"
#include <cuda_runtime.h>
#include "utils.h"
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>

__global__ void gaussianBlurDirectKernelRGB(const uchar3* input, uchar3* output, int width, int height, const float* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        int halfKernel = kernelSize / 2;

        for (int ky = -halfKernel; ky <= halfKernel; ++ky) {
            for (int kx = -halfKernel; kx <= halfKernel; ++kx) {
                int ix = clamp(x + kx, 0, width - 1);
                int iy = clamp(y + ky, 0, height - 1);
                float kernelValue = kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
                uchar3 pixel = input[iy * width + ix];
                sum.x += pixel.x * kernelValue;
                sum.y += pixel.y * kernelValue;
                sum.z += pixel.z * kernelValue;
            }
        }

        output[y * width + x] = make_uchar3(
            static_cast<unsigned char>(sum.x),
            static_cast<unsigned char>(sum.y),
            static_cast<unsigned char>(sum.z)
        );
    }
}

__global__ void gaussianBlurSeparableKernelRGB(const uchar3* input, uchar3* output, int width, int height, const float* kernel, int kernelSize, bool horizontal) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        int halfKernel = kernelSize / 2;

        if (horizontal) {
            for (int k = -halfKernel; k <= halfKernel; ++k) {
                int ix = clamp(x + k, 0, width - 1);
                uchar3 pixel = input[y * width + ix];
                float kernelValue = kernel[k + halfKernel];
                sum.x += pixel.x * kernelValue;
                sum.y += pixel.y * kernelValue;
                sum.z += pixel.z * kernelValue;
            }
        }
        else {
            for (int k = -halfKernel; k <= halfKernel; ++k) {
                int iy = clamp(y + k, 0, height - 1);
                uchar3 pixel = input[iy * width + x];
                float kernelValue = kernel[k + halfKernel];
                sum.x += pixel.x * kernelValue;
                sum.y += pixel.y * kernelValue;
                sum.z += pixel.z * kernelValue;
            }
        }

        output[y * width + x] = make_uchar3(
            static_cast<unsigned char>(sum.x),
            static_cast<unsigned char>(sum.y),
            static_cast<unsigned char>(sum.z)
        );
    }
}
void processImageChunk(const cv::Mat& input, cv::Mat& output, int kernelSize, float sigma, int startY, int endY, bool direct) {
    cv::Mat kernel1D = cv::getGaussianKernel(kernelSize, sigma, CV_32F);
    cv::Mat kernel2D;
    if (direct) {
        kernel2D = kernel1D * kernel1D.t();
    }

    float* d_kernel;
    size_t kernelSize1D = kernel1D.total() * sizeof(float);
    size_t kernelSize2D = direct ? kernel2D.total() * sizeof(float) : 0;
    cudaMalloc(&d_kernel, std::max(kernelSize1D, kernelSize2D));

    if (direct) {
        cudaMemcpy(d_kernel, kernel2D.data, kernelSize2D, cudaMemcpyHostToDevice);
    }
    else {
        cudaMemcpy(d_kernel, kernel1D.data, kernelSize1D, cudaMemcpyHostToDevice);
    }

    int chunkHeight = endY - startY;
    cv::Mat inputChunk = input.rowRange(startY, endY);
    cv::Mat outputChunk = output.rowRange(startY, endY);

    uchar3* d_input;
    uchar3* d_output;
    uchar3* d_temp = nullptr;
    cudaMalloc(&d_input, inputChunk.total() * sizeof(uchar3));
    cudaMalloc(&d_output, outputChunk.total() * sizeof(uchar3));

    if (!direct) {
        cudaMalloc(&d_temp, inputChunk.total() * sizeof(uchar3));
    }

    cudaMemcpy(d_input, inputChunk.data, inputChunk.total() * sizeof(uchar3), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x,
        (chunkHeight + blockSize.y - 1) / blockSize.y);

    if (direct) {
        gaussianBlurDirectKernelRGB << <gridSize, blockSize >> > (d_input, d_output, input.cols, chunkHeight, d_kernel, kernelSize);
    }
    else {
        gaussianBlurSeparableKernelRGB << <gridSize, blockSize >> > (d_input, d_temp, input.cols, chunkHeight, d_kernel, kernelSize, true);
        gaussianBlurSeparableKernelRGB << <gridSize, blockSize >> > (d_temp, d_output, input.cols, chunkHeight, d_kernel, kernelSize, false);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(outputChunk.data, d_output, outputChunk.total() * sizeof(uchar3), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    if (d_temp) cudaFree(d_temp);
}

void applyGaussianBlurDirect(const cv::Mat& input, cv::Mat& output, int kernelSize, float sigma) {
    int optimalChunkSize = 1024;
    for (int startY = 0; startY < input.rows; startY += optimalChunkSize) {
        int endY = std::min(startY + optimalChunkSize, input.rows);
        processImageChunk(input, output, kernelSize, sigma, startY, endY, true);
    }
}

void applyGaussianBlurSeparable(const cv::Mat& input, cv::Mat& output, int kernelSize, float sigma) {
    int optimalChunkSize = 1024; 
    for (int startY = 0; startY < input.rows; startY += optimalChunkSize) {
        int endY = std::min(startY + optimalChunkSize, input.rows);
        processImageChunk(input, output, kernelSize, sigma, startY, endY, false);
    }
}
