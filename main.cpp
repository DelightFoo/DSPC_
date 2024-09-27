//main
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "CUDA.h"
#include "utils.h"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>
#include <mpi.h>
#include <iomanip>


#include "openmp_direct.h"
#include "openmp_separable.h"

const double PI = 3.14159265358979323846;

using namespace cv;
using namespace std;
namespace fs = filesystem;

string outputDir = "C:\\Users\\user\\Desktop\\output_images\\";
string inputFilename;
string imagePath;
int kernelSize, method;
float sigma;
double elapsedTime = 0.0;
double elapsedTimeS = 0.0;
double process_time_CUDA_direct = 0.0;
double process_time_CUDA_separable = 0.0;
double process_time_MPI_direct = 0.0;
double process_time_MPI_separable = 0.0;
double elapsedTime_single_direct, elapsedTime_single_separable;
string image_size;

chrono::high_resolution_clock::time_point start_time_omp;
chrono::high_resolution_clock::time_point end_time_omp;
chrono::high_resolution_clock::time_point start_time_cuda;
chrono::high_resolution_clock::time_point end_time_cuda;
chrono::high_resolution_clock::time_point start_time_mpi;
chrono::high_resolution_clock::time_point end_time_mpi;
chrono::high_resolution_clock::time_point start_time_single;
chrono::high_resolution_clock::time_point end_time_single;

// Function to generate Gaussian kernel (direct convolution)
vector<vector<float>> generateGaussianKernel_single(int size, float sigma) {
    vector<vector<float>> kernel(size, vector<float>(size));
    float sum = 0.0f;
    int halfSize = size / 2;
    float twoSigmaSquare = 2.0f * sigma * sigma;


    for (int x = -halfSize; x <= halfSize; ++x) {
        for (int y = -halfSize; y <= halfSize; ++y) {
            float exponent = -(x * x + y * y) / twoSigmaSquare;
            kernel[x + halfSize][y + halfSize] = exp(exponent) / (PI * twoSigmaSquare);
            sum += kernel[x + halfSize][y + halfSize];
        }
    }


    // Normalize the kernel
    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            kernel[x][y] /= sum;
        }
    }

    return kernel;
}

// Function to apply Gaussian blur to a single channel(direct convolution)
void applyGaussianBlurToChannel_single(const Mat& channel, cv::Mat& output,
    const vector<vector<float>>& kernel) {
    int imgHeight = channel.rows;
    int imgWidth = channel.cols;
    int kernelSize = kernel.size();
    int halfKernel = kernelSize / 2;

    Mat_<float> paddedImage;
    copyMakeBorder(channel, paddedImage, halfKernel, halfKernel, halfKernel, halfKernel, BORDER_REPLICATE);

    //start to process 
    for (int y = 0; y < imgHeight; ++y) {
        for (int x = 0; x < imgWidth; ++x) {
            float sum = 0.0f;
            for (int ky = -halfKernel; ky <= halfKernel; ++ky) {
                for (int kx = -halfKernel; kx <= halfKernel; ++kx) {
                    int ix = x + kx + halfKernel;
                    int iy = y + ky + halfKernel;
                    sum += paddedImage.at<float>(iy, ix) * kernel[ky + halfKernel][kx + halfKernel];
                }
            }
            output.at<float>(y, x) = sum;
        }
    }
}

// Function to create a 1D Gaussian Kernel(separable convolution)
vector<float> createGaussianKernel1D_single(int ksize, double sigma) {
    vector<float> kernel(ksize);
    int halfSize = ksize / 2;
    float sum = 0.0;

    for (int i = -halfSize; i <= halfSize; i++) {
        float value = exp(-(i * i) / (2 * sigma * sigma)) / (sqrt(2 * PI) * sigma);
        kernel[i + halfSize] = value;
        sum += value;
    }

    // Normalize the kernel to make sure the sum of the elements is 1
    for (int i = 0; i < ksize; i++) {
        kernel[i] /= sum;
    }

    return kernel;
}

// Function to apply 1D convolution on a row or column //(separable convolution)
void apply1DConvolution_single(const Mat& src, Mat& dst, const vector<float>& kernel, bool horizontal) {
    int halfSize = kernel.size() / 2;
    dst = src.clone();


    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float sum = 0.0;

            for (int k = -halfSize; k <= halfSize; k++) {
                int idx = horizontal ? x + k : y + k;
                int validIdx = horizontal ? (idx >= 0 && idx < src.cols) : (idx >= 0 && idx < src.rows);

                if (validIdx) {
                    if (horizontal) {
                        sum += kernel[k + halfSize] * src.at<float>(y, x + k);
                    }
                    else {
                        sum += kernel[k + halfSize] * src.at<float>(y + k, x);
                    }
                }
            }

            dst.at<float>(y, x) = sum;
        }
    }
}

// Function to apply Gaussian blur using separable convolution (separable convolution)
void applyGaussianBlurSeparable_single(const Mat& src, Mat& dst, int ksize, double sigma) {
    // Create 1D Gaussian kernel
    vector<float> kernel = createGaussianKernel1D_single(ksize, sigma);

    // Temporary matrix for the result after the first convolution
    Mat temp;

    // Apply 1D convolution horizontally
    apply1DConvolution_single(src, temp, kernel, true);

    // Apply 1D convolution vertically
    apply1DConvolution_single(temp, dst, kernel, false);
}

/*-----------------------------------------------------------------------------CUDA---------------------------------------------------------------------*/
struct ImageProcessingParams {
    int kernelSize;
    float sigma;
    int chunkSize;
};

ImageProcessingParams calculateAdaptiveParams(const cv::Mat& image) {
    ImageProcessingParams params;

    // Calculate the diagonal size of the image
    double diagonalSize = sqrt(image.rows * image.rows + image.cols * image.cols);

    // Adjust kernel size based on image diagonal (1% of diagonal, odd number, min 3, max 31)
    params.kernelSize = max(3, min(40, (int)(diagonalSize * 0.01) | 1));

    // Adjust sigma based on kernel size, with a more conservative growth
    params.sigma = 0.3 * ((params.kernelSize - 1) * 0.5 - 0.5) + 0.8;
    params.sigma = min(params.sigma, 10.0f);  // Cap sigma at 10.0

    // Adjust chunk size based on image size and available GPU memory
    size_t freeMemory, totalMemory;
    cudaMemGetInfo(&freeMemory, &totalMemory);
    double memoryFactor = min(1.0, (double)freeMemory / (totalMemory * 0.8)); // Use up to 80% of free memory

    int maxChunkSize = min(image.rows, image.cols);
    params.chunkSize = max(256, min(1024, (int)(sqrt(freeMemory / 3 / sizeof(float)) * memoryFactor)));

    cout << "Calculated parameters:" << endl;
    cout << "Image size: " << image.cols << "x" << image.rows << endl;
    cout << "Kernel size: " << params.kernelSize << endl;
    cout << "Sigma: " << params.sigma << endl;
    cout << "Chunk size: " << params.chunkSize << endl;

    return params;
}

string generateRandomFileName(const string& extension) {
    static random_device rd;
    static mt19937 gen(rd());
    static uniform_int_distribution<> dis(0, 35); // 0-9 and A-Z
    string result = "output_";
    for (int i = 0; i < 8; ++i) {
        int r = dis(gen);
        result += r < 10 ? '0' + r : 'A' + (r - 10);
    }
    return result + extension;
}

void checkGPUMemory() {
    size_t free_byte;
    size_t total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cudaSuccess != cuda_status) {
        cerr << "Error: cudaMemGetInfo fails, " << cudaGetErrorString(cuda_status) << endl;
        return;
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;

    cout << "GPU memory usage: used = " << used_db / 1024.0 / 1024.0 << " MB, free = " << free_db / 1024.0 / 1024.0 << " MB, total = " << total_db / 1024.0 / 1024.0 << " MB" << endl;
}

void processImageInChunks(const cv::Mat& input, cv::Mat& output, void (*blurFunction)(const cv::Mat&, cv::Mat&, int, float), int kernelSize, float sigma, int chunkSize) {
    for (int y = 0; y < input.rows; y += chunkSize) {
        for (int x = 0; x < input.cols; x += chunkSize) {
            int width = min(chunkSize, input.cols - x);
            int height = min(chunkSize, input.rows - y);

            cv::Mat inputChunk = input(cv::Rect(x, y, width, height));
            cv::Mat outputChunk = output(cv::Rect(x, y, width, height));

            blurFunction(inputChunk, outputChunk, kernelSize, sigma);
        }
    }
}

//direct convolution
int CUDA_direct() {
    string outputDir = "C:\\Users\\user\\Desktop\\output_images\\";
    fs::create_directories(outputDir); // Ensure the output directory exists

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cerr << "No CUDA-capable devices found!" << endl;
        return -1;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cout << "Using GPU: " << deviceProp.name << endl;

        // Load image
        cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (inputImage.empty()) {
            cerr << "Error: Could not read input image" << endl;
            return -1;
        }

        image_size = to_string(inputImage.cols) +"x"+ to_string(inputImage.rows);

        cout << "Image size: " << inputImage.cols << "x" << inputImage.rows << " pixels" << endl;
        cout << "Image size in memory: " << inputImage.total() * inputImage.elemSize() / (1024 * 1024) << " MB" << endl;

        // Check GPU memory before processing
        checkGPUMemory();

        // Calculate adaptive parameters
        ImageProcessingParams params = calculateAdaptiveParams(inputImage);

        // Prepare output images
        cv::Mat output_direct(inputImage.size(), inputImage.type());


        // Process image with adaptive parameters for direct convolution
        start_time_cuda = chrono::high_resolution_clock::now();
        applyGaussianBlurDirect(inputImage, output_direct, params.kernelSize, params.sigma);
        end_time_cuda = chrono::high_resolution_clock::now();
        process_time_CUDA_direct = chrono::duration_cast<chrono::milliseconds>(end_time_cuda - start_time_cuda).count();

        cout << "Direct Convolution Time: " << setprecision(4) << process_time_CUDA_direct << " ms" << endl;

        // Check GPU memory after processing
        checkGPUMemory();

        // Show images
        cv::imshow("Original Image", inputImage);
        cv::imshow("(Direct) Gaussian Blur", output_direct);
        cv::waitKey(1);  // Update the windows

        // Generate random file names and save output images
        string directOutputPath = outputDir + generateRandomFileName("_cuda_direct.jpg");
        imwrite(directOutputPath, output_direct);

        if (!cv::imwrite(directOutputPath, output_direct)) {
            cerr << "Failed to save direct output image" << endl;
        }
        else {
            cout << "Direct output saved as: " << directOutputPath << endl;
        }

    cv::destroyAllWindows();
    return 0;
}

//seprable convolution
int CUDA_separable() {
    string outputDir = "C:\\Users\\user\\Desktop\\output_images\\";
    fs::create_directories(outputDir); // Ensure the output directory exists

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cerr << "No CUDA-capable devices found!" << endl;
        return -1;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cout << "Using GPU: " << deviceProp.name << endl;
            // Load image
            cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (inputImage.empty()) {
                cerr << "Error: Could not read input image" << endl;
                return -1;
            }
            //calculate image size
            string image_col_size = "" + inputImage.cols;
            string image_row_size = "" + inputImage.rows;
            image_size = image_col_size + "x" + image_row_size;

            cout << "Image size: " << inputImage.cols << "x" << inputImage.rows << " pixels" << endl;
            cout << "Image size in memory: " << inputImage.total() * inputImage.elemSize() / (1024 * 1024) << " MB" << endl;

            // Check GPU memory before processing
            checkGPUMemory();

            // Calculate adaptive parameters
            ImageProcessingParams params = calculateAdaptiveParams(inputImage);

            cout << "Adaptive parameters: " << endl;
            cout << "Kernel Size: " << params.kernelSize << endl;
            cout << "Sigma: " << params.sigma << endl;
            cout << "Chunk Size: " << params.chunkSize << endl;

            // Prepare output images
            cv::Mat output_separable(inputImage.size(), inputImage.type());

            // Process image with adaptive parameters
            start_time_cuda = chrono::high_resolution_clock::now();
            applyGaussianBlurSeparable(inputImage, output_separable, params.kernelSize, params.sigma);
            end_time_cuda = chrono::high_resolution_clock::now();
            process_time_CUDA_separable = chrono::duration_cast<chrono::milliseconds>(end_time_cuda - start_time_cuda).count();


            cout << "Separable Convolution Time: " << process_time_CUDA_separable << " ms" << endl;

            // Check GPU memory after processing
            checkGPUMemory();

            // Show images
            cv::imshow("Original Image", inputImage);
            cv::imshow("(Separable) Gaussian Blur", output_separable);
            cv::waitKey(1);  // Update the windows

            // Generate random file names and save output images
            string separableOutputPath = outputDir + generateRandomFileName("_cuda_separable.jpg");

            if (!cv::imwrite(separableOutputPath, output_separable)) {
                cerr << "Failed to save separable output image" << endl;
            }
            else {
                cout << "Separable output saved as: " << separableOutputPath << endl;
            }
    cv::destroyAllWindows();
    return 0;
}
/*-----------------------------------------------------------------------------CUDA---------------------------------------------------------------------*/


/*-----------------------------------------------------------------------------MPI---------------------------------------------------------------------*/
//MPI DIRECT
// Function to generate a 2D Gaussian kernel
vector<vector<float>> generateGaussianKernel2D(float sigma, int kernelRadius) {
    int size = 2 * kernelRadius + 1;
    vector<vector<float>> kernel(size, vector<float>(size));
    float sum = 0.0;
    float sigma2 = 2.0f * sigma * sigma;

    // Generate the 2D Gaussian kernel
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int x = i - kernelRadius;
            int y = j - kernelRadius;
            kernel[i][j] = exp(-(x * x + y * y) / sigma2);
            sum += kernel[i][j];
        }
    }

    // Normalize the kernel
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

// Direct convolution using a 2D Gaussian kernel
void applyDirectConvolution(const Mat& input, Mat& output, const vector<vector<float>>& kernel, int kernelRadius) {
    int width = input.cols;
    int height = input.rows;
    int kernelSize = 2 * kernelRadius + 1;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Vec3f sum(0, 0, 0); // Process each channel (B, G, R)

            // Apply the 2D convolution
            for (int ky = -kernelRadius; ky <= kernelRadius; ++ky) {
                for (int kx = -kernelRadius; kx <= kernelRadius; ++kx) {
                    int yy = min(max(y + ky, 0), height - 1); // Handle edges
                    int xx = min(max(x + kx, 0), width - 1);
                    sum += input.at<Vec3f>(yy, xx) * kernel[ky + kernelRadius][kx + kernelRadius];
                }
            }

            output.at<Vec3f>(y, x) = sum;
        }
    }
}

int mpi_direct() {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    string outputDir = "C:\\Users\\user\\Desktop\\output_images\\";
    fs::create_directories(outputDir); // Ensure the output directory exists


    // Get the number of processes and the rank of the process
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Declare variables for the image
    Mat image, blurred;

    if (world_rank == 0) {
        // Load the image on the root process
        image = imread(imagePath, IMREAD_COLOR);
        if (image.empty()) {
            cerr << "Could not open or find the image!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Convert the image to float type for processing
        image.convertTo(image, CV_32F, 1.0 / 255.0); // Normalize to [0, 1]
    }

    // Broadcast image dimensions to all processes
    int width = image.cols;
    int height = image.rows;
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0) {
        // Allocate space for the image on all other processes
        image = Mat(height, width, CV_32FC3);
    }

    // Broadcast the image data to all processes
    MPI_Bcast(image.ptr<float>(), width * height * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Set the Gaussian blur parameters
    float sigma = 1.0f;
    int kernelRadius = 3;
    vector<vector<float>> kernel = generateGaussianKernel2D(sigma, kernelRadius);

    // Divide the image processing among the processes
    int rows_per_proc = height / world_size;
    int extra_rows = height % world_size;  // Remaining rows to be handled by the last process
    int start_row = world_rank * rows_per_proc + min(world_rank, extra_rows);
    int end_row = start_row + rows_per_proc;
    if (world_rank < extra_rows) {
        end_row += 1;  // Processes 0 to (extra_rows-1) handle one extra row
    }

    Mat local_temp = image.rowRange(start_row, end_row).clone();
    Mat local_blurred = Mat::zeros(local_temp.size(), CV_32FC3);

    // Start timing
    start_time_mpi = chrono::high_resolution_clock::now();

    // Apply direct convolution for the local rows
    applyDirectConvolution(local_temp, local_blurred, kernel, kernelRadius);

    int local_row_count = local_temp.rows;
    vector<int> recvcounts(world_size);
    vector<int> displs(world_size);

    // Root process calculates recvcounts and displacements for each process
    if (world_rank == 0) {
        for (int i = 0; i < world_size; ++i) {
            recvcounts[i] = ((height / world_size) + (i < extra_rows ? 1 : 0)) * width * 3;
            displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
        }
    }

    // Gather using MPI_Gatherv to account for variable-sized chunks
    MPI_Gatherv(local_blurred.ptr<float>(), local_row_count * width * 3, MPI_FLOAT,
        image.ptr<float>(), recvcounts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Stop timing
    end_time_mpi = chrono::high_resolution_clock::now();
    process_time_MPI_direct = chrono::duration_cast<chrono::milliseconds>(end_time_mpi - start_time_mpi).count();

    // On the root process, display the results
    if (world_rank == 0) {
        cout << "Gaussian Blur with Direct Convolution took " << process_time_MPI_direct << " ms." << endl;

        // Convert the blurred image back to 8-bit format
        image.convertTo(blurred, CV_8UC3, 255.0);

        // Display the original and blurred images
        imshow("Original Image", imread(imagePath, IMREAD_COLOR));
        imshow("Blurred Image", blurred);

        // Save the blurred image if needed
        fs::create_directories(outputDir); // Ensure the output directory exists
        string directOutputPath = outputDir + generateRandomFileName("_mpi_direct.jpg");
        imwrite(directOutputPath, blurred);
        // Wait for a key press to close the images
    }

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}
//MPI SEPARABLE
vector<float> generateGaussianKernel(float sigma, int kernelRadius) {
    int size = 2 * kernelRadius + 1;
    vector<float> kernel(size);
    float sum = 0.0;
    float sigma2 = 2.0f * sigma * sigma;

    // Generate the Gaussian kernel
    for (int i = 0; i < size; ++i) {
        int x = i - kernelRadius;
        kernel[i] = exp(-x * x / sigma2);
        sum += kernel[i];
    }

    // Normalize the kernel
    for (int i = 0; i < size; ++i) {
        kernel[i] /= sum;
    }

    return kernel;
}

// Horizontal pass: Convolve a row with the 1D Gaussian kernel
void horizontalPass(const Mat& input, Mat& output, const vector<float>& kernel, int kernelRadius) {
    int width = input.cols;
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < width; ++x) {
            Vec3f sum(0, 0, 0); // Process each channel (B, G, R)
            for (int k = -kernelRadius; k <= kernelRadius; ++k) {
                int index = min(max(x + k, 0), width - 1); // Handle edges
                sum += input.at<Vec3f>(y, index) * kernel[k + kernelRadius];
            }
            output.at<Vec3f>(y, x) = sum;
        }
    }
}

// Vertical pass: Convolve a column with the 1D Gaussian kernel
void verticalPass(const Mat& input, Mat& output, const vector<float>& kernel, int kernelRadius) {
    int height = input.rows;
    for (int x = 0; x < input.cols; ++x) {
        for (int y = 0; y < height; ++y) {
            Vec3f sum(0, 0, 0); // Process each channel (B, G, R)
            for (int k = -kernelRadius; k <= kernelRadius; ++k) {
                int index = min(max(y + k, 0), height - 1); // Handle edges
                sum += input.at<Vec3f>(index, x) * kernel[k + kernelRadius];
            }
            output.at<Vec3f>(y, x) = sum;
        }
    }
}

int mpi_separable() {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes and the rank of the process
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Declare variables for the image
    Mat image, blurred;

    if (world_rank == 0) {
        // Load the image on the root process
        image = imread(imagePath, IMREAD_COLOR);
        if (image.empty()) {
            cerr << "Could not open or find the image!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Convert the image to float type for processing
        image.convertTo(image, CV_32F, 1.0 / 255.0); // Normalize to [0, 1]
    }

    // Broadcast image dimensions to all processes
    int width = image.cols;
    int height = image.rows;
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0) {
        // Allocate space for the image on all other processes
        image = Mat(height, width, CV_32FC3);
    }

    // Broadcast the image data to all processes
    MPI_Bcast(image.ptr<float>(), width * height * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Set the Gaussian blur parameters
    float sigma = 1.0f;
    int kernelRadius = 3;
    vector<float> kernel = generateGaussianKernel(sigma, kernelRadius);

    // Divide the image processing among the processes
    int rows_per_proc = height / world_size;
    int extra_rows = height % world_size;  // Remaining rows to be handled by the last process
    int start_row = world_rank * rows_per_proc + min(world_rank, extra_rows);
    int end_row = start_row + rows_per_proc;
    if (world_rank < extra_rows) {
        end_row += 1;  // Processes 0 to (extra_rows-1) handle one extra row
    }

    Mat local_temp = image.rowRange(start_row, end_row).clone();
    Mat local_blurred = Mat::zeros(local_temp.size(), CV_32FC3);

    // Start timing
    start_time_mpi = chrono::high_resolution_clock::now();

    // Horizontal pass for the local rows
    horizontalPass(local_temp, local_blurred, kernel, kernelRadius);

    // Vertical pass for the local rows
    verticalPass(local_blurred, local_temp, kernel, kernelRadius);

    int local_row_count = local_temp.rows;
    vector<int> recvcounts(world_size);
    vector<int> displs(world_size);

    // Root process calculates recvcounts and displacements for each process
    if (world_rank == 0) {
        for (int i = 0; i < world_size; ++i) {
            recvcounts[i] = ((height / world_size) + (i < extra_rows ? 1 : 0)) * width * 3;
            displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
        }
    }

    // Gather using MPI_Gatherv to account for variable-sized chunks
    MPI_Gatherv(local_temp.ptr<float>(), local_row_count * width * 3, MPI_FLOAT,
        image.ptr<float>(), recvcounts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Stop timing
    end_time_mpi = chrono::high_resolution_clock::now();
    process_time_MPI_separable = chrono::duration_cast<chrono::milliseconds>(end_time_mpi - start_time_mpi).count();

    // On the root process, display the results
    if (world_rank == 0) {
        cout << "Gaussian Blur with Separable Convolution took " << process_time_MPI_separable << " ms." << endl;

        // Convert the blurred image back to 8-bit format
        image.convertTo(blurred, CV_8UC3, 255.0);

        // Display the original and blurred images
        imshow("Original Image", imread(imagePath, IMREAD_COLOR));
        imshow("Blurred Image", blurred);

        // Save the blurred image if needed
        fs::create_directories(outputDir); // Ensure the output directory exists
        string separableOutputPath = outputDir + generateRandomFileName("_mpi_separable.jpg");
        imwrite(separableOutputPath, blurred);

        // Wait for a key press to close the images
    }

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}

/*-----------------------------------------------------------------------------MPI---------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------OpenMP---------------------------------------------------------------------*/
void openmp_direct() {
    // Read the input image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Could not open or find the image at " << imagePath << endl;
    }

    //Prepare for Gaussian blur 
    Mat bChannel, gChannel, rChannel;

    // Direct Convolution
    Mat bOutput, gOutput, rOutput;


    // Split the image into color channels
    vector<cv::Mat> channels(3);
    split(image, channels);

    // Convert channels to float
    channels[0].convertTo(bChannel, CV_32F); // Blue channel
    channels[1].convertTo(gChannel, CV_32F); // Green channel
    channels[2].convertTo(rChannel, CV_32F); // Red channel


    // Automatically calculate kernel size and sigma based on image dimensions
    int minDim = min(image.rows, image.cols);
    kernelSize = (minDim * 1) / 100;  // Kernel size is 1% of the smallest image dimension
    if (kernelSize % 2 == 0) kernelSize++; // Ensure kernel size is odd
    sigma = 0.15f * kernelSize;       // Sigma proportional to kernel size

    cout << "Auto-calculated kernel size: " << kernelSize << " and sigma: " << sigma << endl;


    //direct convolution 
    auto kernel = generateGaussianKernel(kernelSize, sigma);

    start_time_omp = chrono::high_resolution_clock::now();


#pragma omp parallel for
    for (int c = 0; c < 3; c++) {
        if (c == 0) {
            applyGaussianBlurToChannel(bChannel, bOutput, kernel, kernelSize);
        }
        else if (c == 1) {
            applyGaussianBlurToChannel(gChannel, gOutput, kernel, kernelSize);
        }
        else if (c == 2) {
            applyGaussianBlurToChannel(rChannel, rOutput, kernel, kernelSize);
        }
    }

    end_time_omp = chrono::high_resolution_clock::now();



    // Merge channels back into a single image (direct Convolution)
    vector<cv::Mat> mergedChannels = { bOutput, gOutput, rOutput };
    cv::Mat mergedOutput;
    cv::merge(mergedChannels, mergedOutput);

    // Normalize and convert the output image back to CV_8U (direct Convolution)
    cv::Mat output8U;
    cv::normalize(mergedOutput, output8U, 0, 255, cv::NORM_MINMAX);
    output8U.convertTo(output8U, CV_8U);

    elapsedTime = chrono::duration_cast<chrono::milliseconds>(end_time_omp - start_time_omp).count();

    // Write the output image(direct convolution)
    string outputDir = "C:\\Users\\user\\Desktop\\output_images\\";
    fs::create_directories(outputDir); // Ensure the output directory exists
    string directOutputPath = outputDir + generateRandomFileName("_openmp_direct.jpg");
    imwrite(directOutputPath, output8U);

    cout << "Processing time: (Direct Convolution)     " << elapsedTime << " ms. " << endl;

    cout << "Gaussian blur applied and image saved as " << directOutputPath << endl;


    //original
    namedWindow("Image", WINDOW_AUTOSIZE);
    imshow("Image", image);
    //show images
    //direct convolution
    namedWindow("Direct Convolution", WINDOW_AUTOSIZE);
    imshow("Direct Convolution", output8U);
    waitKey(0); // Wait for a key press

    destroyAllWindows();

}
// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void openmp_separable() {
    // Read the input image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Could not open or find the image at " << imagePath << endl;

    }

    //Prepare for Gaussian blur 
    Mat bChannel, gChannel, rChannel;

    //Separable Convolution
    Mat bOutputS, gOutputS, rOutputS;


    // Split the image into color channels
    vector<cv::Mat> channels(3);
    split(image, channels);

    // Convert channels to float
    channels[0].convertTo(bChannel, CV_32F); // Blue channel
    channels[1].convertTo(gChannel, CV_32F); // Green channel
    channels[2].convertTo(rChannel, CV_32F); // Red channel


   


    // Automatically calculate kernel size and sigma based on image dimensions
    int minDim = min(image.rows, image.cols);
    kernelSize = (minDim * 1.5) / 100;  // Kernel size is 2% of the smallest image dimension
    if (kernelSize % 2 == 0) kernelSize++; // Ensure kernel size is odd
    sigma = 0.2f * kernelSize;       // Sigma proportional to kernel size

    cout << "Auto-calculated kernel size: " << kernelSize << " and sigma: " << sigma << endl;

    //separable convolution 

    start_time_omp = chrono::high_resolution_clock::now();

    // Apply separable Gaussian blur to each channel
#pragma omp parallel sections
    {
#pragma omp section
        {
            applyGaussianBlurSeparable_openmp(bChannel, bOutputS, kernelSize, sigma);
        }

#pragma omp section
        {
            applyGaussianBlurSeparable_openmp(gChannel, gOutputS, kernelSize, sigma);
        }

#pragma omp section
        {
            applyGaussianBlurSeparable_openmp(rChannel, rOutputS, kernelSize, sigma);
        }
    }

    end_time_omp = chrono::high_resolution_clock::now();

    // Merge channels back into a single image (Separable Convolution)
    vector<cv::Mat> mergedChannelsS = { bOutputS, gOutputS, rOutputS };
    cv::Mat mergedOutputS;
    cv::merge(mergedChannelsS, mergedOutputS);



    // Normalize and convert the output image back to CV_8U (Separable Convolution)
    cv::Mat output8US;
    cv::normalize(mergedOutputS, output8US, 0, 255, cv::NORM_MINMAX);
    output8US.convertTo(output8US, CV_8U);

    elapsedTimeS = chrono::duration_cast<chrono::milliseconds>(end_time_omp - start_time_omp).count();

    // Write the output image (separable convolution)
    string outputDir = "C:\\Users\\user\\Desktop\\output_images\\";
    fs::create_directories(outputDir); // Ensure the output directory exists
    string separableOutputPath = outputDir + generateRandomFileName("_openmp_separable.jpg");
    imwrite(separableOutputPath, output8US);

    cout << "Processing time: (Separable  Convolution) " << elapsedTimeS << " ms. " << endl;

    cout << "Gaussian blur applied and image saved as " << separableOutputPath << endl;

    //original 
    namedWindow("Image", WINDOW_AUTOSIZE);
    imshow("Image", image);
    //separable convolution
    imshow("Separable Image", output8US);

 // Wait for a key press
    destroyAllWindows();

}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void direct_convolution_single() {
    // Generate output filename based on input filename


    // Read the input image
    //TODO: change the inputFilename 
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Could not open or find the image at " << imagePath << endl;
    }

    //Prepare for Gaussian blur 
    Mat bChannel, gChannel, rChannel;

    // Direct Convolution
    Mat bOutput, gOutput, rOutput;


    // Split the image into color channels
    vector<cv::Mat> channels(3);
    split(image, channels);

    // Convert channels to float
    channels[0].convertTo(bChannel, CV_32F); // Blue channel
    channels[1].convertTo(gChannel, CV_32F); // Green channel
    channels[2].convertTo(rChannel, CV_32F); // Red channel

    // Create output channels (direct Convolution )
    bOutput.create(bChannel.size(), CV_32F);
    gOutput.create(gChannel.size(), CV_32F);
    rOutput.create(rChannel.size(), CV_32F);


    // Automatically calculate kernel size and sigma based on image dimensions
    int minDim = min(image.rows, image.cols);
    kernelSize = (minDim * 2) / 100;  // Kernel size is 2% of the smallest image dimension
    if (kernelSize % 2 == 0) kernelSize++; // Ensure kernel size is odd
    sigma = 0.3f * kernelSize;       // Sigma proportional to kernel size

    cout << "Auto-calculated kernel size: " << kernelSize << " and sigma: " << sigma << endl;


    //direct convolution 
    auto kernel = generateGaussianKernel_single(kernelSize, sigma);

    start_time_single = std::chrono::high_resolution_clock::now();

    // Apply Gaussian blur to each channel



    applyGaussianBlurToChannel_single(bChannel, bOutput, generateGaussianKernel_single(kernelSize, sigma));
    applyGaussianBlurToChannel_single(gChannel, gOutput, generateGaussianKernel_single(kernelSize, sigma));
    applyGaussianBlurToChannel_single(rChannel, rOutput, generateGaussianKernel_single(kernelSize, sigma));



    end_time_single = std::chrono::high_resolution_clock::now();



    // Merge channels back into a single image (direct Convolution)
    std::vector<cv::Mat> mergedChannels = { bOutput, gOutput, rOutput };
    cv::Mat mergedOutput;
    cv::merge(mergedChannels, mergedOutput);

    // Normalize and convert the output image back to CV_8U (direct Convolution)
    cv::Mat output8U_single_direct;
    cv::normalize(mergedOutput, output8U_single_direct, 0, 255, cv::NORM_MINMAX);
    output8U_single_direct.convertTo(output8U_single_direct, CV_8U);

    elapsedTime_single_direct = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_single - start_time_single).count();


   

    string outputDir = "C:\\Users\\user\\Desktop\\output_images\\";
    fs::create_directories(outputDir); // Ensure the output directory exists
    string directOutputPath = outputDir + generateRandomFileName("_single_direct.jpg");
    imwrite(directOutputPath, output8U_single_direct);

    std::string outputFilenameSingle = "Sblurred_" + imagePath; //separable convolution 
    cout << "Processing time: (Direct Convolution)     " << elapsedTime_single_direct << " ms. " << std::endl;

    cout << "Gaussian blur applied and image saved as " << outputFilenameSingle << endl;


    //original
    namedWindow("Image", WINDOW_AUTOSIZE);
    imshow("Image", image);
    //show images
    //direct convolution
    namedWindow("Direct Convolution", WINDOW_AUTOSIZE);
    imshow("Direct Convolution", output8U_single_direct);

    waitKey(0); // Wait for a key press
    destroyAllWindows();

}

void separable_convolution_single() {



    // Read the input image
    //TODO : change the inputfile name 
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Could not open or find the image at " << imagePath << endl;

    }

    //Prepare for Gaussian blur 
    Mat bChannel, gChannel, rChannel;

    //Separable Convolution
    Mat bOutputS, gOutputS, rOutputS;


    // Split the image into color channels
    vector<cv::Mat> channels(3);
    split(image, channels);

    // Convert channels to float
    channels[0].convertTo(bChannel, CV_32F); // Blue channel
    channels[1].convertTo(gChannel, CV_32F); // Green channel
    channels[2].convertTo(rChannel, CV_32F); // Red channel


    // Create output channels (separable Convolution )
    bOutputS.create(bChannel.size(), CV_32F);
    gOutputS.create(bChannel.size(), CV_32F);
    rOutputS.create(bChannel.size(), CV_32F);


    // Automatically calculate kernel size and sigma based on image dimensions
    int minDim = min(image.rows, image.cols);
    kernelSize = (minDim * 2) / 100;  // Kernel size is 2% of the smallest image dimension
    if (kernelSize % 2 == 0) kernelSize++; // Ensure kernel size is odd
    sigma = 0.3f * kernelSize;       // Sigma proportional to kernel size

    cout << "Auto-calculated kernel size: " << kernelSize << " and sigma: " << sigma << endl;

    //separable convolution 

    start_time_single = std::chrono::high_resolution_clock::now();

    // Apply separable Gaussian blur to each channel

    applyGaussianBlurSeparable_single(bChannel, bOutputS, kernelSize, sigma);
    applyGaussianBlurSeparable_single(gChannel, gOutputS, kernelSize, sigma);
    applyGaussianBlurSeparable_single(rChannel, rOutputS, kernelSize, sigma);


    end_time_single = std::chrono::high_resolution_clock::now();


    // Merge channels back into a single image (Separable Convolution)
    std::vector<cv::Mat> mergedChannelsS = { bOutputS, gOutputS, rOutputS };
    cv::Mat mergedOutputS;
    cv::merge(mergedChannelsS, mergedOutputS);



    // Normalize and convert the output image back to CV_8U (Separable Convolution)
    cv::Mat output8U_single_separable;
    cv::normalize(mergedOutputS, output8U_single_separable, 0, 255, cv::NORM_MINMAX);
    output8U_single_separable.convertTo(output8U_single_separable, CV_8U);

    elapsedTime_single_separable = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_single - start_time_single).count();

    //TODO : change the save file locatin 
    // Generate output filename based on input filename


   
    string outputDir = "C:\\Users\\user\\Desktop\\output_images\\";
    fs::create_directories(outputDir); // Ensure the output directory exists
    string directOutputPath = outputDir + generateRandomFileName("_single_separable.jpg");
    imwrite(directOutputPath, output8U_single_separable);

    std::string outputFilenameS = "Sblurred_" + imagePath; //separable convolution 

    cout << "Processing time: (Separable  Convolution) " << elapsedTime_single_separable << " ms. " << std::endl;

    cout << "Gaussian blur applied and image saved as " << outputFilenameS << endl;








    //original 
    namedWindow("Image", WINDOW_AUTOSIZE);
    imshow("Image", image);
    //separable convolution
    imshow("Separable Image", output8U_single_separable);

    waitKey(0); // Wait for a key press
    destroyAllWindows();
}


int main() {

    cout << "Please choose the Gaussian Blur processing method: \n" << "1. Direct Convulution\n" <<"2. Separable Convolution\n";
    cin >> method;
    cout << "Please enter your image path :";
    cin >> imagePath;
    if (method == 1) {
        //openmp_direct();
        CUDA_direct();
        //mpi_direct();
        //direct_convolution_single();
        waitKey(0); // Wait for a key press

       
    }
    else if (method == 2) {
        //openmp_separable();
        CUDA_separable();
        //mpi_separable();
        //separable_convolution_single();
        waitKey(0); // Wait for a key press
    }
            
}