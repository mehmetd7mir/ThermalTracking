/*
ThermalTracking - Custom CUDA Kernels
======================================
Low-level CUDA C++ kernels for image preprocessing.

Bu dosya custom CUDA kernels içerir:
1. normalize_kernel: Image normalization
2. grayscale_kernel: RGB to grayscale
3. threshold_kernel: Binary thresholding
4. nms_kernel: Non-Maximum Suppression

Derleme:
    nvcc -shared -o libkernels.so -Xcompiler -fPIC cuda_kernels.cu

Python'dan kullanım:
    import ctypes
    lib = ctypes.CDLL('./libkernels.so')
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)


// ============================================================================
// KERNEL 1: Image Normalization
// ============================================================================
/*
Görüntüyü [0, 255] aralığından [0, 1] aralığına normalize eder.
Ayrıca ImageNet mean/std normalization uygular.

Formula: output = (input / 255.0 - mean) / std
*/
__global__ void normalize_kernel(
    const unsigned char* input,
    float* output,
    int width,
    int height,
    int channels,
    const float* mean,
    const float* std
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    for (int c = 0; c < channels; c++) {
        int input_idx = (y * width + x) * channels + c;
        int output_idx = c * height * width + y * width + x;  // CHW format
        
        float pixel = (float)input[input_idx] / 255.0f;
        output[output_idx] = (pixel - mean[c]) / std[c];
    }
}

// Host wrapper function
extern "C" void cuda_normalize(
    const unsigned char* h_input,
    float* h_output,
    int width,
    int height,
    int channels,
    const float* h_mean,
    const float* h_std
) {
    int input_size = width * height * channels * sizeof(unsigned char);
    int output_size = width * height * channels * sizeof(float);
    int norm_size = channels * sizeof(float);
    
    // Device memory allocation
    unsigned char* d_input;
    float* d_output;
    float* d_mean;
    float* d_std;
    
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
    CUDA_CHECK(cudaMalloc(&d_mean, norm_size));
    CUDA_CHECK(cudaMalloc(&d_std, norm_size));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mean, h_mean, norm_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_std, h_std, norm_size, cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    
    normalize_kernel<<<gridDim, blockDim>>>(
        d_input, d_output, width, height, channels, d_mean, d_std
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_std));
}


// ============================================================================
// KERNEL 2: RGB to Grayscale
// ============================================================================
/*
RGB görüntüyü grayscale'e çevirir.
Formula: Y = 0.299*R + 0.587*G + 0.114*B
*/
__global__ void grayscale_kernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int rgb_idx = (y * width + x) * 3;
    int gray_idx = y * width + x;
    
    float r = (float)input[rgb_idx];
    float g = (float)input[rgb_idx + 1];
    float b = (float)input[rgb_idx + 2];
    
    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    
    output[gray_idx] = (unsigned char)fminf(fmaxf(gray, 0.0f), 255.0f);
}

extern "C" void cuda_grayscale(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height
) {
    int input_size = width * height * 3 * sizeof(unsigned char);
    int output_size = width * height * sizeof(unsigned char);
    
    unsigned char* d_input;
    unsigned char* d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    
    grayscale_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}


// ============================================================================
// KERNEL 3: Binary Thresholding
// ============================================================================
/*
Thermal görüntülerde sıcak bölgeleri tespit etmek için thresholding.
*/
__global__ void threshold_kernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    unsigned char threshold,
    unsigned char max_value
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    output[idx] = (input[idx] > threshold) ? max_value : 0;
}

extern "C" void cuda_threshold(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    unsigned char threshold,
    unsigned char max_value
) {
    int size = width * height * sizeof(unsigned char);
    
    unsigned char* d_input;
    unsigned char* d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    
    threshold_kernel<<<gridDim, blockDim>>>(
        d_input, d_output, width, height, threshold, max_value
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}


// ============================================================================
// KERNEL 4: IoU Calculation for NMS
// ============================================================================
/*
Bounding box'lar arası IoU hesaplama.
NMS (Non-Maximum Suppression) için kullanılır.
*/
__device__ float iou_device(
    float x1_a, float y1_a, float x2_a, float y2_a,
    float x1_b, float y1_b, float x2_b, float y2_b
) {
    float inter_x1 = fmaxf(x1_a, x1_b);
    float inter_y1 = fmaxf(y1_a, y1_b);
    float inter_x2 = fminf(x2_a, x2_b);
    float inter_y2 = fminf(y2_a, y2_b);
    
    float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
    float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;
    
    float area_a = (x2_a - x1_a) * (y2_a - y1_a);
    float area_b = (x2_b - x1_b) * (y2_b - y1_b);
    
    float union_area = area_a + area_b - inter_area;
    
    return (union_area > 0) ? (inter_area / union_area) : 0.0f;
}

__global__ void nms_kernel(
    const float* boxes,  // [N, 4] - x1, y1, x2, y2
    const float* scores,
    int* keep,
    int num_boxes,
    float iou_threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= num_boxes) return;
    
    // Zaten suppress edilmişse atla
    if (keep[i] == 0) return;
    
    float x1_i = boxes[i * 4];
    float y1_i = boxes[i * 4 + 1];
    float x2_i = boxes[i * 4 + 2];
    float y2_i = boxes[i * 4 + 3];
    float score_i = scores[i];
    
    for (int j = 0; j < num_boxes; j++) {
        if (i == j || keep[j] == 0) continue;
        
        // Eğer j daha yüksek score'a sahipse, i'yi suppress et
        if (scores[j] > score_i) {
            float iou = iou_device(
                x1_i, y1_i, x2_i, y2_i,
                boxes[j * 4], boxes[j * 4 + 1],
                boxes[j * 4 + 2], boxes[j * 4 + 3]
            );
            
            if (iou > iou_threshold) {
                keep[i] = 0;
                return;
            }
        }
    }
}


// ============================================================================
// KERNEL 5: Contrast Enhancement
// ============================================================================
/*
Thermal görüntülerde kontrast artırma.
Formula: output = alpha * (input - 128) + 128 + beta
*/
__global__ void contrast_kernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    float alpha,
    float beta
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    float pixel = (float)input[idx];
    float enhanced = alpha * (pixel - 128.0f) + 128.0f + beta;
    
    output[idx] = (unsigned char)fminf(fmaxf(enhanced, 0.0f), 255.0f);
}

extern "C" void cuda_contrast(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    float alpha,
    float beta
) {
    int size = width * height * sizeof(unsigned char);
    
    unsigned char* d_input;
    unsigned char* d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    
    contrast_kernel<<<gridDim, blockDim>>>(
        d_input, d_output, width, height, alpha, beta
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}


// ============================================================================
// DEVICE INFO
// ============================================================================
extern "C" void cuda_get_device_info() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    printf("====================================\n");
    printf("CUDA Device Information\n");
    printf("====================================\n");
    printf("Number of CUDA devices: %d\n\n", device_count);
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads dim: (%d, %d, %d)\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("\n");
    }
}
