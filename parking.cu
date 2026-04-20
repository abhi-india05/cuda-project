
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

typedef struct
{
    int minx, miny, maxx, maxy;
} BBox;

typedef struct
{
    int occupied;
    float density;
} SlotResult;

#define MAX_SLOTS 30


void rgb_to_gray_cpu(unsigned char *rgb, unsigned char *gray, int w, int h)
{
    for (int i = 0; i < w * h; i++)
    {
        gray[i] = 0.299f * rgb[3 * i] +
                  0.587f * rgb[3 * i + 1] +
                  0.114f * rgb[3 * i + 2];
    }
}

void sobel_cpu(unsigned char *gray, unsigned char *edges, int w, int h)
{
    for (int y = 1; y < h - 1; y++)
    {
        for (int x = 1; x < w - 1; x++)
        {
            int Gx = -gray[(y - 1) * w + (x - 1)] + gray[(y - 1) * w + (x + 1)] - 2 * gray[y * w + (x - 1)] + 2 * gray[y * w + (x + 1)] - gray[(y + 1) * w + (x - 1)] + gray[(y + 1) * w + (x + 1)];

            int Gy = -gray[(y - 1) * w + (x - 1)] - 2 * gray[(y - 1) * w + x] - gray[(y - 1) * w + (x + 1)] + gray[(y + 1) * w + (x - 1)] + 2 * gray[(y + 1) * w + x] + gray[(y + 1) * w + (x + 1)];

            int mag = abs(Gx) + abs(Gy);
            edges[y * w + x] = (mag > 80) ? 255 : 0;
        }
    }
}

void classify_roi_cpu(unsigned char *edges, int w, BBox *boxes,
                      SlotResult *results, int num_slots, float threshold)
{
    for (int i = 0; i < num_slots; i++)
    {
        BBox b = boxes[i];
        int edge_pixels = 0;
        int total = (b.maxx - b.minx + 1) * (b.maxy - b.miny + 1);

        for (int y = b.miny; y <= b.maxy; y++)
        {
            for (int x = b.minx; x <= b.maxx; x++)
            {
                if (edges[y * w + x] == 255)
                    edge_pixels++;
            }
        }

        float d = (float)edge_pixels / total;
        results[i].density = d;
        results[i].occupied = (d > threshold);
    }
}



__global__ void rgb_to_gray(unsigned char *rgb, unsigned char *gray, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    int i = y * w + x;
    gray[i] = 0.299f * rgb[3 * i] + 0.587f * rgb[3 * i + 1] + 0.114f * rgb[3 * i + 2];
}

__global__ void sobel_kernel(unsigned char *gray, unsigned char *edges, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= w - 1 || y <= 0 || y >= h - 1)
        return;

    int Gx = -gray[(y - 1) * w + (x - 1)] + gray[(y - 1) * w + (x + 1)] - 2 * gray[y * w + (x - 1)] + 2 * gray[y * w + (x + 1)] - gray[(y + 1) * w + (x - 1)] + gray[(y + 1) * w + (x + 1)];

    int Gy = -gray[(y - 1) * w + (x - 1)] - 2 * gray[(y - 1) * w + x] - gray[(y - 1) * w + (x + 1)] + gray[(y + 1) * w + (x - 1)] + 2 * gray[(y + 1) * w + x] + gray[(y + 1) * w + (x + 1)];

    int mag = abs(Gx) + abs(Gy);
    edges[y * w + x] = (mag > 80) ? 255 : 0; 
}

__global__ void classify_roi_kernel(unsigned char *edges, int w, BBox *boxes, SlotResult *results, int num_slots, float threshold)
{
    int slot_id = blockIdx.x;
    if (slot_id >= num_slots)
        return;

    BBox box = boxes[slot_id];
    __shared__ int edge_pixels;
    if (threadIdx.x == 0)
        edge_pixels = 0;
    __syncthreads();

    int total_pixels = (box.maxx - box.minx + 1) * (box.maxy - box.miny + 1);

    for (int y = box.miny + threadIdx.y; y <= box.maxy; y += blockDim.y)
    {
        for (int x = box.minx + threadIdx.x; x <= box.maxx; x += blockDim.x)
        {
            if (edges[y * w + x] == 255)
            {
                atomicAdd(&edge_pixels, 1);
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        float d = (float)edge_pixels / total_pixels;
        results[slot_id].density = d;
        results[slot_id].occupied = (d > threshold);
    }
}



int get_rois(const char *filename, BBox *boxes, int w, int h)
{
    if (strstr(filename, "parking.jpg"))
    {
        int count = 0;
        for (int i = 0; i < 6; i++)
        { 
            boxes[count++] = (BBox){100 + i * 83, 70, 175 + i * 83, 170};
        }
        for (int i = 0; i < 6; i++)
        { 
            boxes[count++] = (BBox){105 + i * 83, 280, 185 + i * 83, 420};
        }
        return count;
    }
    else
    {
#define IMG_W 626.0f
#define IMG_H 376.0f

        int count = 0;

#define SX(x) ((int)((x) / IMG_W * w))
#define SY(y) ((int)((y) / IMG_H * h))

        for (int i = 0; i < 3; i++)
            boxes[count++] = (BBox){SX(96), SY(54 + i * 45), SX(168), SY(98 + i * 45)};
        for (int i = 0; i < 3; i++)
            boxes[count++] = (BBox){SX(220), SY(54 + i * 45), SX(289), SY(98 + i * 45)};
        for (int i = 0; i < 3; i++)
            boxes[count++] = (BBox){SX(378), SY(54 + i * 45), SX(452), SY(98 + i * 45)};
        for (int i = 0; i < 3; i++)
            boxes[count++] = (BBox){SX(500), SY(54 + i * 45), SX(572), SY(98 + i * 45)};

        for (int i = 0; i < 8; i++)
            boxes[count++] = (BBox){SX(230 + i * 42 + 2), SY(250), SX(274 + i * 42), SY(322)};

        return count;
    }
}



int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "parking4.jpg";
    int w, h, c;
    unsigned char *img = stbi_load(path, &w, &h, &c, 3);
    if (!img)
    {
        printf("Can't open image\n");
        return 1;
    }

    BBox h_boxes[MAX_SLOTS];
    int num_slots = get_rois(path, h_boxes, w, h);

    unsigned char *d_rgb, *d_gray, *d_edges;
    BBox *d_boxes;
    SlotResult *d_results;

    unsigned char *gray_cpu = (unsigned char *)malloc(w * h);
    unsigned char *edges_cpu = (unsigned char *)malloc(w * h);
    SlotResult cpu_results[MAX_SLOTS];

    clock_t cpu_start = clock();

    rgb_to_gray_cpu(img, gray_cpu, w, h);
    sobel_cpu(gray_cpu, edges_cpu, w, h);
    classify_roi_cpu(edges_cpu, w, h_boxes, cpu_results, num_slots, 0.35f);

    clock_t cpu_end = clock();
    double cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&d_rgb, w * h * 3);
    cudaMalloc(&d_gray, w * h);
    cudaMalloc(&d_edges, w * h);
    cudaMalloc(&d_boxes, sizeof(BBox) * num_slots);
    cudaMalloc(&d_results, sizeof(SlotResult) * num_slots);

    cudaMemcpy(d_rgb, img, w * h * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_boxes, h_boxes, sizeof(BBox) * num_slots, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);

    cudaEventRecord(start);
    rgb_to_gray<<<grid, block>>>(d_rgb, d_gray, w, h);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0, total_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    total_time += gpu_time;
    printf("Gray scale time: %f\n", gpu_time);

    cudaEventRecord(start);
    sobel_kernel<<<grid, block>>>(d_gray, d_edges, w, h);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_time, start, stop);
    total_time += gpu_time;
    printf("Sobel kernel time: %f\n", gpu_time);

    cudaEventRecord(start);
    classify_roi_kernel<<<num_slots, dim3(16, 16)>>>(d_edges, w, d_boxes, d_results, num_slots, 0.35f);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_time, start, stop);
    total_time += gpu_time;
    printf("Classification kernel time: %f\n", gpu_time);

    SlotResult h_results[MAX_SLOTS];
    cudaMemcpy(h_results, d_results, sizeof(SlotResult) * num_slots, cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_slots; i++)
    {
        unsigned char r = h_results[i].occupied ? 255 : 0;
        unsigned char g = h_results[i].occupied ? 0 : 255;
        BBox b = h_boxes[i];

        for (int x = b.minx; x <= b.maxx; x++)
        {
            for (int y : {b.miny, b.maxy})
            { 
                if (x >= 0 && x < w && y >= 0 && y < h)
                {
                    int idx = (y * w + x) * 3;
                    img[idx] = r;
                    img[idx + 1] = g;
                    img[idx + 2] = 0;
                }
            }
        }
        for (int y = b.miny; y <= b.maxy; y++)
        {
            for (int x : {b.minx, b.maxx})
            { 
                if (x >= 0 && x < w && y >= 0 && y < h)
                {
                    int idx = (y * w + x) * 3;
                    img[idx] = r;
                    img[idx + 1] = g;
                    img[idx + 2] = 0;
                }
            }
        }
        printf("%d %d %d %d \n", b.minx, b.maxx, b.miny, b.maxy);
        printf("Slot %d: %s (Density: %.2f%%)\n", i, h_results[i].occupied ? "OCCUPIED" : "VACANT", h_results[i].density * 100);
    }

    printf("\n===== PERFORMANCE =====\n");
    printf("CPU Time: %.2f ms\n", cpu_time);
    printf("GPU Time: %.2f ms\n", total_time);
    printf("Speedup: %.2fx\n", cpu_time / total_time);

    stbi_write_png("final_result.png", w, h, 3, img, w * 3);
    printf("Saved result to final_result.png\n");

    return 0;
}
