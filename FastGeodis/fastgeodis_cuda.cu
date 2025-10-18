// BSD 3-Clause License

// Copyright (c) 2021, Muhammad Asad (masadcv@gmail.com)
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

#define THREAD_COUNT 256

// whether to use float* or Pytorch accessors in CUDA kernels
#define USE_PTR 1

__constant__ float local_dist2d[3];
__constant__ float local_dist3d[3 * 3];

__device__ float l1distance_cuda(const float &in1, const float &in2)
{
    return abs(in1 - in2);
}

template <typename scalar_t>
__global__ void geodesic_updown_single_row_pass_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> image_ptr,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> distance_ptr,
    const float l_grad,
    const float l_eucl,
    const int direction)
{
    const int channel = image_ptr.size(1);
    const int height = image_ptr.size(2);
    const int width = image_ptr.size(3);

    int kernelW = blockIdx.x * blockDim.x + threadIdx.x;

    int h = (direction == 1)? 1 : height - 2; 

    // if outside, then skip distance calculation - dont use the thread
    if (kernelW < width)
    {
        while (h >= 0 && h < height)
        {
            int prevH = h - direction;
            if (prevH < 0 || prevH >= height)
            {
                // read outside bounds, skip
                continue;
            }
            
            float l_dist, cur_dist;
            float new_dist = distance_ptr[0][0][h][kernelW];

            for (int w_i = 0; w_i < 3; w_i++)
            {
                const int kernelW_ind = kernelW + w_i - 1;

                if (kernelW_ind >= 0 && kernelW_ind < width)
                {
                    l_dist = 0.0;
                    if (channel == 1)
                    {
                        l_dist = l1distance_cuda(
                            image_ptr[0][0][h][kernelW], 
                            image_ptr[0][0][prevH][kernelW_ind]
                            );
                    }
                    else
                    {
                        for (int c_i = 0; c_i < channel; c_i++)
                        {
                            l_dist += l1distance_cuda(
                                image_ptr[0][c_i][h][kernelW], 
                                image_ptr[0][c_i][prevH][kernelW_ind]
                                );
                        }
                    }
                    cur_dist = distance_ptr[0][0][prevH][kernelW_ind] + \
                                l_eucl * local_dist2d[w_i] + \
                                l_grad * l_dist;

                    new_dist = std::min(new_dist, cur_dist);
                }
            }
            if (new_dist < distance_ptr[0][0][h][kernelW])
            {
                distance_ptr[0][0][h][kernelW] = new_dist;
            }

            // go to next row
            h += direction;

            // synchronise write for all threads
            __syncthreads();
        }
    }
}

__global__ void geodesic_updown_single_row_pass_ptr_kernel(
    float *image_ptr,
    float *distance_ptr,
    const float l_grad,
    const float l_eucl,
    const int direction,
    const int channel,
    const int height,
    const int width)
{
    int kernelW = blockIdx.x * blockDim.x + threadIdx.x;

    int h = (direction == 1)? 1 : height - 2; 

    // if outside, then skip distance calculation - dont use the thread
    if (kernelW < width)
    {
        while (h >= 0 && h < height)
        {
            int prevH = h - direction;
            if (prevH < 0 || prevH >= height)
            {
                // read outside bounds, skip
                continue;
            }
            float l_dist, cur_dist;
            float new_dist = distance_ptr[h * width + kernelW];

            for (int w_i = 0; w_i < 3; w_i++)
            {
                const int kernelW_ind = kernelW + w_i - 1;

                if (kernelW_ind >= 0 && kernelW_ind < width)
                {
                    l_dist = 0.0;
                    if (channel == 1)
                    {
                        l_dist = l1distance_cuda(
                            image_ptr[h * width + kernelW], 
                            image_ptr[(prevH) * width + kernelW_ind]
                            );
                    }
                    else
                    {
                        for (int c_i = 0; c_i < channel; c_i++)
                        {
                            l_dist += l1distance_cuda(
                                image_ptr[c_i * height * width + h * width + kernelW], 
                                image_ptr[c_i * height * width + (prevH) * width + kernelW_ind]
                                );
                        }
                    }
                    cur_dist = distance_ptr[(prevH) * width + kernelW_ind] + \
                                l_eucl * local_dist2d[w_i] + \
                                l_grad * l_dist;

                    new_dist = std::min(new_dist, cur_dist);
                }
            }
            if (new_dist < distance_ptr[h * width + kernelW])
            {
                distance_ptr[h * width + kernelW] = new_dist;
            }

            // go to next row
            h += direction;

            // synchronise write for all threads
            __syncthreads();
        }
    }
}

void geodesic_updown_pass_cuda(
    const torch::Tensor image, 
    torch::Tensor distance, 
    const float &l_grad, 
    const float &l_eucl
    )
{
    // batch, channel, height, width
    const int channel = image.size(1);
    const int height = image.size(2);
    const int width = image.size(3);

    // constexpr float local_dist[] = {sqrt(2.), 1., sqrt(2.)};
    const float local_dist[] = {sqrt(float(2.)), float(1.), sqrt(float(2.))};

    // copy local distances to GPU __constant__ memory
    cudaMemcpyToSymbol(local_dist2d, local_dist, sizeof(float) * 3);

    int blockCountUpDown = (width + 1) / THREAD_COUNT + 1;

    // direction variable used to indicate read from previous (-1) or next (+1) row
    int direction;

    // top-down
    direction = +1;
    // each distance calculation in down pass require previous row i.e. +1
    // process each row in parallel with CUDA kernel
    #if USE_PTR
        geodesic_updown_single_row_pass_ptr_kernel<<<blockCountUpDown, THREAD_COUNT>>>(
            image.data_ptr<float>(),
            distance.data_ptr<float>(),
            l_grad,
            l_eucl,
            direction,
            channel,
            height,
            width);
    #else
        AT_DISPATCH_FLOATING_TYPES(image.type(), "geodesic_updown_single_row_pass_kernel", ([&]
            { geodesic_updown_single_row_pass_kernel<scalar_t><<<blockCountUpDown, THREAD_COUNT>>>(
                image.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                distance.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                l_grad,
                l_eucl,
                direction); 
            }));
    #endif


    // bottom-up
    direction = -1;
    #if USE_PTR
        geodesic_updown_single_row_pass_ptr_kernel<<<blockCountUpDown, THREAD_COUNT>>>(
            image.data_ptr<float>(),
            distance.data_ptr<float>(),
            l_grad,
            l_eucl,
            direction,
            channel,
            height,
            width);
    #else
        AT_DISPATCH_FLOATING_TYPES(image.type(), "geodesic_updown_single_row_pass_kernel", ([&]
            { geodesic_updown_single_row_pass_kernel<scalar_t><<<blockCountUpDown, THREAD_COUNT>>>(
                image.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                distance.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                l_grad,
                l_eucl,
                direction); 
            }));
    #endif
    
}

torch::Tensor generalised_geodesic2d_cuda(
    const torch::Tensor &image, 
    const torch::Tensor &mask, 
    const float &v, 
    const float &l_grad, 
    const float &l_eucl, 
    const int &iterations
    )
{
    int device = image.get_device();
    // std::cout << "Running with CUDA Device: " << device << std::endl;
    c10::cuda::CUDAGuard device_guard(device);
    
    torch::Tensor image_local = image.clone();
    torch::Tensor distance = v * mask.clone();

    // iteratively run the distance transform
    for (int itr = 0; itr < iterations; itr++)
    {
        image_local = image_local.contiguous();
        distance = distance.contiguous();

        // top-bottom - width*, height
        geodesic_updown_pass_cuda(image_local, distance, l_grad, l_eucl);

        // left-right - height*, width
        image_local = image_local.transpose(2, 3);
        distance = distance.transpose(2, 3);

        image_local = image_local.contiguous();
        distance = distance.contiguous();
        geodesic_updown_pass_cuda(image_local, distance, l_grad, l_eucl);

        // tranpose back to original - width, height
        image_local = image_local.transpose(2, 3);
        distance = distance.transpose(2, 3);

        // * indicates the current direction of pass
    }
    return distance;
}


torch::Tensor pairwise_geodesic2d_cuda(
    const torch::Tensor &image,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations)
{
    int device = image.get_device();
    c10::cuda::CUDAGuard device_guard(device);

    const int height = image.size(2);
    const int width = image.size(3);
    const int num_pixels = height * width;

    // 预分配最终的成对距离矩阵
    auto options = image.options();
    torch::Tensor pairwise_matrix = torch::zeros({num_pixels, num_pixels}, options);

    // 创建一个掩码张量，将当前像素设置为唯一的种子点
    torch::Tensor mask = torch::ones({1,1,height, width}, options);

    // 为每个像素作为种子点进行迭代
    for (int i = 0; i < num_pixels; ++i)
    {
        // 将一维索引转换为二维坐标
        int seed_h = i / width;
        int seed_w = i % width;

        // 设置种子点值为0，其他为1
        mask.index_put_({0, 0, seed_h, seed_w}, 0.0);

        //torch::Tensor single_source_distances = torch::zeros({1,1,height, width}, options);
        torch::Tensor single_source_distances = generalised_geodesic2d_cuda(image, mask, v, l_grad, l_eucl, iterations);

        // 将 HxW 的距离图展平为 1D 向量，并存入结果矩阵的对应行
        pairwise_matrix.select(0, i).copy_(single_source_distances.flatten());
        mask.index_put_({0, 0, seed_h, seed_w}, 1.0);
    }

    return pairwise_matrix;
}

template <typename scalar_t>
__global__ void geodesic_frontback_single_plane_pass_kernel(
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> image_ptr,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> distance_ptr,
    const float l_grad,
    const float l_eucl,
    const int direction)
{
    const int channel = image_ptr.size(1);
    const int depth = image_ptr.size(2);
    const int height = image_ptr.size(3);
    const int width = image_ptr.size(4);

    int kernelW = blockIdx.x * blockDim.x + threadIdx.x;
    int kernelH = blockIdx.y * blockDim.y + threadIdx.y;

    int z = (direction == 1) ? 1 : depth - 2;

    // if outside, then skip distance calculation - dont use the thread
    if (kernelH >= 0 && kernelH < height && kernelW >= 0 && kernelW < width)
    {
        while (z >= 0 && z < depth)
        {
            int prevZ = z - direction;
            if (prevZ < 0 || prevZ >= depth)
            {
                // go to next plane
                z += direction;

                // read outside bounds, skip
                continue;
            }
            float l_dist, cur_dist;
            float new_dist = distance_ptr[0][0][z][kernelH][kernelW];

            for (int h_i = 0; h_i < 3; h_i++)
            {
                for (int w_i = 0; w_i < 3; w_i++)
                {
                    const int kernelH_ind = kernelH + h_i - 1;
                    const int kernelW_ind = kernelW + w_i - 1;

                    if (kernelH_ind >= 0 && \
                        kernelH_ind < height && \
                        kernelW_ind >= 0 && \
                        kernelW_ind < width)
                    {
                        l_dist = 0.0;
                        if (channel == 1)
                        {
                            l_dist = l1distance_cuda(
                                        image_ptr[0][0][z][kernelH][kernelW], 
                                        image_ptr[0][0][prevZ][kernelH_ind][kernelW_ind]
                                        );
                        }
                        else
                        {
                            for (int c_i = 0; c_i < channel; c_i++)
                            {
                                l_dist += l1distance_cuda(
                                    image_ptr[0][c_i][z][kernelH][kernelW], 
                                    image_ptr[0][c_i][prevZ][kernelH_ind][kernelW_ind]
                                    );
                            }
                        }
                        cur_dist = distance_ptr[0][0][prevZ][kernelH_ind][kernelW_ind] + \
                                    l_eucl * local_dist3d[h_i * 3 + w_i] + \
                                    l_grad * l_dist;

                        new_dist = std::min(new_dist, cur_dist);
                    }
                }
            }
            if (new_dist < distance_ptr[0][0][z][kernelH][kernelW])
            {
                distance_ptr[0][0][z][kernelH][kernelW] = new_dist;
            }

            // go to next plane
            z += direction;

            // synchronise write for all threads
            __syncthreads();
        }
    }
}

__global__ void geodesic_frontback_single_plane_pass_ptr_kernel(
    float *image_ptr,
    float *distance_ptr,
    const float l_grad,
    const float l_eucl,
    const int direction,
    const int channel,
    const int depth,
    const int height,
    const int width
    )
{

    int kernelW = blockIdx.x * blockDim.x + threadIdx.x;
    int kernelH = blockIdx.y * blockDim.y + threadIdx.y;

    int z = (direction == 1) ? 1 : depth - 2;

    // if outside, then skip distance calculation - dont use the thread
    if (kernelH >= 0 && \
        kernelH < height && \
        kernelW >= 0 && \
        kernelW < width)
    {
        while (z >= 0 && z < depth)
        {
            int prevZ = z - direction;
            if (prevZ < 0 || prevZ >= depth)
            {
                // go to next plane
                z += direction;
                
                // read outside bounds, skip
                continue;
            }
            float l_dist, cur_dist;
            float new_dist = distance_ptr[z * height * width + kernelH * width + kernelW];

            for (int h_i = 0; h_i < 3; h_i++)
            {
                for (int w_i = 0; w_i < 3; w_i++)
                {
                    const int kernelH_ind = kernelH + h_i - 1;
                    const int kernelW_ind = kernelW + w_i - 1;

                    if (kernelH_ind >= 0 && \
                        kernelH_ind < height && \
                        kernelW_ind >= 0 && \
                        kernelW_ind < width)
                    {
                        l_dist = 0.0;
                        if (channel == 1)
                        {
                            l_dist = l1distance_cuda(
                                        image_ptr[z * height * width + kernelH * width + kernelW], 
                                        image_ptr[prevZ * height * width + kernelH_ind * width + kernelW_ind]
                                        );
                        }
                        else
                        {
                            for (int c_i = 0; c_i < channel; c_i++)
                            {
                                l_dist += l1distance_cuda(
                                            image_ptr[c_i * depth * height * width + z * height * width + kernelH * width + kernelW], 
                                            image_ptr[c_i * depth * height * width + prevZ * height * width + kernelH_ind * width + kernelW_ind]);
                            }
                        }
                        cur_dist = distance_ptr[prevZ * height * width + kernelH_ind * width + kernelW_ind] + \
                                    l_eucl * local_dist3d[h_i * 3 + w_i] + \
                                    l_grad * l_dist;
                        new_dist = std::min(new_dist, cur_dist);
                    }
                }
            }
            if (new_dist < distance_ptr[z * height * width + kernelH * width + kernelW])
            {
                distance_ptr[z * height * width + kernelH * width + kernelW] = new_dist;
            }

            // go to next plane
            z += direction;

            // synchronise write for all threads
            __syncthreads();
        }
    }
}

void geodesic_frontback_pass_cuda(
    const torch::Tensor &image, 
    torch::Tensor &distance, 
    const std::vector<float> &spacing, 
    const float &l_grad, 
    const float &l_eucl
    )
{
    // batch, channel, depth, height, width
    const int channel = image.size(1);
    const int depth = image.size(2);
    const int height = image.size(3);
    const int width = image.size(4);

    // convert allowed number of threads into a 2D grid
    // helps if the THREAD_COUNT is N*N already
    const int THREAD_COUNT_2D = sqrt(THREAD_COUNT);
    int blockCountUpDown = (width + 1) / THREAD_COUNT_2D + 1;
    int blockCountLeftRight = (height + 1) / THREAD_COUNT_2D + 1;

    // pre-calculate local distances based on spacing
    float local_dist[3 * 3];
    for (int h_i = 0; h_i < 3; h_i++)
    {
        for (int w_i = 0; w_i < 3; w_i++)
        {
            float ld = spacing[0];
            ld += float(std::abs(h_i - 1)) * spacing[1];
            ld += float(std::abs(w_i - 1)) * spacing[2];

            local_dist[h_i * 3 + w_i] = sqrt(ld);
        }
    }
    // copy local distances to GPU __constant__ memory
    cudaMemcpyToSymbol(local_dist3d, local_dist, sizeof(float) * 3 * 3);

    dim3 dimGrid(blockCountUpDown, blockCountLeftRight);
    dim3 dimBlock(THREAD_COUNT_2D, THREAD_COUNT_2D);
    // Kernel<<<dimGrid, dimBlock>>>( arg1, arg2, arg2);

    // direction variable used to indicate read from previous (+1) or next (-1) plane
    int direction;

    // front-back
    direction = +1;
    #if USE_PTR
        geodesic_frontback_single_plane_pass_ptr_kernel<<<dimGrid, dimBlock>>>(
            image.data_ptr<float>(),
            distance.data_ptr<float>(),
            l_grad,
            l_eucl,
            direction,
            channel,
            depth,
            height,
            width);
    #else
        AT_DISPATCH_FLOATING_TYPES(image.type(), "geodesic_frontback_single_plane_pass_kernel", ([&]
            { geodesic_frontback_single_plane_pass_kernel<scalar_t><<<dimGrid, dimBlock>>>(
                image.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                distance.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                l_grad,
                l_eucl,
                direction); 
            }));
    #endif

    // back-front
    direction = -1;
    #if USE_PTR
        geodesic_frontback_single_plane_pass_ptr_kernel<<<dimGrid, dimBlock>>>(
            image.data_ptr<float>(),
            distance.data_ptr<float>(),
            l_grad,
            l_eucl,
            direction,
            channel,
            depth,
            height,
            width);
    #else
        AT_DISPATCH_FLOATING_TYPES(image.type(), "geodesic_frontback_single_plane_pass_kernel", ([&]
            { geodesic_frontback_single_plane_pass_kernel<scalar_t><<<dimGrid, dimBlock>>>(
                image.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                distance.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                l_grad,
                l_eucl,
                direction); 
            }));
    #endif
}

torch::Tensor generalised_geodesic3d_cuda(
    const torch::Tensor &image, 
    const torch::Tensor &mask, 
    std::vector<float> spacing, 
    const float &v, 
    const float &l_grad, 
    const float &l_eucl, 
    const int &iterations
    )
{
    int device = image.get_device();
    // std::cout << "Running with CUDA Device: " << device << std::endl;
    c10::cuda::CUDAGuard device_guard(device);

    // square spacing with transform
    std::transform(spacing.begin(), spacing.end(), spacing.begin(), spacing.begin(), std::multiplies<float>());
    
    torch::Tensor image_local = image.clone();
    torch::Tensor distance = v * mask.clone();

    // iteratively run the distance transform
    for (int itr = 0; itr < iterations; itr++)
    {
        image_local = image_local.contiguous();
        distance = distance.contiguous();

        // front-back - depth*, height, width
        geodesic_frontback_pass_cuda(image_local, distance, spacing, l_grad, l_eucl);

        // top-bottom - height*, depth, width
        image_local = torch::transpose(image_local, 3, 2);
        distance = torch::transpose(distance, 3, 2);

        image_local = image_local.contiguous();
        distance = distance.contiguous();
        geodesic_frontback_pass_cuda(
            image_local,
            distance,
            {spacing[1], spacing[0], spacing[2]},
            l_grad,
            l_eucl
            );

        // transpose back to original depth, height, width
        image_local = torch::transpose(image_local, 3, 2);
        distance = torch::transpose(distance, 3, 2);

        // left-right - width*, height, depth
        image_local = torch::transpose(image_local, 4, 2);
        distance = torch::transpose(distance, 4, 2);

        image_local = image_local.contiguous();
        distance = distance.contiguous();
        geodesic_frontback_pass_cuda(
            image_local,
            distance,
            {spacing[2], spacing[1], spacing[0]},
            l_grad,
            l_eucl
            );

        // transpose back to original depth, height, width
        image_local = torch::transpose(image_local, 4, 2);
        distance = torch::transpose(distance, 4, 2);

        // * indicates the current direction of pass
    }

    return distance;
}

// image: (1, C, H, W) float32 contiguous
// distance: (1,1,H,W) float32
// low/high: (C,H,W) float32 buffers (per-channel)
// direction: +1 for top->bottom, -1 for bottom->top
__global__ void mbd_updown_single_row_pass_ptr_kernel(
    const float* __restrict__ image_ptr,
    float* __restrict__ distance_ptr,
    float* __restrict__ low_ptr,
    float* __restrict__ high_ptr,
    const int direction,
    const int channel,
    const int height,
    const int width
){
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = (direction == 1)? 1 : height - 2;

    if (w >= width) return;

    while (h >= 0 && h < height) {
        int ph = h - direction; // previous row index in the scanning direction
        if (ph < 0 || ph >= height) {
            // advance to next row if out of bounds
            h += direction;
            __syncthreads();
            continue;
        }

        // Current best values at (h,w)
        int cur_idx = h * width + w;
        float best_cost = distance_ptr[cur_idx];

        // Load current per-channel low/high as fallback (if no improvement)
        // Note: low/high arrays are laid out [C, H, W]
        // idx(c,h,w) = c*H*W + h*W + w
        // We'll keep temporary best low/high in registers (per-channel)
        // If channel is large, we loop twice: once to compute candidate, once to commit.
        // Here we store only when improving to avoid extra memory traffic.
        // We will compute candidates per neighbor and keep the best seen.

        // Variables to remember which neighbor produced the best candidate
        int best_neighbor_w = -999; // -1,0,+1 offset encoded by absolute w index
        // We won't store per-channel best L/U here to save registers; we recompute once when applying.

        // Try 3 neighbors from the previous row: (ph, w-1), (ph, w), (ph, w+1)
        for (int wi = -1; wi <= 1; ++wi) {
            int nw = w + wi;
            if (nw < 0 || nw >= width) continue;

            int n_idx = ph * width + nw;

            // For candidate cost, we need to combine neighbor's low/high with current pixel intensity
            // Compute D' = sum_c ( max(U_n[c], I[h,w,c]) - min(L_n[c], I[h,w,c]) )
            float cand_cost = 0.f;

            for (int c = 0; c < channel; ++c) {
                int off_cur = c * (height * width) + cur_idx;
                int off_nei = c * (height * width) + n_idx;
                float Icur = image_ptr[off_cur];
                float Lnei = low_ptr[off_nei];
                float Unei = high_ptr[off_nei];
                float Lc = fminf(Lnei, Icur);
                float Uc = fmaxf(Unei, Icur);
                cand_cost += (Uc - Lc);
            }

            if (cand_cost < best_cost) {
                best_cost = cand_cost;
                best_neighbor_w = nw;
            }
        }

        if (best_cost < distance_ptr[cur_idx]) {
            // Commit update: write distance, and update low/high at (h,w) using the neighbor that gave best_cost
            distance_ptr[cur_idx] = best_cost;

            int n_idx = ph * width + best_neighbor_w;
            for (int c = 0; c < channel; ++c) {
                int off_cur = c * (height * width) + cur_idx;
                int off_nei = c * (height * width) + n_idx;
                float Icur = image_ptr[off_cur];
                float Lnei = low_ptr[off_nei];
                float Unei = high_ptr[off_nei];
                float Lc = fminf(Lnei, Icur);
                float Uc = fmaxf(Unei, Icur);
                low_ptr[off_cur]  = Lc;
                high_ptr[off_cur] = Uc;
            }
        }

        h += direction;
        __syncthreads();
    }
}

static inline void mbd_updown_pass_cuda(
    const torch::Tensor& image,
    torch::Tensor& distance,
    torch::Tensor& low,
    torch::Tensor& high
){
    const int channel = image.size(1);
    const int height  = image.size(2);
    const int width   = image.size(3);

    int blockCount = (width + THREAD_COUNT - 1) / THREAD_COUNT;

    // top -> bottom
    mbd_updown_single_row_pass_ptr_kernel<<<blockCount, THREAD_COUNT>>>(
        image.data_ptr<float>(),
        distance.data_ptr<float>(),
        low.data_ptr<float>(),
        high.data_ptr<float>(),
        +1, channel, height, width
    );

    // bottom -> top
    mbd_updown_single_row_pass_ptr_kernel<<<blockCount, THREAD_COUNT>>>(
        image.data_ptr<float>(),
        distance.data_ptr<float>(),
        low.data_ptr<float>(),
        high.data_ptr<float>(),
        -1, channel, height, width
    );
}

// Public API: single-source MBD with given seeds mask (0 at seeds, 1 elsewhere)
// Returns (1,1,H,W) distance tensor
torch::Tensor mbd2d_cuda(
    const torch::Tensor& image,   // (1,C,H,W) float32, on CUDA
    const torch::Tensor& mask,    // (1,1,H,W) float32, 0 at seeds, 1 elsewhere
    const float& v,               // initial large value, e.g., 1e6
    const int& iterations         // number of full 4-direction iterations
){
    TORCH_CHECK(image.is_cuda(), "image must be CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be CUDA tensor");
    TORCH_CHECK(image.dim() == 4 && image.size(0) == 1, "image must be (1,C,H,W)");
    TORCH_CHECK(mask.dim() == 4 && mask.sizes() == torch::IntArrayRef({1,1,image.size(2), image.size(3)}),
                "mask must be (1,1,H,W) matching image");

    c10::cuda::CUDAGuard device_guard(image.get_device());
    auto optsF = image.options().dtype(torch::kFloat32);

    const int C = image.size(1);
    const int H = image.size(2);
    const int W = image.size(3);

    // Distance init: seeds=0, others=v
    torch::Tensor distance = v * mask.clone();

    // Low/High init: start with per-pixel intensity (so extending a path simply tightens/loosens)
    // Shape (C,H,W)
    torch::Tensor img_chw = image[0]; // (C,H,W)
    torch::Tensor low  = img_chw.clone();
    torch::Tensor high = img_chw.clone();

    for (int itr = 0; itr < iterations; ++itr) {
        // Vertical passes
        mbd_updown_pass_cuda(image, distance, low, high);

        // Transpose to reuse the same kernel as "vertical" over width
        // We need to transpose image (C,H,W) -> (C,W,H) and distance/low/high accordingly.
        // Work in (1,C,H,W) for image and (C,H,W) for low/high to keep pointers consistent.
        torch::Tensor image_t   = image.transpose(2, 3).contiguous();
        torch::Tensor distance_t= distance.transpose(2, 3).contiguous();
        torch::Tensor low_t     = low.transpose(1, 2).contiguous();   // (C,W,H)
        torch::Tensor high_t    = high.transpose(1, 2).contiguous();  // (C,W,H)

        mbd_updown_pass_cuda(image_t, distance_t, low_t, high_t);

        // transpose back
        distance = distance_t.transpose(2, 3).contiguous();
        low      = low_t.transpose(1, 2).contiguous();
        high     = high_t.transpose(1, 2).contiguous();
    }

    return distance;
}