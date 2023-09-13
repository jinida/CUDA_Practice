#pragma once

#define NUM_THREAD_IN_BLOCK (blockDim.x * blockDim.y * blockDim.z)
#define TID_IN_BLOCK (threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x)

#define TID_1D (blockIdx.x * NUM_THREAD_IN_BLOCK) + TID_IN_BLOCK
#define TID_2D (blockIdx.y * NUM_THREAD_IN_BLOCK) + TID_1D

#define TID (blockIdx.z * NUM_THREAD_IN_BLOCK) + TID_2D

#define TID_X (blockDim.x * blockIdx.x + threadIdx.x)
#define TID_Y (blockDim.y * blockIdx.y + threadIdx.y)
#define TID_Z (blockDim.z * blockIdx.z + threadIdx.z)