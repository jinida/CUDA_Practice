#if ! defined(__CUDACC__) 
#define __host__ 
#define __device__ 
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <cmath>
#include <numbers>
#include "Timer.h"

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include "device_launch_parameters.h"
#endif

template <typename T>
T div_up(const T lhs, const T rhs)
{
	return (lhs + rhs - 1) / rhs;
}

void addWithCuda(float* dst, const float* srcA, const float* srcB, int num);
void axpyWithCuda(float* dst, const float a, const float* srcX, const float* srcY, int num);
void addMatWithCuda(float* dst, const float* srcA, const float* srcB, const int width, const int height);
void addMat2DWithCuda(float* dst, const float* srcA, const float* srcB, const int width, const int height);
void addMat3DWithCuda(float* dst, const float* srcA, const float* srcB, const int depth, const int width, const int height);
void adjDiffWithCuda(float* dst, const float* src, int num);
void transposeMatrixWithCuda(float* dst, const float* src, const int width, const int height);
void multiplyMatWithCuda(float* dst, const float* srcA, const float* srcB, const int widthA, const int heightA, const int widthB, const int heightB);
void generalMatMulWithCuda(float* dst, const float alpha, const float beta, const float* srcA, const float* srcB, const float* srcC, const int widthA, const int heightA, const int widthB, const int heightB);
__global__ void addVector(float* dst, const float* srcA, const float* srcB, int num);
__global__ void axpyVector(float* dst, const float a, const float* srcX, const float* srcY, int num);
__global__ void addMatrix(float* dst, const float* srcA, const float* srcB, const int width, const int height);
__global__ void addMatrix2D(float* dst, const float* srcA, const float* srcB, int nRow, int nCol, size_t dev_pitch);
__global__ void addMatrix3D(float* dst, const float* srcA, const float* srcB, int nCh, int nRow, int nCol, size_t dev_pitch);
__global__ void adjDiffVector(float* dst, const float* src, int num);
__global__ void adjDiffVectorWithShared(float* dst, const float* src, int num);
__global__ void transposeMatrix(float* dst, const float* src, int nRow, int nCol, size_t srcPitch, size_t dstPitch);
__global__ void transposeMatrixOptim(float* dst, const float* src, int nRow, int nCol, size_t srcPitch, size_t dstPitch);
__global__ void transposeMatrixOptimNBC(float* dst, const float* src, int nRow, int nCol, size_t srcPitch, size_t dstPitch);
__global__ void multiplyMat(float* dst, const float* srcA, const float* srcB, const int nColA, const int nRowA, const int nColB, const int nRowB, size_t srcAPitch, size_t srcBPitch, size_t dstPitch);
__global__ void generalMatMulgeneralMatMul(float* dst, const float alpha, const float beta, const float* srcA, const float* srcB, const float* srcC, const int nRowA, const int nColA, const int nColB, size_t pitchA, size_t pitchB);