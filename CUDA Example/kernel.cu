#include "kernel.cuh"
#include "cuda_definition.cuh"

__global__ void addVector(float* dst, const float* srcA, const float* srcB, int num)
{
    int i = TID;
    
    if (i < num)
    {
        dst[i] = srcA[i] + srcB[i];
    }
}

__global__ void axpyVector(float* dst, const float a, const float* srcX, const float* srcY, int num)
{
	int i = TID;
	if (i < num)
	{
		//dst[i] = a * srcX[i] + srcY[i];
		dst[i] = fmaf(a, srcX[i], srcY[i]);
	}
}

__global__ void addMatrix(float* dst, const float* srcA, const float* srcB, int nRow, int nCol)
{
	int col = TID_X;
	int row = TID_Y;

	if (col < nCol && row < nRow)
	{
		int i = nCol * row + col;
		dst[i] = srcA[i] + srcB[i];
	}
}

__global__ void addSwapMatrix(float* dst, const float* srcA, const float* srcB, int nRow, int nCol)
{
	int col = TID_Y;
	int row = TID_X;

	if (col < nCol && row < nRow)
	{
		int i = nCol * row + col;
		dst[i] = srcA[i] + srcB[i];
	}
}

__global__ void addMatrix2D(float* dst, const float* srcA, const float* srcB, int nRow, int nCol, size_t dev_pitch)
{
	int col = TID_X;
	int row = TID_Y;

	if (col < nCol && row < nRow)
	{
		unsigned offset = row * dev_pitch + col * sizeof(float);
		*((float*)((char*)dst + offset)) = *((const float*)((const char*)srcA + offset))
			+ *((const float*)((const char*)srcB + offset));
	}
}

__global__ void addMatrix3D(float* dst, const float* srcA, const float* srcB, int nCh, int nRow, int nCol, size_t dev_pitch)
{
	int col = TID_X;
	int row = TID_Y;
	int channel = TID_Z;

	if (col < nCol && row < nRow && channel < nCh)
	{
		unsigned offset = (channel * nRow + row) * dev_pitch + col * sizeof(float);
		*((float*)((char*)dst + offset)) = *((const float*)((const char*)srcA + offset))
			+ *((const float*)((const char*)srcB + offset));
	}
}

__global__ void adjDiffVector(float* dst, const float* src, int num)
{
	int i = TID_X;
	if (i < num)
	{
		if (i == 0)
		{
			dst[i] = src[i];
		}
		else
		{
			dst[i] = src[i] - src[i - 1];
		}
	}
}

__global__ void adjDiffVectorWithShared(float* dst, const float* src, int num)
{
	__shared__ float sData[1024];
	int i = TID_X;
	if (i < num)
	{
		unsigned tx = threadIdx.x;
		sData[tx] = src[i];
		__syncthreads();
		if (tx > 0)
		{
			dst[i] = sData[tx] - sData[tx - 1];
		}
		else if (i > 0)
		{
			dst[i] = sData[tx] - src[i - 1];
		}
		else
		{
			dst[i] = sData[tx];
		}
	}
}

__global__ void transposeMatrix(float* dst, const float* src, int nRow, int nCol, size_t srcPitch, size_t dstPitch)
{
	int col = TID_X;
	int row = TID_Y;

	__shared__ float sData[32][32];
	if (col < nCol && row < nRow)
	{
		unsigned idxSrc = row * srcPitch / sizeof(float) + col;
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		sData[ty][tx] = src[idxSrc];
		__syncthreads();

		unsigned idxDst = col * dstPitch / sizeof(float) + row;
		dst[idxDst] = sData[ty][tx];
	}
}

__global__ void transposeMatrixOptim(float* dst, const float* src, int nRow, int nCol, size_t srcPitch, size_t dstPitch)
{
	int col = TID_X;
	int row = TID_Y;

	__shared__ float sData[32][32];
	if (col < nCol && row < nRow)
	{
		unsigned idxSrc = row * srcPitch / sizeof(float) + col;
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		sData[ty][tx] = src[idxSrc];
		__syncthreads();

		unsigned idxDst = (blockDim.x + blockIdx.x + threadIdx.y) * dstPitch / sizeof(float) + (blockDim.y + blockIdx.y + threadIdx.x);
		dst[idxDst] = sData[tx][ty];
	}
}

__global__ void transposeMatrixOptimNBC(float* dst, const float* src, int nRow, int nCol, size_t srcPitch, size_t dstPitch)
{
	int col = TID_X;
	int row = TID_Y;

	__shared__ float sData[32][33];
	if (col < nCol && row < nRow)
	{
		unsigned idxSrc = row * srcPitch / sizeof(float) + col;
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		sData[ty][tx] = src[idxSrc];
		__syncthreads();

		unsigned idxDst = (blockDim.x + blockIdx.x + threadIdx.y) * dstPitch / sizeof(float) + (blockDim.y + blockIdx.y + threadIdx.x);
		dst[idxDst] = sData[tx][ty];
	}
}

__global__ void multiplyMat(float* dst, const float* srcA, const float* srcB, const int nColA, const int nRowA, const int nColB, const int nRowB, size_t srcAPitch, size_t srcBPitch, size_t dstPitch)
{
	int col = TID_X;
	int row = TID_Y;
	
	__shared__ float sDataA[32][32];
	__shared__ float sDataB[32][32];

	float sum = 0.f;
	int tileCols = static_cast<int> (blockDim.x);
	int nTile = (nRowA + tileCols - 1) / tileCols;
	int rem = nRowA;

	for (int tile = 0; tile < nTile; ++tile)
	{
		unsigned nElems = static_cast<unsigned>(__min(tileCols, rem));
		rem -= tileCols;

		if (row < nRowA && threadIdx.x < nElems)
		{
			int idxSrcA = row * srcAPitch + (tile * tileCols + threadIdx.x);
			sDataA[threadIdx.y][threadIdx.x] = srcA[idxSrcA];
		}
			
		if (col < nColA && threadIdx.y < nElems)
		{
			int idxSrcB = (tile * tileCols + threadIdx.y) * srcBPitch + col;
			sDataB[threadIdx.y][threadIdx.x] = srcB[idxSrcB];
		}

		__syncthreads();
	
		if (row < nRowA && col < nColB)
		{
			for (unsigned k = 0; k < nElems; ++k)
			{
				sum += (sDataA[threadIdx.y][k] * sDataB[k][threadIdx.x]);
			}
		}
		__syncthreads();
	}
	

	if (row < nRowA && col < nColB)
	{
		unsigned int idxDst = row * dstPitch + col;
		dst[idxDst] = sum;
	}
}

__global__ void generalMatMul(float* dst, const float alpha, const float beta,const float* srcA, const float* srcB, const float* srcC, const int nRowA, const int nColA, const int nColB, size_t pitchA, size_t pitchB)
{
	int col = TID_X;
	int row = TID_Y;

	__shared__ float sDataA[32][32];
	__shared__ float sDataB[32][32];

	float sum = 0.f;
	int tileCols = static_cast<int> (blockDim.x);
	int nTile = (nRowA + tileCols - 1) / tileCols;
	int rem = nRowA;

	for (int tile = 0; tile < nTile; ++tile)
	{
		unsigned nElems = static_cast<unsigned>(__min(tileCols, rem));
		rem -= tileCols;

		if (row < nRowA && threadIdx.x < nElems)
		{
			int idxSrcA = row * pitchA + (tile * tileCols + threadIdx.x);
			sDataA[threadIdx.y][threadIdx.x] = srcA[idxSrcA];
		}

		if (col < nColA && threadIdx.y < nElems)
		{
			int idxSrcB = (tile * tileCols + threadIdx.y) * pitchB + col;
			sDataB[threadIdx.y][threadIdx.x] = srcB[idxSrcB];
		}

		__syncthreads();

		if (row < nRowA && col < nColB)
		{
			for (unsigned k = 0; k < nElems; ++k)
			{
				sum += (sDataA[threadIdx.y][k] * sDataB[k][threadIdx.x]);
			}
		}
		__syncthreads();
	}


	if (row < nRowA && col < nColB)
	{
		unsigned int idxDst = row * pitchB + col;
		dst[idxDst] = fmaf(alpha, sum, beta * srcC[idxDst]);
	}
}

void addWithCuda(float* dst, const float* srcA, const float* srcB, int num)
{
	float* dev_srcA;
	float* dev_srcB;
	float* dev_dst;
	Timer t("cuda");

	t.start(0);
	cudaMalloc(&dev_srcA, sizeof(float) * num);
	cudaMalloc(&dev_srcB, sizeof(float) * num);
	cudaMalloc(&dev_dst, sizeof(float) * num);
	t.end(0);
	t.setTimerContents(0, "cuda Memory Alloc");

	t.start(1);
	cudaMemcpy(dev_srcA, srcA, sizeof(float) * num, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_srcB, srcB, sizeof(float) * num, cudaMemcpyHostToDevice);
	t.end(1);
	t.setTimerContents(1, "cuda Memory Copy (host to dev)");

	t.start(2);
	addVector<<< std::ceil(num / 1024) , 1024>>>(dev_dst, dev_srcA, dev_srcB, num);
	cudaDeviceSynchronize();
	t.end(2);
	t.setTimerContents(2, "excute addVector");
	
	t.start(3);
	cudaMemcpy(dst, dev_dst, sizeof(float) * num, cudaMemcpyDeviceToHost);
	t.end(3);
	t.setTimerContents(3, "cuda Memory Copy (dev to host)");

	t.start(4);
	cudaFree(dev_srcA);
	cudaFree(dev_srcB);
	cudaFree(dev_dst);
	t.end(4);
	t.setTimerContents(4, "cuda Free");

	t.printReport();
	t.release();
}

void axpyWithCuda(float* dst, const float a, const float* srcX, const float* srcY, int num)
{
	float* dev_srcX;
	float* dev_srcY;
	float* dev_dst;
	Timer t("cuda");

	t.start(0);
	cudaMalloc(&dev_srcX, sizeof(float) * num);
	cudaMalloc(&dev_srcY, sizeof(float) * num);
	cudaMalloc(&dev_dst, sizeof(float) * num);
	t.end(0);
	t.setTimerContents(0, "cuda Memory Alloc");

	t.start(1);
	cudaMemcpy(dev_srcX, srcX, sizeof(float) * num, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_srcY, srcY, sizeof(float) * num, cudaMemcpyHostToDevice);
	t.end(1);
	t.setTimerContents(1, "cuda Memory Copy (host to dev)");

	t.start(2);
	axpyVector << < std::ceil(num / 1024), 1024 >> > (dev_dst, a, dev_srcX, dev_srcY, num);
	cudaDeviceSynchronize();
	t.end(2);
	t.setTimerContents(2, "excute axpyVector");

	t.start(3);
	cudaMemcpy(dst, dev_dst, sizeof(float) * num, cudaMemcpyDeviceToHost);
	t.end(3);
	t.setTimerContents(3, "cuda Memory Copy (dev to host)");

	t.start(4);
	cudaFree(dev_srcX);
	cudaFree(dev_srcY);
	cudaFree(dev_dst);
	t.end(4);
	t.setTimerContents(4, "cuda Free");

	t.printReport();
	t.release();
}


void addMatWithCuda(float* dst, const float* srcA, const float* srcB, const int width, const int height)
{
	float* dev_srcA;
	float* dev_srcB;
	float* dev_dst;
	Timer t("cuda");
	const int num = width * height;

	t.start(0);
	cudaMalloc(&dev_srcA, sizeof(float) * num);
	cudaMalloc(&dev_srcB, sizeof(float) * num);
	cudaMalloc(&dev_dst, sizeof(float) * num);
	t.end(0);
	t.setTimerContents(0, "cuda Memory Alloc");

	t.start(1);
	cudaMemcpy(dev_srcA, srcA, sizeof(float) * num, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_srcB, srcB, sizeof(float) * num, cudaMemcpyHostToDevice);
	t.end(1);
	t.setTimerContents(1, "cuda Memory Copy (host to dev)");

	t.start(2);
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(div_up(static_cast<unsigned int>(width), dimBlock.x), div_up(static_cast<unsigned int>(height), dimBlock.y), 1);
	addMatrix <<< dimGrid, dimBlock >>> (dev_dst, dev_srcA, dev_srcB, width, height);
	cudaDeviceSynchronize();
	t.end(2);
	t.setTimerContents(2, "excute addMatrix");

	t.start(3);
	addSwapMatrix << < dimGrid, dimBlock >> > (dev_dst, dev_srcA, dev_srcB, width, height);
	cudaDeviceSynchronize();
	t.end(3);
	t.setTimerContents(3, "excute addSwapMatrix");

	t.start(4);
	cudaMemcpy(dst, dev_dst, sizeof(float) * num, cudaMemcpyDeviceToHost);
	t.end(4);
	t.setTimerContents(4, "cuda Memory Copy (dev to host)");

	t.start(5);
	cudaFree(dev_srcA);
	cudaFree(dev_srcB);
	cudaFree(dev_dst);
	t.end(5);
	t.setTimerContents(5, "cuda Free");

	t.printReport();
	t.release();
}

void addMat2DWithCuda(float* dst, const float* srcA, const float* srcB, const int width, const int height)
{
	float* dev_srcA;
	float* dev_srcB;
	float* dev_dst;
	Timer t("cuda");
	const int num = width * height;
	
	size_t dev_pitch, host_pitch = width * sizeof(float);
	t.start(0);
	cudaMallocPitch(&dev_srcA, &dev_pitch, sizeof(float) * width, height);
	cudaMallocPitch(&dev_srcB, &dev_pitch, sizeof(float) * width, height);
	cudaMallocPitch(&dev_dst, &dev_pitch, sizeof(float) * width, height);
	t.end(0);
	t.setTimerContents(0, "cuda Memory Alloc Pitch");

	t.start(1);
	cudaMemcpy2D(dev_srcA, dev_pitch, srcA, host_pitch, width * sizeof(float), height, cudaMemcpyHostToDevice);
	cudaMemcpy2D(dev_srcB, dev_pitch, srcB, host_pitch, width * sizeof(float), height, cudaMemcpyHostToDevice);
	t.end(1);
	t.setTimerContents(1, "cuda Memory Copy 2D (host to dev)");

	t.start(2);
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(div_up(static_cast<unsigned int>(width), dimBlock.x), div_up(static_cast<unsigned int>(height), dimBlock.y), 1);
	addMatrix2D <<< dimGrid, dimBlock >>> (dev_dst, dev_srcA, dev_srcB, width, height, dev_pitch);
	cudaDeviceSynchronize();
	t.end(2);
	t.setTimerContents(2, "excute addMatrix2D");

	t.start(4);
	cudaMemcpy2D(dst, host_pitch, dev_dst, dev_pitch, sizeof(float) * width, height, cudaMemcpyDeviceToHost);
	t.end(4);
	t.setTimerContents(4, "cuda Memory Copy 2D (dev to host)");

	t.start(5);
	cudaFree(dev_srcA);
	cudaFree(dev_srcB);
	cudaFree(dev_dst);
	t.end(5);
	t.setTimerContents(5, "cuda Free");

	t.printReport();
	t.release();
}

void addMat3DWithCuda(float* dst, const float* srcA, const float* srcB, const int depth, const int width, const int height)
{
	Timer t("cuda");
	const int num = depth * width * height;

	t.start(0);
	cudaPitchedPtr hostPitchedA = make_cudaPitchedPtr((void*)srcA, width * sizeof(float), width * sizeof(float), height);
	cudaPitchedPtr hostPitchedB = make_cudaPitchedPtr((void*)srcB, width * sizeof(float), width * sizeof(float), height);
	cudaPitchedPtr hostPitchedDst = make_cudaPitchedPtr(dst, width * sizeof(float), width * sizeof(float), height);
	cudaExtent extentInByte = make_cudaExtent(width * sizeof(float), height, depth);
	cudaPitchedPtr devPitchedA = { 0 }, devPitchedB = { 0 }, devPitchedDst = { 0 };
	cudaMalloc3D(&devPitchedA, extentInByte);
	cudaMalloc3D(&devPitchedB, extentInByte);
	cudaMalloc3D(&devPitchedDst, extentInByte);
	t.end(0);
	t.setTimerContents(0, "cuda Memory Alloc 3D");

	t.start(1);
	cudaPos defaultPos = make_cudaPos(0, 0, 0);
	cudaMemcpy3DParms paramA = { 0 }, paramB = { 0 };
	paramA.srcPos = defaultPos; paramA.srcPtr = hostPitchedA; paramA.dstPos = defaultPos;
	paramA.dstPtr = devPitchedA; paramA.extent = extentInByte; paramA.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&paramA);

	paramB.srcPos = defaultPos; paramB.srcPtr = hostPitchedB; paramB.dstPos = defaultPos;
	paramB.dstPtr = devPitchedB; paramB.extent = extentInByte; paramB.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&paramB);
	t.end(1);
	t.setTimerContents(1, "cuda Memory Copy 3D (host to dev)");

	t.start(2);
	dim3 dimBlock(8, 8, 8);
	dim3 dimGrid(div_up(static_cast<unsigned int>(width), dimBlock.x), div_up(static_cast<unsigned int>(height), dimBlock.y), div_up(static_cast<unsigned int>(depth), dimBlock.z));
	addMatrix3D << < dimGrid, dimBlock >> >((float*)devPitchedDst.ptr, (const float*)devPitchedA.ptr, (const float*)devPitchedB.ptr, depth, width, height, devPitchedA.pitch);
	cudaDeviceSynchronize();
	t.end(2);
	t.setTimerContents(2, "excute addMatrix3D");
	t.start(4);
	cudaMemcpy3DParms paramC = { 0 };
	paramC.srcPos = defaultPos; paramC.srcPtr = devPitchedDst; paramC.dstPos = defaultPos;
	paramC.dstPtr = hostPitchedDst; paramC.extent = extentInByte; paramC.kind = cudaMemcpyDeviceToHost;
	paramC.kind = cudaMemcpyDeviceToHost;
	cudaMemcpy3D(&paramC);
	t.end(4);
	t.setTimerContents(4, "cuda Memory Copy 3D (dev to host)");

	t.start(5);
	cudaFree(devPitchedA.ptr);
	cudaFree(devPitchedB.ptr);
	cudaFree(devPitchedDst.ptr);
	t.end(5);
	t.setTimerContents(5, "cuda Free");

	t.printReport();
	t.release();
}

void adjDiffWithCuda(float* dst, const float* src, int num)
{
	float* dev_src;
	float* dev_dst;
	Timer t("cuda");

	t.start(0);
	cudaMalloc(&dev_src, sizeof(float) * num);
	cudaMalloc(&dev_dst, sizeof(float) * num);
	t.end(0);
	t.setTimerContents(0, "cuda Memory Alloc");

	t.start(1);
	cudaMemcpy(dev_src, src, sizeof(float) * num, cudaMemcpyHostToDevice);
	t.end(1);
	t.setTimerContents(1, "cuda Memory Copy (host to dev)");

	t.start(2);
	adjDiffVector << < div_up(num, 1024), 1024 >> > (dev_dst, dev_src, num);
	cudaDeviceSynchronize();
	t.end(2);
	t.setTimerContents(2, "excute adjDiffVector");

	t.start(3);
	adjDiffVectorWithShared << < div_up(num, 1024), 1024 >> > (dev_dst, dev_src, num);
	cudaDeviceSynchronize();
	t.end(3);
	t.setTimerContents(3, "excute adjDiffVectorWithShared");

	t.start(4);
	cudaMemcpy(dst, dev_dst, sizeof(float) * num, cudaMemcpyDeviceToHost);
	t.end(4);
	t.setTimerContents(4, "cuda Memory Copy (dev to host)");

	t.start(5);
	cudaFree(dev_src);
	cudaFree(dev_dst);
	t.end(5);
	t.setTimerContents(5, "cuda Free");

	t.printReport();
	t.release();
}

void transposeMatrixWithCuda(float* dst, const float* src, const int width, const int height)
{
	float* dev_src;
	float* dev_dst;
	Timer t("cuda");
	const int num = width * height;

	size_t devSrcPitch, devDstPitch, hostSrcPitch = width * sizeof(float), hostDstPitch = height * sizeof(float);
	t.start(0);
	cudaMallocPitch(&dev_src, &devSrcPitch, sizeof(float) * width, height);
	cudaMallocPitch(&dev_dst, &devDstPitch, sizeof(float) * height, width);
		t.end(0);
	t.setTimerContents(0, "cuda Memory Alloc Pitch");

	t.start(1);
	cudaMemcpy2D(dev_src, devSrcPitch, src, hostSrcPitch, width * sizeof(float), height, cudaMemcpyHostToDevice);
	t.end(1);
	t.setTimerContents(1, "cuda Memory Copy 2D (host to dev)");

	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(div_up(static_cast<unsigned int>(width), dimBlock.x), div_up(static_cast<unsigned int>(height), dimBlock.y), 1);

	t.start(2);
	transposeMatrix << < dimGrid, dimBlock >> > (dev_dst, dev_src, width, height, devSrcPitch, devDstPitch);
	cudaDeviceSynchronize();
	t.end(2);

	t.start(3);
	transposeMatrixOptim << < dimGrid, dimBlock >> > (dev_dst, dev_src, width, height, devSrcPitch, devDstPitch);
	cudaDeviceSynchronize();
	t.end(3);
	t.setTimerContents(3, "excute transposeMatrixOptim");
	
	t.start(4);
	transposeMatrixOptimNBC << < dimGrid, dimBlock >> > (dev_dst, dev_src, width, height, devSrcPitch, devDstPitch);
	cudaDeviceSynchronize();
	t.end(4);
	t.setTimerContents(4, "excute transposeMatrixOptim No Bank Conflict");


	t.start(5);
	cudaMemcpy2D(dst, hostDstPitch, dev_dst, devDstPitch, height * sizeof(float), width, cudaMemcpyDeviceToHost);
	t.end(5);
	t.setTimerContents(5, "cuda Memory Copy 2D (dev to host)");


	t.setTimerContents(2, "excute transposeMatrix");
	t.start(6);
	cudaFree(dev_src);
	cudaFree(dev_dst);
	t.end(6);
	t.setTimerContents(6, "cuda Free");

	t.printReport();
	t.release();
}

void multiplyMatWithCuda(float* dst, const float* srcA, const float* srcB, const int widthA, const int heightA, const int widthB, const int heightB)
{
	float* dev_srcA;
	float* dev_srcB;
	float* dev_dst;
	Timer t("cuda");
	const int numA = widthA * heightA;

	size_t hostPitchA = widthA * sizeof(float), hostPitchB = widthB * sizeof(float);
	size_t devPitchA, devPitchB, devPitchDst;

	t.start(0);
	cudaMallocPitch(&dev_srcA, &devPitchA, sizeof(float) * widthA, heightA);
	cudaMallocPitch(&dev_srcB, &devPitchB, sizeof(float) * widthB, heightB);
	cudaMallocPitch(&dev_dst, &devPitchDst, sizeof(float) * widthB, heightA);
	t.end(0);
	t.setTimerContents(0, "cuda Memory Alloc Pitch");

	t.start(1);
	cudaMemcpy2D(dev_srcA, devPitchA, srcA, hostPitchA, widthA * sizeof(float), heightA, cudaMemcpyHostToDevice);
	cudaMemcpy2D(dev_srcB, devPitchB, srcB, hostPitchB, widthB * sizeof(float), heightB, cudaMemcpyHostToDevice);
	t.end(1);
	t.setTimerContents(1, "cuda Memory Copy 2D (host to dev)");
	size_t elemA, elemB, elemDst;
	elemA = devPitchA / sizeof(float);
	elemB = devPitchB / sizeof(float);
	elemDst = devPitchDst / sizeof(float);
	
	t.start(2);
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(div_up(static_cast<unsigned int>(widthB), dimBlock.x), div_up(static_cast<unsigned int>(heightA), dimBlock.y), 1);
	multiplyMat << < dimGrid, dimBlock >> > (dev_dst, dev_srcA, dev_srcB, widthA, heightA, widthB, heightB, elemA, elemB, elemDst);
	cudaDeviceSynchronize();
	t.end(2);
	t.setTimerContents(2, "excute Multiply Matrix2D");

	t.start(4);
	cudaMemcpy2D(dst, hostPitchB, dev_dst, devPitchDst, sizeof(float) * widthB, heightA, cudaMemcpyDeviceToHost);
	t.end(4);
	t.setTimerContents(4, "cuda Memory Copy 2D (dev to host)");

	t.start(5);
	cudaFree(dev_srcA);
	cudaFree(dev_srcB);
	cudaFree(dev_dst);
	t.end(5);
	t.setTimerContents(5, "cuda Free");

	t.printReport();
	t.release();

	/*
	Mutiply Matrix2D - (Matrix Size:  2048 x  2048)
	*** cuda Timer Report ***
	Number of Timer: 5
	  Timer[0] cuda Memory Alloc Pitch - Time: 92869(usec)
	  Timer[1] cuda Memory Copy 2D (host to dev) - Time: 4498(usec)
	  Timer[2] excute Multiply Matrix2D - Time: 28690(usec)
	  Timer[4] cuda Memory Copy 2D (dev to host) - Time: 3242(usec)
	  Timer[5] cuda Free - Time: 548(usec)
	*** End of report ***
	*** main Timer Report ***
	Number of Timer: 4
	  Timer[0] Excute CPP MatMul - Time: 43171057(usec)
	  Timer[1] Excute CPP MatMul Outer K - Time: 5597499(usec)
	  Timer[2] Excute CPP MatMul Outer Row - Time: 4909134(usec)
	  Timer[3] Excute CUDA MatMul - Time: 130457(usec)
	*** End of report ***
	dst_host=[502.43646 523.52637 502.42062 ... 507.51706 510.09119 503.78641
			505.91006 525.83258 499.76251 ... 509.14545 510.46594 502.85284
			501.39395 522.79553 499.03729 ... 508.65152 508.29178 499.34222
			........ ........ ........ ... ........ ........ ........
			500.82980 518.09973 494.40350 ... 499.89572 509.83496 500.52328
			507.49548 522.33942 502.25476 ... 505.95227 523.62408 499.39661
			489.12701 512.50214 493.45197 ... 502.39368 504.33810 489.14203]
	dst_cuda=[502.43643 523.52643 502.42062 ... 507.51706 510.09119 503.78635
			505.91006 525.83258 499.76251 ... 509.14545 510.46591 502.85284
			501.39395 522.79553 499.03729 ... 508.65152 508.29181 499.34222
			........ ........ ........ ... ........ ........ ........
			500.82980 518.09967 494.40350 ... 499.89572 509.83496 500.52328
			507.49548 522.33942 502.25473 ... 505.95224 523.62415 499.39664
			489.12698 512.50214 493.45197 ... 502.39368 504.33813 489.14206]
	Total Diff: 61.15140
	*/
}

void generalMatMulWithCuda(float* dst, const float alpha, const float beta, const float* srcA, const float* srcB, const float* srcC, const int widthA, const int heightA, const int widthB, const int heightB)
{
	float* dev_srcA;
	float* dev_srcB;
	float* dev_srcC;
	float* dev_dst;
	Timer t("cuda");
	const int numA = widthA * heightA;

	size_t hostPitchA = widthA * sizeof(float), hostPitchB = widthB * sizeof(float), hostPitchC = widthB * sizeof(float);
	size_t devPitchA, devPitchB, devPitchC, devPitchDst;

	t.start(0);
	cudaMallocPitch(&dev_srcA, &devPitchA, sizeof(float) * widthA, heightA);
	cudaMallocPitch(&dev_srcB, &devPitchB, sizeof(float) * widthB, heightB);
	cudaMallocPitch(&dev_srcC, &devPitchC, sizeof(float) * widthB, heightB);
	cudaMallocPitch(&dev_dst, &devPitchDst, sizeof(float) * widthB, heightA);
	t.end(0);
	t.setTimerContents(0, "cuda Memory Alloc Pitch");

	t.start(1);
	cudaMemcpy2D(dev_srcA, devPitchA, srcA, hostPitchA, widthA * sizeof(float), heightA, cudaMemcpyHostToDevice);
	cudaMemcpy2D(dev_srcB, devPitchB, srcB, hostPitchB, widthB * sizeof(float), heightB, cudaMemcpyHostToDevice);
	cudaMemcpy2D(dev_srcC, devPitchC, srcC, hostPitchC, widthB * sizeof(float), heightA, cudaMemcpyHostToDevice);
	t.end(1);
	t.setTimerContents(1, "cuda Memory Copy 2D (host to dev)");
	size_t elemA, elemB, elemDst;
	elemA = devPitchA / sizeof(float);
	elemB = devPitchB / sizeof(float);
	elemDst = devPitchDst / sizeof(float);

	t.start(2);
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(div_up(static_cast<unsigned int>(widthB), dimBlock.x), div_up(static_cast<unsigned int>(heightA), dimBlock.y), 1);
	generalMatMul << < dimGrid, dimBlock >> > (dev_dst, alpha, beta, dev_srcA, dev_srcB, dev_srcC, heightA, widthA, widthB, elemA, elemB);
	cudaDeviceSynchronize();
	t.end(2);
	t.setTimerContents(2, "excute GEMM");

	t.start(4);
	cudaMemcpy2D(dst, hostPitchB, dev_dst, devPitchDst, sizeof(float) * widthB, heightA, cudaMemcpyDeviceToHost);
	t.end(4);
	t.setTimerContents(4, "cuda Memory Copy 2D (dev to host)");

	t.start(5);
	cudaFree(dev_srcA);
	cudaFree(dev_srcB);
	cudaFree(dev_dst);
	t.end(5);
	t.setTimerContents(5, "cuda Free");

	t.printReport();
	t.release();

	/*
	GEMM Matrix2D - (Matrix Size:  2048 x  2048)
	*** cuda Timer Report ***
	Number of Timer: 5
	  Timer[0] cuda Memory Alloc Pitch - Time: 83157(usec)
	  Timer[1] cuda Memory Copy 2D (host to dev) - Time: 6027(usec)
	  Timer[2] excute GEMM - Time: 28279(usec)
	  Timer[4] cuda Memory Copy 2D (dev to host) - Time: 1512(usec)
	  Timer[5] cuda Free - Time: 436(usec)
	*** End of report ***
	*** main Timer Report ***
	Number of Timer: 3
	  Timer[0] Excute CPP GEMM - Time: 55177202(usec)
	  Timer[1] Excute CPP GEMM Outer Row - Time: 4547251(usec)
	  Timer[2] Excute CUDA GEMM - Time: 119728(usec)
	*** End of report ***
	dst_host=[614.49243 635.55292 628.41943 ... 635.57013 611.32422 615.94855
			623.93353 644.55316 648.95172 ... 645.46387 622.85583 632.66406
			622.20685 650.73767 634.99036 ... 657.78772 635.29724 632.61914
			........ ........ ........ ... ........ ........ ........
			627.28436 641.48840 650.02594 ... 651.02502 624.28186 637.82269
			633.50250 655.80920 645.91327 ... 655.32147 633.87885 643.00189
			639.15967 668.23584 669.51459 ... 657.33594 637.95947 640.58881]
	dst_cuda=[614.49243 635.55292 628.41943 ... 635.57007 611.32416 615.94855
			623.93347 644.55316 648.95172 ... 645.46387 622.85577 632.66394
			622.20685 650.73767 634.99036 ... 657.78772 635.29724 632.61914
			........ ........ ........ ... ........ ........ ........
			627.28436 641.48840 650.02594 ... 651.02502 624.28192 637.82269
			633.50244 655.80920 645.91327 ... 655.32147 633.87891 643.00189
			639.15961 668.23590 669.51447 ... 657.33594 637.95947 640.58893]
	Total Diff: 118.48645
	*/
}
