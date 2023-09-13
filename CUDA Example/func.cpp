#include "func.h"
#include <algorithm>
#include <numeric>
#include <ppl.h>

void addWithCpp(float* dst, const float* srcA, const float* srcB, int num)
{
	/*for (int i = 0; i != num; ++i)
	{
		dst[i] = srcA[i] + srcB[i];
	}*/
	std::transform(srcA, srcA + num, srcB, dst, [](float a, float b) {return a + b; });
}

void axpyWithCpp(float* dst, const float a, const float* srcX, const float* srcY, int num)
{
	std::transform(srcX, srcX + num, srcY, dst, [a](float x, float y) {return a * x + y; });
}

void addMatWithCpp(float* dst, const float* srcA, const float* srcB, int nCol, int nRow)
{
	std::transform(srcA, srcA + (nCol * nRow), srcB, dst, [](float a, float b) {return a + b; });
}

void addMat3DWithCpp(float* dst, const float* srcA, const float* srcB, int nCh, int nCol, int nRow)
{
	std::transform(srcA, srcA + (nCh * nCol * nRow), srcB, dst, [](float a, float b) {return a + b; });
}

void adjDiffWithCpp(float* dst, const float* src, int num)
{
	std::adjacent_difference(src, src + num, dst);
}

void transposeMatWithCpp(float* dst, const float* src, int nRow, int nCol)
{
	for (int j = 0; j != nRow; ++j)
	{
		for (int i = 0; i != nCol; ++i)
		{
			dst[i * nCol + j] = src[j * nCol + i];
		}
	}
}

void transposeMatWithCpp2(float* dst, const float* src, int nRow, int nCol)
{
	Concurrency::parallel_for(0, nRow, [&](int i) {
		for (int j = 0; j != nCol; ++j)
		{
			dst[i * nCol + j] = src[j * nCol + i];
		}
	});

}

void multiplyMatWithCpp(float* dst, const float* srcA, const float* srcB, int nColA, int nRowA, int nColB, int nRowB)
{
	// (m * k) @ (k * n) => (m * n)
	// m = nRowA, k = nColA, nRowB, n = nColB

	for (int j = 0; j < nRowA; ++j)
	{
		for (int i = 0; i < nColB; ++i)
		{
			register float sum = 0;
			for (int k = 0; k < nColA; ++k)
			{
				sum += (srcA[j * nColA + k] * srcB[k * nColB + i]);
			}
			dst[j * nColB + i] = sum;
		}
	}
}

void multiplyMatWithCpp2(float* dst, const float* srcA, const float* srcB, int nColA, int nRowA, int nColB, int nRowB)
{
	memset(dst, 0, nRowA * nColB * sizeof(float));
	for (int k = 0; k < nColA; ++k)
	{
		for (int j = 0; j < nRowA; ++j)
		{
			for (int i = 0; i < nColB; ++i)
			{
				dst[j * nColB + i] += (srcA[j * nColA + k] * srcB[k * nColB + i]);
			}
		}
	}
}

void multiplyMatWithCpp3(float* dst, const float* srcA, const float* srcB, int nColA, int nRowA, int nColB, int nRowB)
{
	memset(dst, 0, nRowA * nColB * sizeof(float));
	for (int j = 0; j < nRowA; ++j)
	{
		for (int k = 0; k < nColA; ++k)
		{
			for (int i = 0; i < nColB; ++i)
			{
				dst[j * nColB + i] += (srcA[j * nColA + k] * srcB[k * nColB + i]);
			}
		}
	}
}

void gemMatWithCpp(float* dst, const float alpha, const float beta,const float* srcA, const float* srcB, const float* srcC, int nColA, int nRowA, int nColB)
{

	for (int j = 0; j < nRowA; ++j)
	{
		for (int i = 0; i < nColB; ++i)
		{
			float sum = 0.f;
			for (int k = 0; k < nColA; ++k)
			{
				sum += (srcA[j * nColA + k] * srcB[k * nColB + i]);
			}
			dst[j * nColB + i] = alpha * sum + beta * srcC[j * nColB + i];
		}
	}
}

void gemMatWithCpp2(float* dst, const float alpha, const float beta, const float* srcA, const float* srcB, const float* srcC, int nColA, int nRowA, int nColB)
{
	memset(dst, 0, nRowA * nColB * sizeof(float));
	for (int j = 0; j < nRowA; ++j)
	{
		for (int k = 0; k < nColA; ++k)
		{
			for (int i = 0; i < nColB; ++i)
			{
				dst[j * nColB + i] += (srcA[j * nColA + k] * srcB[k * nColB + i]);
			}
		}
	}

	for (int j = 0; j != nRowA; ++j)
	{
		for (int i = 0; i != nColB; ++i)
		{
			dst[j * nColB + i] = alpha * dst[j * nColB + i] + beta * srcC[j * nColB + i];
		}
	}
}
