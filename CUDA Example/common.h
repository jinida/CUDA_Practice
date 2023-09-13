#pragma once

#include <assert.h>
#include <string.h>
#define _USE_MATH_DEFINES // to use M_PI
#include <math.h>

#include <typeinfo>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <execution>
#include <random>
#include <format>

template<typename T>
void setArangeData(T* dst, int num)
{
	std::iota(dst, dst + num, 0);
}

template <typename T>
void setRandomData(T* pDst, int num, T bound = static_cast<T>(1000))
{
	int32_t bnd = static_cast<int32_t>(bound);

	std::random_device rd;
	std::mt19937 generator(rd());

	std::uniform_real_distribution<T> uniformDist(0.0, bound);

	while (num--)
	{
		*pDst++ = uniformDist(generator);
	}
}

template <typename T>
void setNormalizeRandomData(T* pDst, int num)
{
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<T> uniformDist(0.0, 1.0);
	while (num--)
	{
		*pDst++ = uniformDist(generator);
	}
}

template<typename T>
T getNaiveSum(const T* pSrc, int num)
{
	register T sum = static_cast<T>(0);
	const int chunk = 128 * 1024;

	while (num > chunk)
	{
		register T partialSum = static_cast<T>(0);
		register int n = chunk;
		while (n--)
		{
			partialSum += *pSrc++;
		}
		sum += partialSum;
		num -= chunk;
	}

	register T partialSum = static_cast<T>(0);
	while (num > 0)
	{
		partialSum += *pSrc++;
		--num;
	}
	sum += partialSum;

	return sum;
}

template<typename T>
T getSum(const T* pSrc, int num)
{
	return std::accumulate(pSrc, pSrc + num, 0.0);
}

template <typename T>
T parallelSum(const T* pSrc, int num)
{
	return std::reduce(std::execution::par, pSrc, pSrc + num);
}

template <typename T>
T getToalDiff(const T* lhs, const T* rhs, int num)
{
	register T sum = static_cast<T>(0);

	const int chunk = 128 * 1024;
	while (num > chunk) {
		register T partial = static_cast<T>(0);
		register int n = chunk;
		while (n--) {
			if (typeid(T) == typeid(unsigned int)) {
				partial += std::abs(static_cast<int>(*lhs++) - static_cast<int>(*rhs++));
			}
			else {
				partial += std::abs((*lhs++) - (*rhs++));
			}
		}
		sum += partial;
		num -= chunk;
	}

	register T partial = static_cast<T>(0);
	while (num--) {
		if (typeid(T) == typeid(unsigned int)) {
			partial += std::abs(static_cast<int>(*lhs++) - static_cast<int>(*rhs++));
		}
		else {
			partial += std::abs((*lhs++) - (*rhs++));
		}
	}
	sum += partial;
	return sum;
}


template <typename T>
void printVec(const char* name, const T* vec, int num)
{
	std::streamsize ss = std::cout.precision();
	std::cout.precision(5);
	std::cout << std::setw(8) << name << "=[";
	std::cout << std::fixed << std::showpoint << std::setw(8) << vec[0] << " ";
	std::cout << std::fixed << std::showpoint << std::setw(8) << vec[1] << " ";
	std::cout << std::fixed << std::showpoint << std::setw(8) << vec[2] << " ";
	std::cout << std::fixed << std::showpoint << std::setw(8) << vec[3] << " ... " ;
	std::cout << std::fixed << std::showpoint << std::setw(8) << vec[num - 4] << " ";
	std::cout << std::fixed << std::showpoint << std::setw(8) << vec[num - 3] << " ";
	std::cout << std::fixed << std::showpoint << std::setw(8) << vec[num - 2] << " ";
	std::cout << std::fixed << std::showpoint << std::setw(8) << vec[num - 1] << "]";
	std::cout << std::endl;
	std::cout.precision(ss);
}

//void printMat(const char* name, const int* mat, int nRow, int nCol)
//{
//#define M(row, col) mat[(row) * nCol + (col)]
//	std::cout << std::format("{}=", name);
//	std::cout << std::format("[{:7} {:8} {:8} ... {:8} {:8} {:8}\n"
//		, M(0, 0), M(0, 1), M(0, 2), M(0, nCol - 3), M(0, nCol - 2), M(0, nCol - 1));
//	std::cout << std::format("\t{:8} {:8} {:8} ... {:8} {:8} {:8}\n"
//		, M(1, 0), M(1, 1), M(1, 2), M(1, nCol - 3), M(1, nCol - 2), M(1, nCol - 1));
//	std::cout << std::format("\t{:8} {:8} {:8} ... {:8} {:8} {:8}\n"
//		, M(2, 0), M(2, 1), M(2, 2), M(2, nCol - 3), M(2, nCol - 2), M(2, nCol - 1));
//	std::cout << std::format("\t{:.<8} {:.<8} {:.<8} ... {:.<8} {:.<8} {:.<8}\n", "", "", "", "", "", "");
//	std::cout << std::format("\t{:8} {:8} {:8} ... {:8} {:8} {:8}\n"
//		, M(nRow - 3, 0), M(nRow - 3, 1), M(nRow - 3, 2), M(nRow - 3, nCol - 3), M(nRow - 3, nCol - 2), M(nRow - 3, nCol - 1));
//	std::cout << std::format("\t{:8} {:8} {:8} ... {:8} {:8} {:8}\n"
//		, M(nRow - 2, 0), M(nRow - 2, 1), M(nRow - 2, 2), M(nRow - 2, nCol - 3), M(nRow - 2, nCol - 2), M(nRow - 2, nCol - 1));
//	std::cout << std::format("\t{:8} {:8} {:8} ... {:8} {:8} {:8}]\n"
//		, M(nRow - 1, 0), M(nRow - 1, 1), M(nRow - 1, 2), M(nRow - 1, nCol - 3), M(nRow - 1, nCol - 2), M(nRow - 1, nCol - 1));
//#undef M
//}

template <typename T>
void printMat(const char* name, const T* mat, int nRow, int nCol)
{
#define M(row, col) mat[(row) * nCol + (col)]
	std::cout << std::format("{}=", name);
	std::cout << std::format("[{:7.5f} {:8.5f} {:8.5f} ... {:8.5f} {:8.5f} {:8.5f}\n"
		, M(0, 0), M(0, 1), M(0, 2), M(0, nCol - 3), M(0, nCol - 2), M(0, nCol - 1));
	std::cout << std::format("\t{:8.5f} {:8.5f} {:8.5f} ... {:8.5f} {:8.5f} {:8.5f}\n"
		, M(1, 0), M(1, 1), M(1, 2), M(1, nCol - 3), M(1, nCol - 2), M(1, nCol - 1));
	std::cout << std::format("\t{:8.5f} {:8.5f} {:8.5f} ... {:8.5f} {:8.5f} {:8.5f}\n"
		, M(2, 0), M(2, 1), M(2, 2), M(2, nCol - 3), M(2, nCol - 2), M(2, nCol - 1));
	std::cout << std::format("\t{:.<8} {:.<8} {:.<8} ... {:.<8} {:.<8} {:.<8}\n", "", "", "", "", "", "");
	std::cout << std::format("\t{:8.5f} {:8.5f} {:8.5f} ... {:8.5f} {:8.5f} {:8.5f}\n"
		, M(nRow - 3, 0), M(nRow - 3, 1), M(nRow - 3, 2), M(nRow - 3, nCol - 3), M(nRow - 3, nCol - 2), M(nRow - 3, nCol - 1));
	std::cout << std::format("\t{:8.5f} {:8.5f} {:8.5f} ... {:8.5f} {:8.5f} {:8.5f}\n"
		, M(nRow - 2, 0), M(nRow - 2, 1), M(nRow - 2, 2), M(nRow - 2, nCol - 3), M(nRow - 2, nCol - 2), M(nRow - 2, nCol - 1));
	std::cout << std::format("\t{:8.5f} {:8.5f} {:8.5f} ... {:8.5f} {:8.5f} {:8.5f}]\n"
		, M(nRow - 1, 0), M(nRow - 1, 1), M(nRow - 1, 2), M(nRow - 1, nCol - 3), M(nRow - 1, nCol - 2), M(nRow - 1, nCol - 1));
#undef M
}

template <typename T>
void printMat3D(const char* name, const T* mat, int dimX, int dimY, int dimZ)
{
	printMat(std::format("{}[0]", name).c_str(), mat, dimY, dimX);
	printMat(std::format("{}[{}]", name, dimZ - 1).c_str(), mat + (dimZ - 1) * (dimY * dimX), dimY, dimX);
}
