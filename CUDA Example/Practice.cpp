#include "Practice.h"

// Comparison of Vector Plus Using CPP and Cuda
// In this problem, C++ shows better speed
void practice1()
{
	const int SIZE = 1024 * 1024 * 128;
	std::cout << std::format("Vector Add (A + B) - (vector length: {:10})\n", SIZE);
	float* srcA = new float[SIZE];
	float* srcB = new float[SIZE];
	setNormalizeRandomData(srcA, SIZE);
	setNormalizeRandomData(srcB, SIZE);
	Timer t("main");

	float* dst_cuda = new float[SIZE];
	float* dst_host = new float[SIZE];

	t.start(0);
	addWithCpp(dst_host, srcA, srcB, SIZE);
	t.end(0);
	t.setTimerContents(0, "Excute CPP addVector");

	t.start(1);
	addWithCuda(dst_cuda, srcA, srcB, SIZE);
	t.end(1);
	t.setTimerContents(1, "Excute CUDA addVector");

	float diff = getToalDiff(dst_host, dst_cuda, SIZE);

	t.printReport();
	t.release();
	printVec("dst_host", dst_host, SIZE);
	printVec("dst_cuda", dst_cuda, SIZE);
	std::cout << std::format("Total Diff: {:8.5f}\n", diff);
	delete[] srcA;
	delete[] srcB;

	/*
	Vector Add(A + B) - (vector length : 134217728)
	* **cuda Timer Report * **
	Number of Timer : 5
	Timer[0] cuda Memory Alloc - Time : 112253(usec)
	Timer[1] cuda Memory Copy(host to dev) - Time : 130096(usec)
	Timer[2] excute addVector - Time : 8077(usec)
	Timer[3] cuda Memory Copy(dev to host) - Time : 110598(usec)
	Timer[4] cuda Free - Time : 8451(usec)
	* **End of report * **
	***main Timer Report * **
	Number of Timer : 2
	Timer[0] Excute CPP addVector - Time : 101031(usec)
	Timer[1] Excute CUDA addVector - Time : 369745(usec)
	* **End of report * **
	dst_host = [1.06900  1.43609  1.19731  1.37215 ...  1.41547  1.39442  0.88352  0.70877]
	dst_cuda = [1.06900  1.43609  1.19731  1.37215 ...  1.41547  1.39442  0.88352  0.70877]
	Total Diff : 0.00000
	*/
}

void practice2()
{
	/* AXPY Problem */
	// FMA applications - Dot product, Linear interpolation

	const float a = 1.234f;
	const int SIZE = 256 * 1024 * 1024;

	std::cout << std::format("Vector SAXPY (AX + B) - (vector length: {:10})\n", SIZE);
	float* srcX = new float[SIZE];
	float* srcY = new float[SIZE];
	setNormalizeRandomData(srcX, SIZE);
	setNormalizeRandomData(srcY, SIZE);
	Timer t("main");
	float* dst_cuda = new float[SIZE];
	float* dst_host = new float[SIZE];

	t.start(0);
	axpyWithCpp(dst_host, a, srcX, srcY, SIZE);
	t.end(0);
	t.setTimerContents(0, "Excute CPP axpyVector");

	t.start(1);
	axpyWithCuda(dst_cuda, a, srcX, srcY, SIZE);
	t.end(1);
	t.setTimerContents(1, "Excute CUDA axpyVector");

	float diff = getToalDiff(dst_host, dst_cuda, SIZE);

	t.printReport();
	t.release();

	printVec("dst_host", dst_host, SIZE);
	printVec("dst_cuda", dst_cuda, SIZE);
	std::cout << std::format("Total Diff: {:8.5f}\n", diff);
	delete[] srcX;
	delete[] srcY;

	/*
	Vector SAXPY (AX + B) - (vector length:  268435456)
	*** cuda Timer Report ***
	Number of Timer: 5
	  Timer[0] cuda Memory Alloc - Time: 122421(usec)
	  Timer[1] cuda Memory Copy (host to dev) - Time: 262862(usec)
	  Timer[2] excute axpyVector - Time: 16169(usec)
	  Timer[3] cuda Memory Copy (dev to host) - Time: 192590(usec)
	  Timer[4] cuda Free - Time: 22054(usec)
	*** End of report ***
	*** main Timer Report ***
	Number of Timer: 2
	  Timer[0] Excute CPP axpyVector - Time: 213901(usec)
	  Timer[1] Excute CUDA axpyVector - Time: 618667(usec)
	*** End of report ***
	dst_host=[ 0.58951  1.83589  1.67597  1.74126 ...  1.98980  1.20209  1.90191  1.49549]
	dst_cuda=[ 0.58951  1.83589  1.67597  1.74126 ...  1.98980  1.20209  1.90191  1.49549]
	Total Diff:  4.91284
	*/
}

void practice3()
{
	/* Mat Add Problem */

	const int width = 4096;
	const int height = 4096;
	const int SIZE = width * height;

	std::cout << std::format("Add Matrix - (Matrix Size: {:5} x {:5})\n", width, height);
	float* srcA = new float[SIZE];
	float* srcB = new float[SIZE];
	setNormalizeRandomData(srcA, SIZE);
	setNormalizeRandomData(srcB, SIZE);
	Timer t("main");
	float* dst_cuda = new float[SIZE];
	float* dst_host = new float[SIZE];

	t.start(0);
	addMatWithCpp(dst_host, srcA, srcB, width, height);
	t.end(0);
	t.setTimerContents(0, "Excute CPP addMat");

	t.start(1);
	addMatWithCuda(dst_cuda, srcA, srcB, width, height);
	t.end(1);
	t.setTimerContents(1, "Excute CUDA addMat");

	float diff = getToalDiff(dst_host, dst_cuda, SIZE);

	t.printReport();
	t.release();

	printMat("dst_host", dst_host, height, width);
	printMat("dst_cuda", dst_cuda, height, width);
	std::cout << std::format("Total Diff: {:8.5f}\n", diff);
	delete[] srcA;
	delete[] srcB;

	/*
	Add Matrix2D - (Matrix Size:  4096 x  4096)
	*** cuda Timer Report ***
	Number of Timer: 5
	  Timer[0] cuda Memory Alloc Pitch - Time: 112261(usec)
	  Timer[1] cuda Memory Copy 2D (host to dev) - Time: 18305(usec)
	  Timer[2] excute addMatrix2D - Time: 1105(usec)
	  Timer[4] cuda Memory Copy 2D (dev to host) - Time: 14133(usec)
	  Timer[5] cuda Free - Time: 483(usec)
	*** End of report ***
	*** main Timer Report ***
	Number of Timer: 2
	  Timer[0] Excute CPP addMat - Time: 14370(usec)
	  Timer[1] Excute CUDA addMat - Time: 146581(usec)
	*** End of report ***
	dst_host=[1.40362  0.78794  1.51083 ...  1.02902  1.41752  1.33741
			 1.57639  1.07672  0.37824 ...  1.06753  1.09097  0.04540
			 0.89169  1.09276  1.72118 ...  1.05971  0.89021  1.09594
			........ ........ ........ ... ........ ........ ........
			 1.15290  1.31583  0.68448 ...  1.33799  1.27038  1.02202
			 1.47485  0.74706  0.50712 ...  1.16629  1.25789  0.52641
			 0.46893  1.81596  1.26689 ...  0.89997  0.59851  0.55032]
	dst_cuda=[1.40362  0.78794  1.51083 ...  1.02902  1.41752  1.33741
			 1.57639  1.07672  0.37824 ...  1.06753  1.09097  0.04540
			 0.89169  1.09276  1.72118 ...  1.05971  0.89021  1.09594
			........ ........ ........ ... ........ ........ ........
			 1.15290  1.31583  0.68448 ...  1.33799  1.27038  1.02202
			 1.47485  0.74706  0.50712 ...  1.16629  1.25789  0.52641
			 0.46893  1.81596  1.26689 ...  0.89997  0.59851  0.55032]
	Total Diff:  0.00000
	*/
}

void practice4()
{
	/* Matrix 2D Add Problem */

	const int width = 4096;
	const int height = 4096;
	const int SIZE = width * height;

	std::cout << std::format("Add Matrix2D - (Matrix Size: {:5} x {:5})\n", width, height);
	float* srcA = new float[SIZE];
	float* srcB = new float[SIZE];
	setNormalizeRandomData(srcA, SIZE);
	setNormalizeRandomData(srcB, SIZE);
	Timer t("main");
	float* dst_cuda = new float[SIZE];
	float* dst_host = new float[SIZE];

	t.start(0);
	addMatWithCpp(dst_host, srcA, srcB, width, height);
	t.end(0);
	t.setTimerContents(0, "Excute CPP addMat");

	t.start(1);
	addMat2DWithCuda(dst_cuda, srcA, srcB, width, height);
	t.end(1);
	t.setTimerContents(1, "Excute CUDA addMat");

	float diff = getToalDiff(dst_host, dst_cuda, SIZE);

	t.printReport();
	t.release();

	printMat("dst_host", dst_host, height, width);
	printMat("dst_cuda", dst_cuda, height, width);
	std::cout << std::format("Total Diff: {:8.5f}\n", diff);
	delete[] srcA;
	delete[] srcB;
	delete[] dst_cuda;
	delete[] dst_host;

	/*
	Add Matrix2D - (Matrix Size:  4096 x  4096)
	*** cuda Timer Report ***
	Number of Timer: 5
	  Timer[0] cuda Memory Alloc Pitch - Time: 105520(usec)
	  Timer[1] cuda Memory Copy 2D (host to dev) - Time: 16029(usec)
	  Timer[2] excute addMatrix2D - Time: 2073(usec)
	  Timer[4] cuda Memory Copy 2D (dev to host) - Time: 5541(usec)
	  Timer[5] cuda Free - Time: 812(usec)
	*** End of report ***
	*** main Timer Report ***
	Number of Timer: 2
	  Timer[0] Excute CPP addMat - Time: 53747(usec)
	  Timer[1] Excute CUDA addMat - Time: 130710(usec)
	*** End of report ***
	dst_host=[0.42492  0.75060  0.67953 ...  0.72424  0.70598  0.90987
			 1.17958  0.75346  0.59147 ...  1.19028  1.06665  0.94460
			 0.70630  1.05359  0.54010 ...  0.58850  1.45044  0.61736
			........ ........ ........ ... ........ ........ ........
			 0.87320  0.67056  0.63167 ...  1.28350  0.84685  1.08056
			 0.99875  1.08004  1.05435 ...  0.79626  0.89331  0.45439
			 0.72189  0.97689  0.74081 ...  0.69156  1.03095  1.11250]
	dst_cuda=[0.42492  0.75060  0.67953 ...  0.72424  0.70598  0.90987
			 1.17958  0.75346  0.59147 ...  1.19028  1.06665  0.94460
			 0.70630  1.05359  0.54010 ...  0.58850  1.45044  0.61736
			........ ........ ........ ... ........ ........ ........
			 0.87320  0.67056  0.63167 ...  1.28350  0.84685  1.08056
			 0.99875  1.08004  1.05435 ...  0.79626  0.89331  0.45439
			 0.72189  0.97689  0.74081 ...  0.69156  1.03095  1.11250]
	Total Diff:  0.00000
	*/
}

void practice5()
{
	/* Matrix 3D Add Problem */
	const int width = 300;
	const int height = 300;
	const int depth = 256;
	const int SIZE = width * height * depth;

	std::cout << std::format("Add Matrix3D - (Matrix Size: {:5} x {:5} x {:5})\n", depth, width, height);
	float* srcA = new float[SIZE];
	float* srcB = new float[SIZE];
	setNormalizeRandomData(srcA, SIZE);
	setNormalizeRandomData(srcB, SIZE);
	Timer t("main");
	float* dst_cuda = new float[SIZE];
	float* dst_host = new float[SIZE];

	t.start(0);
	addMat3DWithCpp(dst_host, srcA, srcB, depth, width, height);
	t.end(0);
	t.setTimerContents(0, "Excute CPP addMat");

	t.start(1);
	addMat3DWithCuda(dst_cuda, srcA, srcB, depth, width, height);
	t.end(1);
	t.setTimerContents(1, "Excute CUDA addMat");

	float diff = getToalDiff(dst_host, dst_cuda, SIZE);

	t.printReport();
	t.release();

	printMat("dst_host", dst_host, height, width);
	printMat("dst_cuda", dst_cuda, height, width);
	std::cout << std::format("Total Diff: {:8.5f}\n", diff);
	delete[] srcA;
	delete[] srcB;
	delete[] dst_host;
	delete[] dst_cuda;

	/*
	Add Matrix3D - (Matrix Size:   256 x   300 x   300)
	*** cuda Timer Report ***
	Number of Timer: 5
	  Timer[0] cuda Memory Alloc 3D - Time: 83713(usec)
	  Timer[1] cuda Memory Copy 3D (host to dev) - Time: 24511(usec)
	  Timer[2] excute addMatrix3D - Time: 1516(usec)
	  Timer[4] cuda Memory Copy 3D (dev to host) - Time: 18453(usec)
	  Timer[5] cuda Free - Time: 1743(usec)
	*** End of report ***
	*** main Timer Report ***
	Number of Timer: 2
	  Timer[0] Excute CPP addMat - Time: 18444(usec)
	  Timer[1] Excute CUDA addMat - Time: 130216(usec)
	*** End of report ***
	dst_host=[0.64722  1.29485  0.42658 ...  1.21286  1.93072  0.59664
			 1.81024  0.66941  1.65920 ...  1.19637  0.33931  0.95443
			 1.24471  1.62201  1.69836 ...  0.95723  1.22692  0.10876
			........ ........ ........ ... ........ ........ ........
			 0.69075  0.88873  0.56800 ...  0.69027  0.95832  0.67299
			 1.27810  1.12165  0.33043 ...  0.60353  0.58152  0.91419
			 1.87638  0.40079  1.16243 ...  0.58452  0.98832  0.74058]
	dst_cuda=[0.64722  1.29485  0.42658 ...  1.21286  1.93072  0.59664
			 1.81024  0.66941  1.65920 ...  1.19637  0.33931  0.95443
			 1.24471  1.62201  1.69836 ...  0.95723  1.22692  0.10876
			........ ........ ........ ... ........ ........ ........
			 0.69075  0.88873  0.56800 ...  0.69027  0.95832  0.67299
			 1.27810  1.12165  0.33043 ...  0.60353  0.58152  0.91419
			 1.87638  0.40079  1.16243 ...  0.58452  0.98832  0.74058]
	Total Diff:  0.00000
	*/
}

void practice6()
{
	/* adjacent difference Problem */
	
	const int SIZE = 16 * 1024 * 1024;

	std::cout << std::format("adjacent difference vector - (Matrix Size: {:9})\n", SIZE);
	float* src = new float[SIZE];
	setArangeData(src, SIZE);
	
	Timer t("main");
	float* dst_cuda = new float[SIZE];
	float* dst_host = new float[SIZE];

	t.start(0);
	adjDiffWithCpp(dst_host, src, SIZE);
	t.end(0);
	t.setTimerContents(0, "Excute CPP adjDiff");

	t.start(1);
	adjDiffWithCuda(dst_cuda, src, SIZE);
	t.end(1);
	t.setTimerContents(1, "Excute CUDA adjDiff");

	float diff = getToalDiff(dst_host, dst_cuda, SIZE);

	t.printReport();
	t.release();

	printVec("dst_host", dst_host, SIZE);
	printVec("dst_cuda", dst_cuda, SIZE);
	std::cout << std::format("Total Diff: {:8.5f}\n", diff);
	delete[] src;
	delete[] dst_host;
	delete[] dst_cuda;
	/*
	adjacent difference vector - (Matrix Size:  16777216)
	*** cuda Timer Report ***
	Number of Timer: 6
	  Timer[0] cuda Memory Alloc - Time: 89281(usec)
	  Timer[1] cuda Memory Copy (host to dev) - Time: 9475(usec)
	  Timer[2] excute adjDiffVector - Time: 818(usec)
	  Timer[3] excute adjDiffVectorWithShared - Time: 1248(usec)
	  Timer[4] cuda Memory Copy (dev to host) - Time: 10546(usec)
	  Timer[5] cuda Free - Time: 353(usec)
	*** End of report ***
	*** main Timer Report ***
	Number of Timer: 2
	  Timer[0] Excute CPP adjDiff - Time: 27611(usec)
	  Timer[1] Excute CUDA adjDiff - Time: 112003(usec)
	*** End of report ***
	dst_host=[ 0.00000  1.00000  1.00000  1.00000 ...  1.00000  1.00000  1.00000  1.00000]
	dst_cuda=[ 0.00000  1.00000  1.00000  1.00000 ...  1.00000  1.00000  1.00000  1.00000]
	Total Diff:  0.00000
	*/
}

void practice7()
{
	/* Matrix 2D Transpose Problem */

	const int width = 16384;
	const int height = 16384;
	const int SIZE = width * height;

	std::cout << std::format("Add Matrix2D - (Matrix Size: {:5} x {:5})\n", width, height);
	float* srcA = new float[SIZE];
	setNormalizeRandomData(srcA, SIZE);

	Timer t("main");
	float* dst_cuda = new float[SIZE];
	float* dst_host = new float[SIZE];

	t.start(0);
	transposeMatWithCpp(dst_host, srcA, height, width);
	t.end(0);
	t.setTimerContents(0, "Excute CPP transposeMatrix");

	t.start(1);
	transposeMatWithCpp2(dst_host, srcA, height, width);
	t.end(1);
	t.setTimerContents(1, "Excute CPP transposeMatrix(Parallel)");

	t.start(2);
	transposeMatrixWithCuda(dst_cuda, srcA, width, height);
	t.end(2);
	t.setTimerContents(2, "Excute CUDA transposeMatrix");

	float diff = getToalDiff(dst_host, dst_cuda, SIZE);

	t.printReport();
	t.release();

	printMat("dst_host", dst_host, height, width);
	printMat("dst_cuda", dst_cuda, height, width);
	std::cout << std::format("Total Diff: {:8.5f}\n", diff);
	delete[] srcA;
	delete[] dst_cuda;
	delete[] dst_host;
	/*
	Add Matrix2D - (Matrix Size: 16384 x 16384)
	*** cuda Timer Report ***
	Number of Timer: 7
	  Timer[0] cuda Memory Alloc Pitch - Time: 91266(usec)
	  Timer[1] cuda Memory Copy 2D (host to dev) - Time: 143971(usec)
	  Timer[2] excute transposeMatrix - Time: 34264(usec)
	  Timer[3] excute transposeMatrixOptim - Time: 16361(usec)
	  Timer[4] excute transposeMatrixOptim No Bank Conflict - Time: 10463(usec)
	  Timer[5] cuda Memory Copy 2D (dev to host) - Time: 193621(usec)
	  Timer[6] cuda Free - Time: 14207(usec)
	*** End of report ***
	*** main Timer Report ***
	Number of Timer: 3
	  Timer[0] Excute CPP transposeMatrix - Time: 3500822(usec)
	  Timer[1] Excute CPP transposeMatrix(Parallel) - Time: 909270(usec)
	  Timer[2] Excute CUDA transposeMatrix - Time: 504573(usec)
	*** End of report ***
	dst_host=[0.90733  0.40499  0.69589 ...  0.61737  0.32287  0.12873
			 0.17185  0.96325  0.05849 ...  0.83882  0.51669  0.33833
			 0.90967  0.74670  0.71186 ...  0.94735  0.40123  0.33894
			........ ........ ........ ... ........ ........ ........
			 0.65913  0.42273  0.11220 ...  0.52982  0.44781  0.95180
			 0.79725  0.73541  0.72180 ...  0.85930  0.08519  0.18762
			 0.13578  0.38893  0.50833 ...  0.88814  0.52640  0.51606]
	dst_cuda=[0.90733  0.40499  0.69589 ...  0.61737  0.32287  0.12873
			 0.17185  0.96325  0.05849 ...  0.83882  0.51669  0.33833
			 0.90967  0.74670  0.71186 ...  0.94735  0.40123  0.33894
			........ ........ ........ ... ........ ........ ........
			 0.65913  0.42273  0.11220 ...  0.52982  0.44781  0.95180
			 0.79725  0.73541  0.72180 ...  0.85930  0.08519  0.18762
			 0.13578  0.38893  0.50833 ...  0.88814  0.52640  0.51606]
	Total Diff: 98245.85938
	*/
}

void practice8()
{
	const int width = 2048;
	const int height = 2048;
	const int SIZE = width * height;

	std::cout << std::format("Mutiply Matrix2D - (Matrix Size: {:5} x {:5})\n", width, height);
	float* srcA = new float[SIZE];
	float* srcB = new float[SIZE];
	
	setNormalizeRandomData(srcA, SIZE);
	setNormalizeRandomData(srcB, SIZE);
	//std::fill(srcA, srcA + SIZE, 1.0f);
	//std::fill(srcB, srcB + SIZE, 1.0f);

	Timer t("main");
	float* dst_cuda = new float[SIZE];
	float* dst_host = new float[SIZE];

	t.start(0);
	multiplyMatWithCpp(dst_host, srcA, srcB, width, height, width, height);
	t.end(0);
	t.setTimerContents(0, "Excute CPP MatMul");

	t.start(1);
	multiplyMatWithCpp2(dst_host, srcA, srcB, width, height, width, height);
	t.end(1);
	t.setTimerContents(1, "Excute CPP MatMul Outer K");

	t.start(2);
	multiplyMatWithCpp3(dst_host, srcA, srcB, width, height, width, height);
	t.end(2);
	t.setTimerContents(2, "Excute CPP MatMul Outer Row");

	t.start(3);
	multiplyMatWithCuda(dst_cuda, srcA, srcB, width, height, width, height);
	t.end(3);
	t.setTimerContents(3, "Excute CUDA MatMul");
	float diff = getToalDiff(dst_host, dst_cuda, SIZE);

	t.printReport();
	t.release();
	//printMat("srcA", srcA, height, width);
	//printMat("srcB", srcB, height, width);
	printMat("dst_host", dst_host, height, width);
	printMat("dst_cuda", dst_cuda, height, width);
	std::cout << std::format("Total Diff: {:8.5f}\n", diff);
	delete[] srcA;
	delete[] srcB;
	delete[] dst_cuda;
	delete[] dst_host;

}

void practice9()
{
	const int width = 2048;
	const int height = 2048;
	const int SIZE = width * height;

	std::cout << std::format("GEMM Matrix2D - (Matrix Size: {:5} x {:5})\n", width, height);
	const float a = 1.234f;
	const float b = 4.321f;
	float* srcA = new float[SIZE];
	float* srcB = new float[SIZE];
	float* srcC = new float[SIZE];

	setNormalizeRandomData(srcA, SIZE);
	setNormalizeRandomData(srcB, SIZE);
	setNormalizeRandomData(srcC, SIZE);

	Timer t("main");
	float* dst_cuda = new float[SIZE];
	float* dst_host = new float[SIZE];

	t.start(0);
	gemMatWithCpp(dst_host, a, b,srcA, srcB, srcC, width, height, width);
	t.end(0);
	t.setTimerContents(0, "Excute CPP GEMM");

	t.start(1);
	gemMatWithCpp2(dst_cuda, a, b, srcA, srcB, srcC, width, height, width);
	t.end(1);
	t.setTimerContents(1, "Excute CPP GEMM Outer Row");

	t.start(2);
	generalMatMulWithCuda(dst_cuda, a, b,srcA, srcB, srcC, width, height, width, height);
	t.end(2);
	t.setTimerContents(2, "Excute CUDA GEMM");
	float diff = getToalDiff(dst_host, dst_cuda, SIZE);

	t.printReport();
	t.release();
	printMat("dst_host", dst_host, height, width);
	printMat("dst_cuda", dst_cuda, height, width);
	std::cout << std::format("Total Diff: {:8.5f}\n", diff);
	delete[] srcA;
	delete[] srcB;
	delete[] dst_cuda;
	delete[] dst_host;
}
