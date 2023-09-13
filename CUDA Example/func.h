#pragma once

void addWithCpp(float* dst, const float* srcA, const float* srcB, int num);
void axpyWithCpp(float* dst, const float a, const float* srcX, const float* srcY, int num);
void addMatWithCpp(float* dst, const float* srcA, const float* srcB, int nCol, int nRow);
void addMat3DWithCpp(float* dst, const float* srcA, const float* srcB, int nCh, int nCol, int nRow);
void adjDiffWithCpp(float* dst, const float* src, int num);
void transposeMatWithCpp(float* dst, const float* src, int nRow, int nCol);
void transposeMatWithCpp2(float* dst, const float* src, int nRow, int nCol);
void multiplyMatWithCpp(float* dst, const float* srcA, const float* srcB, int nColA, int nRowA, int nColB, int nRowB);
void multiplyMatWithCpp2(float* dst, const float* srcA, const float* srcB, int nColA, int nRowA, int nColB, int nRowB);
void multiplyMatWithCpp3(float* dst, const float* srcA, const float* srcB, int nColA, int nRowA, int nColB, int nRowB);
void gemMatWithCpp(float* dst, const float alpha, const float beta, const float* srcA, const float* srcB, const float* srcC, int nColA, int nRowA, int nColB);
void gemMatWithCpp2(float* dst, const float alpha, const float beta, const float* srcA, const float* srcB, const float* srcC, int nColA, int nRowA, int nColB);