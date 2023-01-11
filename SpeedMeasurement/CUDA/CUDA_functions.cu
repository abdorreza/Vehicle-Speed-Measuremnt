#include <iostream>
#include <time.h>
#include <ratio>
#include <chrono>
#include "opencv2/imgproc.hpp"
#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include "..\Motion_Detection\motion_detection.h"



using namespace cv;
using namespace std;
using namespace std::chrono;


#define BX 30
#define BY 30
#define DX 1920
#define DY 1080

__global__ void _SegmentationBy_GPU(uchar* mt, uchar* motion, size_t step, int h, int w)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	//int index = ((w*row) + col);
	int index = col + row*(step / sizeof(uchar));
	if (index >= (h*w))
		return;
	if (motion[index] > 0)
		mt[index] = 1;
	else
		mt[index] = 0;
}


void SegmentationGPU(const Mat& motion, Mat& dst)
{
	dim3 block(BX, BY);
	//dim3 grid((DX + block.x - 1) / block.x, (DY + block.y - 1) / block.y);
	dim3 grid(DX / block.x, DY / block.y);
	cuda::GpuMat GMat;
	cuda::GpuMat GMotionMat;
	dst = Mat::zeros(motion.size(), CV_8U);
	GMat.upload(dst);
	GMotionMat.upload(motion);
	//_SegmentationBy_GPU << <numBlocks, threadsPerBlock >> > ((uchar *)GMat.data, (uchar *)GMotionMat.data, dst.rows, dst.cols);
	//high_resolution_clock::time_point t1 = high_resolution_clock::now();
	//_SegmentationBy_GPU << <grid, block >> > ((uchar *)GMat.data, (uchar *)GMotionMat.data, dst.rows, dst.cols);
	_SegmentationBy_GPU << <grid, block >> > ((uchar *)GMat.data, (uchar *)GMotionMat.data, GMat.step, dst.rows, dst.cols);
	//high_resolution_clock::time_point t2 = high_resolution_clock::now();
	//duration<double, std::milli> time_span = t2 - t1;
	//cout << "*** GPU Segmentation time for M [" << time_span.count() << "] milliseconds.\n";
	GMat.download(dst);
}

__global__ void _Update_MHI_GPU(uchar*_GMHI, uchar* GMHI, uchar* GSILH, size_t step, int h, int w, int MHI_DURATION)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	//int index = ((w*row) + col);
	int index = col + row*(step / sizeof(uchar));
	if (index >= (h*w))
		return;
	if ( GSILH[index] == 1 )
		_GMHI[index] = MHI_DURATION;
	else
		if (GMHI[index] > 0)
			_GMHI[index] = GMHI[index] - 1;

}


void  update_mhiGPU(Mat& mhi, const Mat& img, Mat& dst, int diff_threshold)
{
	Size size = img.size();
	int i, idx1 = last;
	Rect comp_rect;
	double count;
	double angle;
	Point center;
	double magnitude;
	Scalar color;
	Mat silh;

	if (mhi.size() != size)
	{
		mhi = Mat::zeros(size, CV_8U);
		buf[0] = Mat::zeros(size, CV_8U);
		buf[1] = Mat::zeros(size, CV_8U);
	}
	//cout << "buf[0] " << buf[0].size() << "\n";
	//cout << "buf[0] " << buf[1].size() << "\n";

	//cvtColor(img, buf[last], COLOR_BGR2GRAY); // convert frame to grayscale
    cuda:cvtColor(img, buf[last], COLOR_BGR2GRAY);

	int idx2 = (last + 1) % 2; // index of (last - (N-1))th frame
	last = idx2;

	silh = Mat::zeros(size, CV_8U);
	//absdiff(buf[idx2], buf[idx1], silh);
	cuda::absdiff(buf[idx2], buf[idx1], silh);

	// Threshold for small points
	//threshold(silh, silh, 80, 255, CV_THRESH_BINARY);
	cuda::threshold(silh, silh, 80, 255, CV_THRESH_BINARY);

	// Convert to Binary
	silh.convertTo(silh, CV_8U, 1.0 / 255.5);  //CV_32FC3
	normalize(silh, silh, 0, 1, CV_MINMAX);

	// Calculate Motion History H
	dim3 block(BX, BY);
	//dim3 grid((DX + block.x - 1) / block.x, (DY + block.y - 1) / block.y);
	dim3 grid(DX / block.x, DY / block.y);
	cuda::GpuMat _GMHI, GMHI, GSILH;

	Mat _mhi = Mat::zeros(size, CV_8U);
	dst = Mat::zeros(size, CV_8U);
	_GMHI.upload(_mhi);
	GMHI.upload(mhi);
	GSILH.upload(silh);
	//_Update_MHI_GPU << <numBlocks, threadsPerBlock >> > ((uchar *)_GMHI.data, (uchar *)GMHI.data, (uchar *)GSILH.data, size.height, size.width, MHI_DURATION);
	//high_resolution_clock::time_point t1 = high_resolution_clock::now();
	//_Update_MHI_GPU << <grid, block >> > ((uchar *)_GMHI.data, (uchar *)GMHI.data, (uchar *)GSILH.data, size.height, size.width, MHI_DURATION);
	_Update_MHI_GPU << <grid, block >> > ((uchar *)_GMHI.data, (uchar *)GMHI.data, (uchar *)GSILH.data, GMHI.step, size.height, size.width, MHI_DURATION);
	//high_resolution_clock::time_point t2 = high_resolution_clock::now();
	//duration<double, std::milli> time_span = t2 - t1;
	//cout << "*** GPU MHI time for H [" << time_span.count() << "] milliseconds.\n";
	_GMHI.download(mhi);
	dst = mhi.clone();
	//dst = mhi;
}
