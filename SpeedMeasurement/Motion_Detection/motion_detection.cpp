#include "opencv2/optflow.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <time.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <chrono>

#include "opencv2\core\cuda.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\cudalegacy.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2\xfeatures2d\cuda.hpp"
#include "opencv2\cudaimgproc.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"


#include "motion_detection.h"
#include "..\CUDA\CUDA_functions.cuh"

using namespace cv;
using namespace std;
using namespace cv::motempl;
using namespace std::chrono;


void  update_mhi(Mat& mhi,const Mat& img, Mat& dst, int diff_threshold)
{
	double timestamp = (double)clock() / CLOCKS_PER_SEC; // get current time in seconds
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
		//mhi = Mat::zeros(size, CV_8UC1);
		mhi = Mat::zeros(size, CV_8U);
		buf[0] = Mat::zeros(size, CV_8U);
		buf[1] = Mat::zeros(size, CV_8U);
	}

	cvtColor(img, buf[last], COLOR_BGR2GRAY); // convert frame to grayscale

	int idx2 = (last + 1) % 2; // index of (last - (N-1))th frame
	last = idx2;

	silh = Mat::zeros(size, CV_8UC1);
	absdiff(buf[idx2], buf[idx1], silh);

	// Threshold for small points
	threshold(silh, silh, 80, 255, CV_THRESH_BINARY);

	// Convert to Binary
	silh.convertTo(silh, CV_8UC1, 1.0 / 255.5);  //CV_32FC3
	normalize(silh, silh, 0, 1, CV_MINMAX);

	// Calculate Motion History H
	Mat _mhi = Mat::zeros(size, CV_8UC1);
	//high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (int y=0; y<silh.rows; y++)  // y+=4
		for (int x=0; x < silh.cols; x++)  //x+=4
		{
			if (silh.at<uchar>(y, x) == 1)
				_mhi.at<uchar>(y, x) = MHI_DURATION;
			else
				if (mhi.at<uchar>(y, x) > 0)
					_mhi.at<uchar>(y, x) = mhi.at<uchar>(y, x) - 1;
		}
	//high_resolution_clock::time_point t2 = high_resolution_clock::now();
	//duration<double, std::milli> time_span = t2 - t1;
	//cout << "*** CPU MHI time for H [" << time_span.count() << "] milliseconds.\n";

	mhi=_mhi;
	dst = mhi;
}

void  update_mhi_GPU(Mat& mhi, const Mat& img, Mat& dst, int diff_threshold)
{
	//update_mhiGPU(mhi, img, dst, diff_threshold, last, buf, MHI_DURATION);
	update_mhiGPU(mhi, img, dst, diff_threshold);
}


void  Segmentation(const Mat& motion, Mat& dst)
{
	// Calculate Mask M
	Mat segmask = Mat::zeros(motion.size(), CV_8UC1);
	dst = Mat::zeros(motion.size(), CV_8UC1);
	//high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (int y = 0; y < motion.rows; y++)
		for (int x = 0; x < motion.cols; x++)
			//for (int y = 0; y < motion.rows; y += 4)
				//for (int x = 0; x < motion.cols; x += 4)
				{
			if (motion.at<uchar>(y, x) > 0)
				dst.at<uchar>(y, x) = 1;
			else
				dst.at<uchar>(y, x) = 0;
		}
	//high_resolution_clock::time_point t2 = high_resolution_clock::now();
	//duration<double, std::milli> time_span = t2 - t1;
	//cout << "*** CPU MHI time for M [" << time_span.count() << "] milliseconds.\n";
}

/*void SegmentationGPU(const Mat& motion, Mat& dst)
{
	int max_blocks = 2048;              //maximum number of active blocks per SM
	int max_threads_per_Block = 1015;   //maximum number of active threads per SM
	dim3 threadsPerBlock(max_threads_per_Block);
	dim3 numBlocks(max_blocks);
	cuda::GpuMat GMat;
	dst = Mat::zeros(motion.size(), CV_8UC1);
	GMat.upload(dst);
	SegmentationBy_GPU << <numBlocks, threadsPerBlock >> > ((int *)GMat.data, dst.rows, dst.cols); //Kernel invocation
	GMat.download(dst);
}*/

vector<int> Projection(const Mat& _segment,_vehicle_boundaries &VBoundaries, const bool Show_Vertical_Projection)
{
	// Calculate Vertical Projection
	VBoundaries.left = 0;
	VBoundaries.right = 0;
	VBoundaries.bottom = -10000;
	VBoundaries.top = 10000;
	vector<int> ver_proj;
	ver_proj.clear();
	ver_proj.resize(_segment.cols,0);
	int ver_max = 0;
	//high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (int x = 0; x<_segment.cols; ++x)
		//for (int y = 0; y <_segment.rows ; ++y)
			for (int y = _segment.rows-1; y>=0; --y)
			{
			if (_segment.at<uchar>(y, x) != 0)
			{
				if (VBoundaries.top > y && y>_segment.rows/2) VBoundaries.top = y-100;
				if (VBoundaries.bottom < y) VBoundaries.bottom = y;
				ver_proj.at(x)++;
				if (ver_proj.at(x) > ver_max)
					ver_max = ver_proj.at(x);
			}
		}
	//high_resolution_clock::time_point t2 = high_resolution_clock::now();
	//duration<double, std::milli> time_span = t2 - t1;
	//cout << "*** CPU time for Vertical Projection [" << time_span.count() << "] milliseconds.\n";

	// Show Vertical Projection
	if (Show_Vertical_Projection == true)
	{
		namedWindow("Vertical projection", CV_WINDOW_AUTOSIZE);
		moveWindow("Vertical projection", 0, 405);
		//////resizeWindow("Vertical projection", 960,100);
		IplImage *ver_proj_view;
		ver_proj_view = cvCreateImage(Size(_segment.cols, 100), 8, 1);
		cvZero(ver_proj_view);
		for (int i = 0; i < _segment.cols; ++i)
			if (ver_max != 0)
				for (int j = 0; j < ver_proj.at(i) * 100 / ver_max; ++j)
					cvSetReal2D(ver_proj_view, 99 - j, i, 255);
		cvShowImage("Vertical projection", ver_proj_view);
	}
	return ver_proj;
}


void ProjectionSmooth(vector<int> ver_proj)
{
	//for (int i = 4; i < ver_proj.size() - 5; i++)
		//if (ver_proj.at(i) != 0 && (ver_proj.at(i - 4) == 0 && ver_proj.at(i - 3) == 0 && ver_proj.at(i - 2) == 0 && ver_proj.at(i - 1) == 0 && ver_proj.at(i + 1) == 0 && ver_proj.at(i + 2) == 0 && ver_proj.at(i + 3) == 0 && ver_proj.at(i + 4) == 0))
			//ver_proj.at(i) = 0;
	for (int i = 0; i < ver_proj.size() - 20; i++)
		if (ver_proj.at(i) == 0 && (ver_proj.at(i + 1) != 0 || ver_proj.at(i + 2) != 0 || ver_proj.at(i + 3) != 0 || ver_proj.at(i + 4) != 0 || ver_proj.at(i + 5) != 0 || ver_proj.at(i + 6) != 0 || ver_proj.at(i + 7) != 0 || ver_proj.at(i + 8) != 0 || ver_proj.at(i + 9) != 0 || ver_proj.at(i + 10) != 0 || ver_proj.at(i + 11) != 0 || ver_proj.at(i + 12) != 0 || ver_proj.at(i + 13) != 0 || ver_proj.at(i + 14) != 0 || ver_proj.at(i + 15) != 0 || ver_proj.at(i + 16) != 0 || ver_proj.at(i + 17) != 0 || ver_proj.at(i + 18) != 0 || ver_proj.at(i + 19) != 0 || ver_proj.at(i + 20) != 0))
		if (ver_proj.at(i + 1) != 0) ver_proj.at(i) = ver_proj.at(i + 1);
		else if (ver_proj.at(i + 2) != 0) ver_proj.at(i) = ver_proj.at(i + 2);
		else if (ver_proj.at(i + 3) != 0) ver_proj.at(i) = ver_proj.at(i + 3);
		else if (ver_proj.at(i + 4) != 0) ver_proj.at(i) = ver_proj.at(i + 4);
		else if (ver_proj.at(i + 5) != 0) ver_proj.at(i) = ver_proj.at(i + 5);
		else if (ver_proj.at(i + 6) != 0) ver_proj.at(i) = ver_proj.at(i + 6);
		else if (ver_proj.at(i + 7) != 0) ver_proj.at(i) = ver_proj.at(i + 7);
		else if (ver_proj.at(i + 8) != 0) ver_proj.at(i) = ver_proj.at(i + 8);
		else if (ver_proj.at(i + 9) != 0) ver_proj.at(i) = ver_proj.at(i + 9);
		else if (ver_proj.at(i + 10) != 0) ver_proj.at(i) = ver_proj.at(i + 10);
		else if (ver_proj.at(i + 11) != 0) ver_proj.at(i) = ver_proj.at(i + 11);
		else if (ver_proj.at(i + 12) != 0) ver_proj.at(i) = ver_proj.at(i + 12);
		else if (ver_proj.at(i + 13) != 0) ver_proj.at(i) = ver_proj.at(i + 13);
		else if (ver_proj.at(i + 14) != 0) ver_proj.at(i) = ver_proj.at(i + 14);
		else if (ver_proj.at(i + 15) != 0) ver_proj.at(i) = ver_proj.at(i + 15);
		else if (ver_proj.at(i + 16) != 0) ver_proj.at(i) = ver_proj.at(i + 16);
		else if (ver_proj.at(i + 17) != 0) ver_proj.at(i) = ver_proj.at(i + 17);
		else if (ver_proj.at(i + 18) != 0) ver_proj.at(i) = ver_proj.at(i + 18);
		else if (ver_proj.at(i + 19) != 0) ver_proj.at(i) = ver_proj.at(i + 19);
		else if (ver_proj.at(i + 20) != 0) ver_proj.at(i) = ver_proj.at(i + 20);
}


void Print_Vertical_Projection(vector<int> ver_proj,const bool Print_Vertical_Projections)
{
	if (Print_Vertical_Projections == true)
	{
		printf("\n\nVer_Proj Size : %i\n\n", ver_proj.size());
		for (int i = 0; i < ver_proj.size(); i++)
			printf("Ver_Proj[%i]=%i\n", i, ver_proj.at(i));
	}
}

void Get_Boundaries(vector<int> ver_proj,int lane, _vehicle_boundaries &VBoundaries)
{
	int start, end;
	VBoundaries.left = -1;
	VBoundaries.right = -1;
	if (lane == 3) { start = 1100; end = ver_proj.size() - 1; }
	else
	if (lane == 2) { start = 500; end = 1200; }
	else
	if (lane == 1) { start = 0; end = 500; }
	for (int i = start; i < end; i++)
	{
		if (ver_proj.at(i) == 0 && ver_proj.at(i + 1) != 0)
			VBoundaries.left = i;
		else
		if (ver_proj.at(i) != 0 && ver_proj.at(i + 1) == 0)
			VBoundaries.right = i;
	}
	if (lane == 3) end = ver_proj.size();
	if (VBoundaries.left == -1) VBoundaries.left = start;
	if (VBoundaries.right == -1) VBoundaries.right = end;
	if (VBoundaries.left > start + 400) VBoundaries.left = start;
	if (VBoundaries.right > end) VBoundaries.right = end;
	if (VBoundaries.left > VBoundaries.right) VBoundaries.right = end;
	VBoundaries.top = 600;
}

ifstream Open_XML_File(string FileName)
{
	ifstream inFile;
	int andis = 1;
	inFile.open(FileName);
	return inFile;
}

struct _XML Read_XML_File(ifstream& inFile)
{
	_XML XML;
	string str;
	int andis = 1;
	XML.frame_end = 0;
	XML.speed = 0.0;
	while (inFile)
	{
		inFile >> str;
		str.erase(remove_if(str.begin(), str.end(), isspace), str.end());
		if (str != "</vehicle>")
		{
			//cout << str << "\n";
			if (str.substr(0, 6) == "iframe") XML.frame_start = stoi(str.substr(8, str.length() - 9));
			if (str.substr(0, 4) == "lane") XML.lane = stoi(str.substr(6, str.length() - 7));
			if (str.substr(0, 4) == "moto") if (str.substr(6, str.length() - 7) == "True") XML.moto = true; else XML.moto = false;
			if (str.substr(0, 5) == "plate") if (str.substr(7, str.length() - 8) == "True") XML.plate = true; else XML.plate = false;
			if (str.substr(0, 5) == "radar") if (str.substr(7, str.length() - 8) == "True") XML.radar = true; else XML.radar = false;
			if (str.substr(0, 9) == "frame_end") XML.frame_end = stoi(str.substr(11, str.length() - 12));
			if (str.substr(0, 5) == "speed") XML.speed = stod(str.substr(7, str.length() - 8));
		}
		else
			return XML;
		andis++;
	}
}

void Show_XML(_XML XML)
{
	cout << "\n\n----------------------------------------------------\n";
	cout << "\nFrame Start = " << XML.frame_start << "\n";
	cout << "Lane = " << XML.lane << "\n";
	cout << "Moto = " << boolalpha << XML.moto << "\n";
	cout << "Plate = " << boolalpha << XML.plate << "\n";
	cout << "Radar = " << boolalpha << XML.radar << "\n";
	cout << "Frame End = " << XML.frame_end << "\n";
	cout << "Speed = " << XML.speed << "\n";
	cout << "----------------------------------------------------\n";
}
