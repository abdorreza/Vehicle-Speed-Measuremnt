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
#include <cstring>
#include "motion_detection.h"


using namespace cv;
using namespace std;
using namespace cv::motempl;

void  update_mhi(const Mat& img, Mat& dst, int diff_threshold)
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
		mhi = Mat::zeros(size, CV_8UC1);
		zplane = Mat::zeros(size, CV_8U);

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
	_mhi = Mat::zeros(size, CV_8UC1);
	for (int y=0; y<silh.rows; y++)
		for (int x=0; x < silh.cols; x++)
		{
			if (silh.at<uchar>(y, x) == 1)
				_mhi.at<uchar>(y, x) = MHI_DURATION;
			else
				if ( mhi.at<uchar>(y,x)!=0 )
					_mhi.at<uchar>(y, x) = mhi.at<uchar>(y, x) - 1;
				else
					_mhi.at<uchar>(y, x) = 0;
		}

	mhi=_mhi;
	dst = mhi;
}

void  Segmentation(const Mat& motion, Mat& dst)
{
	// Calculate Mask M
	segmask = Mat::zeros(motion.size(), CV_8UC1);
	dst = Mat::zeros(motion.size(), CV_8UC1);
	for (int y = 0; y < motion.rows; y++)
		for (int x = 0; x < motion.cols; x++)
		{
			if (motion.at<uchar>(y, x) > 0)
				dst.at<uchar>(y, x) = 1;
			else
				dst.at<uchar>(y, x) = 0;
		}
}

void  Projection(const Mat& _segment,_vehicle_boundaries &VBoundaries)
{
	// Calculate Vertical Projection
	VBoundaries.left = 0;
	VBoundaries.right = 0;
	VBoundaries.bottom = -10000;
	VBoundaries.top = 10000;
	ver_proj.clear();
	ver_proj.resize(_segment.cols,0);
	//for (int i = 0; i<_segment.cols; ++i)
		//ver_proj.at(i) = 0;
	int ver_max = 0;
	for (int x = 0; x<_segment.cols; ++x)
		//for (int y = 0; y <_segment.rows ; ++y)
			for (int y = _segment.rows-1; y>=0; --y)
			{
			if (_segment.at<uchar>(y, x) != 0)
			{
				if (VBoundaries.top > y && y>_segment.rows/2) VBoundaries.top = y;
				if (VBoundaries.bottom < y) VBoundaries.bottom = y;
				ver_proj.at(x)++;
				if (ver_proj.at(x) > ver_max)
					ver_max = ver_proj.at(x);
			}
		}
	// Show Vertical Projection
	namedWindow("Vertical projection", CV_WINDOW_AUTOSIZE);
	//////resizeWindow("Vertical projection", 960,100);
	
	IplImage *ver_proj_view;
	ver_proj_view = cvCreateImage(Size(segment.cols, 100), 8, 1);
	cvZero(ver_proj_view);
	for (int i = 0; i < segment.cols; ++i)
		if ( ver_max!=0 ) 
		for (int j = 0; j < ver_proj.at(i) * 100 / ver_max; ++j)
			cvSetReal2D(ver_proj_view, 99-j, i, 255);
	cvShowImage("Vertical projection", ver_proj_view);
	
}


void Find_Inclination(double const FFT)
{
	F.clear();
	R.clear();
	F.resize(ver_proj.size());
	R.resize(ver_proj.size());
	// array initialization
	for (int x = 0; x < ver_proj.size(); x++)
		F.at(x) = R.at(x) = 0;
	for (int x = 0; x < ver_proj.size()-1; x++)
	{
			double j;
			if (ver_proj.at(x + 1) != 0)
				j = (1 - (ver_proj.at(x) / ver_proj.at(x + 1)));
			else
				j = -1;
			if (j > FFT)
				R.at(x) = 1;
			if (j < FFT*-1)
				F.at(x) = 1;
	}
}

void PHASES(String type)
{
	if (type == "Ascending")
	{
		Sa.clear();
		Sa.resize(ver_proj.size());
		for (int x = 0; x < ver_proj.size(); x++)
			Sa.at(x) = 0;
		for (int x = 1; x < ver_proj.size() - 1; x++)
		{
			if (R.at(x) == 0 && R.at(x + 1) == 1)
				Sa.at(x) = 1;
			else
				if (F.at(x) == 0 && F.at(x + 1) == 1)
					Sa.at(x) = 0;
				else
					Sa.at(x) = Sa.at(x - 1);
		}

	}
	if (type == "Descending")
	{
		Sd.clear();
		reverse(R.begin(), R.end());
		reverse(F.begin(), F.end());
		Sd.resize(ver_proj.size());
		for (int x = 0; x < ver_proj.size(); x++)
			Sd.at(x) = 0;
		for (int x = 1; x < ver_proj.size() - 1; x++)
		{
			if (R.at(x) == 0 && R.at(x + 1) == 1)
				Sd.at(x) = 1;
			else
				if (F.at(x) == 0 && F.at(x + 1) == 1)
					Sd.at(x) = 0;
				else
					Sd.at(x) = Sd.at(x - 1);
		}
		//reverse(R.begin(), R.end());
		//reverse(F.begin(), F.end());
	}
}


void Find_Hills(double const FFT)
{
	Find_Inclination(FIND_HILLS_THRE);
	PHASES("Ascending"); //Ascending
	PHASES("Descending"); //Descending
	A.clear();
	D.clear();
	A.resize(ver_proj.size());
	D.resize(ver_proj.size());
	int andisA=0, andisD=0;
	for (int x = 0; x <= ver_proj.size() - 2; x++)
	{
		if (Sa.at(x) == 0 && Sa.at(x+1) == 1)
		{
			A.at(andisA) = x;
			andisA++;
		}
		if (Sd.at(x) == 1 && Sd.at(x+1) == 0)
		{
			D.at(andisD) = x;
			andisD++;
		}
	}
	A.resize(andisA);
	D.resize(andisD);
}


void ProjectionSmooth(void)
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


void FindBounderies(void)
{
	A.clear();
	D.clear();
	A.resize(ver_proj.size());
	D.resize(ver_proj.size());
	int andisA = -1, andisD = -1, FindStart = 0, FindEnd = 1;
	for (int x = 0; x <= ver_proj.size()-1; x++)
	{
		if (ver_proj.at(x) != 0 && FindStart == 0)
		{
			andisA++;
			A.at(andisA) = x;
			FindStart = 1;
			FindEnd = 0;
		}
		if (ver_proj.at(x) == 0 && FindEnd == 0)
		{
			andisD++;
			D.at(andisD) = x;
			FindStart = 0;
			FindEnd = 1;
		}
		if (x >= ver_proj.size() - 1 && FindStart == 1 && FindEnd == 0)
		{
			andisD++;
			D.at(andisD) = x;
			FindStart = 0;
			FindEnd = 1;
		}
	}
	if ( andisA!=-1 )
		A.resize(andisA+1);
	if ( andisD!=-1 )
		D.resize(andisD+1);
}


void Print_Vertical_Projection(vector<int> ver_proj)
{
	printf("\n\nVer_Proj Size : %i\n\n", ver_proj.size());
	for (int i = 0; i < ver_proj.size(); i++)
		printf("Ver_Proj[%i]=%i\n", i, ver_proj.at(i));
}

void Get_Boundaries(vector<int> ver_proj,int lane, _vehicle_boundaries &VBoundaries)
{
	int start, end;
	A.clear();
	D.clear();
	A.resize(3,0);
	D.resize(3,0);
	VBoundaries.left = -1;
	VBoundaries.right = -1;
	if (lane == 3) { start = 1000; end = ver_proj.size() - 21; }
	else
	if (lane == 2) { start = 500; end = 1200; }
	else
	if (lane == 1) { start = 0; end = 500; }
	for (int i = start; i < end; i++)
	{
		if (ver_proj.at(i) == 0 && ver_proj.at(i + 1) != 0)
		{
			A.at(0) = i;
			VBoundaries.left = i;
		}
		else
		if (ver_proj.at(i) != 0 && ver_proj.at(i + 1) == 0)
		{
			D.at(0) = i;
			VBoundaries.right = i;
		}
	}
	cout <<"LEFT"<< VBoundaries.left << "\n";
	if (D.at(0) == 0) D.at(0) = ver_proj.size()-1;
	if (VBoundaries.left == -1) VBoundaries.left = start;
	if (VBoundaries.right == -1) VBoundaries.right = end;
	if (VBoundaries.left > start + 400) VBoundaries.left = start;
	if (VBoundaries.right > end) VBoundaries.right = end;
	if (VBoundaries.left > VBoundaries.right) VBoundaries.right = end;
}

void Open_XML_File(string FileName)
{
	int andis = 1;
	inFile.open(FileName);
}

struct _XML Read_XML_File(void)
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
			cout << str << "\n";
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

