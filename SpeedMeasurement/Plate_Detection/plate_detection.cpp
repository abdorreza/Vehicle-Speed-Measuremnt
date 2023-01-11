#include "opencv2/optflow.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <time.h>
#include <stdio.h>
#include <chrono>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include "plate_detection.h"
#include "..\TextDetection\TextDetection.h"

using namespace cv;
using namespace std;
using namespace cv::motempl;
using namespace std::chrono;




void Vertical_Edge(const Mat& img, Mat& dst,const bool Show_Vertical_Edge)
{
	Mat ImageGray, VSobelImage, FilterImage, FilterOut, tmp;
	int new_pixel = 0;
	int andis = 0;

	double noise = 0.03;
	double eps = sqrt(2.0)*noise;
	double eps2 = eps * eps;

	int sobel_v[3][3] = {
		{ -1, 0, 1 },
		{ -2, 0, 2 },
		{ -1, 0, 1 }
	};
	double magnitude = 0.0;
	VSobelImage = Mat::zeros(img.size(), CV_8U);
	dst = Mat::zeros(img.size(), CV_8U);
	cvtColor(img, ImageGray, COLOR_BGR2GRAY);
	// Calculate Vertical Sobel Edges
	for (int i = 0; i < ImageGray.size().height - 2; i++)
		for (int j = 0; j < ImageGray.size().width - 2; j++)
		{
			new_pixel = 0;
			new_pixel = (ImageGray.at<uchar>(i, j + 2)) + (2 * ImageGray.at<uchar>(i + 1, j + 2)) + (ImageGray.at<uchar>(i + 2, j + 2)) +
				(-1 * ImageGray.at<uchar>(i, j)) + (-2 * ImageGray.at<uchar>(i + 1, j)) + (-1 * ImageGray.at<uchar>(i + 2, j));
			//new_pixel /= 4.0;
			double d2 = new_pixel*new_pixel;
			if (d2 <= eps2)
				VSobelImage.at<uchar>(i + 1, j + 1) = 0;
			else
				VSobelImage.at<uchar>(i + 1, j + 1) = sqrt(d2 - eps2);
			magnitude += VSobelImage.at<uchar>(i + 1, j + 1);
			andis++;
		}

	// Calculate Edge Image Average Magnitude
	magnitude /= (ImageGray.size().height*ImageGray.size().width);
	// Convert Edge to 0&1 and Save in MagImage
	for (int i = 0; i < VSobelImage.size().height; i++)
		for (int j = 0; j < VSobelImage.size().width; j++)
		{
			if (abs(VSobelImage.at<uchar>(i, j)) > magnitude*MAGNITUDE_VALUE)
				dst.at<uchar>(i, j) = 1;
			else
				dst.at<uchar>(i, j) = 0;
		}
	if (Show_Vertical_Edge == true)
	{
		namedWindow("Vertical Edge", WINDOW_NORMAL);
		resizeWindow("Vertical Edge", dst.size().width / 2, dst.size().height / 2);
		imshow("Vertical Edge", dst * 255);
	}

}


int max1(int n1, int n2, int n3, int n4)
{
	int max = -1;
	if (n1 > max) max = n1;
	if (n2 > max) max = n2;
	if (n3 > max) max = n3;
	if (n4 > max) max = n4;
	return max;
}

int max2(int n1, int n2, int n3, int n4, int n5, int n6)
{
	int max = -1;
	if (n1 > max) max = n1;
	if (n2 > max) max = n2;
	if (n3 > max) max = n3;
	if (n4 > max) max = n4;
	if (n5 > max) max = n5;
	if (n6 > max) max = n6;
	return max;
}


void Dilate(const Mat& ComponentImage, Mat& dst,const bool Show_Dilate_Image)
{

	dst = Mat::zeros(ComponentImage.size(), CV_8U);
	// Horiziontal Dilation
	for (int i = 0; i < ComponentImage.size().height; i++)
		for (int j = 0; j < ComponentImage.size().width - 7; j++)
		{
			if (ComponentImage.at<int>(i, j) + ComponentImage.at<int>(i, j + 1) + ComponentImage.at<int>(i, j + 2) + ComponentImage.at<int>(i, j + 3) + ComponentImage.at<int>(i, j + 4) + ComponentImage.at<int>(i, j + 5) + ComponentImage.at<int>(i, j + 6) != 0)
				dst.at<uchar>(i, j + 3) = 170;
		}
	if (Show_Dilate_Image == true)
	{
		namedWindow("Dilate", WINDOW_NORMAL);
		resizeWindow("Dilate", dst.size().width / 2, dst.size().height / 2);
		imshow("Dilate", dst * 255);
	}

}


int Single(int n1, int n2, int n3, int n4)
{
	int l;
	if (n1 != 0) l = n1;
	if (n2 != 0) l = n2;
	if (n3 != 0) l = n3;
	if (n4 != 0) l = n4;
	if (n1 != 0 && n1 != l) l = 0;
	else
	if (n2 != 0 && n2 != l) l = 0;
	else
	if (n3 != 0 && n3 != l) l = 0;
	else
	if (n4 != 0 && n4 != l) l = 0;
	return l; // 0 : isn't single
}

int Min(int n1, int n2, int n3, int n4)
{
	int min = INT_MAX;
	if (n1 < min && n1 != 0) min = n1;
	if (n2 < min && n2 != 0) min = n2;
	if (n3 < min && n3 != 0) min = n3;
	if (n4 < min && n4 != 0) min = n4;
	return min;
}



void DeleteExtraRegions(Mat& dst,int max_candid, vector< vector<int> > candids_regios,const bool Show_Plate_Dilate_Candidate)
{
	int flag;
	for (int i = 0; i < dst.size().height; i++)
		for (int j = 0; j < dst.size().width; j++)
		{
			flag = 0;
			for (int c = 0; c < max_candid; c++)
				if (j >= candids_regios[c][0] && j <= candids_regios[c][2] && i >= candids_regios[c][1] && i <= candids_regios[c][3])
					flag = 1;
			if (flag == 0)
				dst.at<uchar>(i, j) = 0;
		}
	if (Show_Plate_Dilate_Candidate == true)
	{
		Mat tmp;
		dst.copyTo(tmp);
		for (int i = 0; i < max_candid; i++)
			rectangle(tmp, cvPoint(candids_regios[i][0], candids_regios[i][1]), cvPoint(candids_regios[i][2], candids_regios[i][3]), CV_RGB(255, 255, 0), 1, 8);
		namedWindow("Regions Color", WINDOW_NORMAL);
		resizeWindow("Regions Color", tmp.size().width / 2, tmp.size().height / 2);
		imshow("Regions Color", tmp);
	}

}


void FindRegionsColsCenter(Mat& dlt, int max_candid, vector< vector<int> > candids_regios)
{
	int fp, ep;
	for (int c = 0; c < max_candid; c++)
		for (int j = candids_regios[c][0]; j <= candids_regios[c][2]; j++)
		{
			fp = ep = 0;
			for (int i = candids_regios[c][1]; i <= candids_regios[c][3]; i++)
				if (dlt.at<uchar>(i, j) != 0 && fp == 0)
					fp = i;
				else
					if (dlt.at<uchar>(i, j) != 0 && fp != 0)
						ep = i;
			if (fp != 0 && ep != 0)
				dlt.at<uchar>((fp + ep) / 2, j) = 255;
		}
}



int GetCenter(const Mat& dlt, int col, int row1, int row2)
{
	for (int i = row1; i <= row2; i++)
		if (dlt.at<uchar>(i, col) == 255)
			return i;
	return -1;
}



int Find_Plate_Candid_Content_Text(Mat croppedImage, vector< vector<int> >& candids_regions, const bool Show_Plate_Region)
{
	int nrows, ncols, andis=1;
	vector< vector<int> > tmp_regions;
	nrows = croppedImage.size().height;
	ncols = croppedImage.size().width;
	double Classification;
	unsigned char *img = croppedImage.data;
	unsigned char *image = convert_rgb_to_gray(img, nrows, ncols);
	//cvtColor(croppedImage, croppedImage, COLOR_BGR2GRAY);
	//unsigned char *image = croppedImage.data;

	struct svm_model *model = svm_load_model(SVM_MODEL_FILE);

	int nr_class = svm_get_nr_class(model);
	double *prob = (double *)malloc(nr_class * sizeof(double));

	if (svm_check_probability_model(model) == 0)
	{
		fprintf(stderr, "Model does not support probabiliy estimates\n");
		exit(1);
	}
	struct_thog sthog = load_settings(THOG_SETTINGS_FILE);
	int x1, y1, x2, y2;
	//high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (int i = 0; i < candids_regions.size(); i++)
	{
		Classification = classify(image, nrows, ncols, candids_regions[i][0], candids_regions[i][1], (candids_regions[i][2] - candids_regions[i][0]), (candids_regions[i][3] - candids_regions[i][1]), model, prob, sthog);
		if (Classification == 1)
		{
			tmp_regions.resize(andis);
			tmp_regions[andis-1] = candids_regions[i];
			andis++;
		}
	}
	//high_resolution_clock::time_point t2 = high_resolution_clock::now();
	//duration<double, std::milli> time_span = t2 - t1;
	//cout << "*** CPU time for Find Plates Regions Contain Text [" << time_span.count() << "] milliseconds.\n";

	candids_regions.clear();
	candids_regions = tmp_regions;
	if (candids_regions.size() > 0)
	{
		tmp_regions.resize(1);
		// Erase Top Candids
		if (candids_regions.size() > 1)
		{
			int max = 0;
			for (int i = 0; i < candids_regions.size(); i++)
			{
				//if (candids_regions[i][1] > max && candids_regions[i][3] < croppedImage.rows - 20)
				if (candids_regions[i][1] > max)
				{
					max = candids_regions[i][1];
					tmp_regions[0] = candids_regions[i];
				}
			}

		}
		candids_regions.clear();
		candids_regions = tmp_regions;
	}
	if (candids_regions.size() <= 0)
		return 0;
	if (Show_Plate_Region == true)
	{
		Mat tmp;
		croppedImage.copyTo(tmp);
		for (int i = 0; i < candids_regions.size(); i++)
			rectangle(tmp, cvPoint(candids_regions[i][0], candids_regions[i][1]), cvPoint(candids_regions[i][2], candids_regions[i][3]), CV_RGB(51, 255, 85), 3, 8, 0);
		namedWindow("Plate Region", WINDOW_NORMAL);
		moveWindow("Plate Region", 710, 0);
		//resizeWindow("Plate Region", croppedImage.size().width / 2, croppedImage.size().height / 2);
		resizeWindow("Plate Region", 300, 200);
		//moveWindow("Plate Region", 1000,0);
		imshow("Plate Region", tmp);
	}
	return 1;
}



void DeleteSmallLargeComponent(Mat& dst, Mat& stats, int _short, int _long)
{
	/*
	Mat New_Stats;
	int andis = 0;
	New_Stats = Mat::zeros(stats.size(), stats.type());
	for (int i = 0; i < stats.rows; i++)
	{
		if ((stats.at<int>(i, 2) >= _short && stats.at<int>(i, 3) >= _short) && (stats.at<int>(i, 2) <= _long && stats.at<int>(i, 3) <= _long))
		{
			stats.row(i).copyTo(New_Stats.row(andis));
			andis++;
		}
	}
	stats = New_Stats(Rect(0, 0, 4, andis - 1));
	*/
	/*////////////////////
	for (int i = 0; i<dst.rows; i++)
		for (int j = 0; j<dst.cols; j++)
			if ( stats.at<int>(dst.at<int>(i, j),2)<_short || stats.at<int>(dst.at<int>(i, j), 3)<_short || stats.at<int>(dst.at<int>(i, j), 2)>_long || stats.at<int>(dst.at<int>(i, j), 3)>_long)
				dst.at<int>(i, j) = 0;
	*/////////////////////
	int andis = 0;
	Mat New_Stats = Mat::zeros(stats.size(), stats.type());
	for (int i = 0; i < stats.rows; i++)
	{
		if (stats.at<int>(i, 2) < _short || stats.at<int>(i, 3) < _short || stats.at<int>(i, 2) > _long || stats.at<int>(i, 3) > _long)
		{
			for (int j = stats.at<int>(i, 0); j < stats.at<int>(i, 0) + stats.at<int>(i, 2); j++)
				for (int k = stats.at<int>(i, 1); k < stats.at<int>(i, 1) + stats.at<int>(i, 3); k++)
					if (dst.at<int>(k, j) == i)
						dst.at<int>(k, j) = 0;
		}
		else
		{
			stats.row(i).copyTo(New_Stats.row(andis));
			andis++;
		}
	}

	stats = New_Stats(Rect(0, 0, 5, andis));
	/*
	cout << "Andis = " << andis << "\n";
	cout << "Numbers = " << stats.rows << "\n\n\n";
	for (int i = 0; i < stats.rows; i++)
	{
		cout << "Label = " << i << "\n";
		cout << "Stats(0,0) = " << stats.at<int>(i, 0) << "\n";
		cout << "Stats(0,1) = " << stats.at<int>(i, 1) << "\n";
		cout << "Stats(0,2) = " << stats.at<int>(i, 2) << "\n";
		cout << "Stats(0,3) = " << stats.at<int>(i, 3) << "\n";
		cout << "Stats(0,4) = " << stats.at<int>(i, 4) << "\n";
		cout << "--------------------------------------------\n";
	}
	*/
}


int FindSet(int b, vector<int> Father)
{
	if (Father[b] == b)
		return b;
	else
		return FindSet(Father[b], Father);
}

void Union(int b1, int b2, const Mat stats, vector<int>& Father)
{
	int cv1, cv2;
	// float t1 = 2, t2 = 2.5, t3 = 1.5;
	// float t1 = 0.4, t2 = 0.9, t3 = 1.5;
	float t1 = 0.7, t2 = 1.1, t3 = 0.4;
	int f1 = FindSet(b1, Father);
	int f2 = FindSet(b2, Father);
	if (f1 != f2)
	{
		int h1 = stats.at<int>(b1, 3);    // Box1 Heigh
		int w1 = stats.at<int>(b1, 2);    // Box1 Width
		int x1 = stats.at<int>(b1, 0);    // Box1 Width Center
		int y1 = stats.at<int>(b1, 1);    // Box1 Height Center

		int h2 = stats.at<int>(b2, 3);    // Box2 Heigh
		int w2 = stats.at<int>(b2, 2);    // Box2 Width
		int x2 = stats.at<int>(b2, 0);    // Box2 Width Center
		int y2 = stats.at<int>(b2, 1);    // Box2 Height Center


		int h = h1;
		if (h2 < h) h = h2;  // h = Min(h1,h2)
							 //int h = min(h1, h2);

		int dx = abs(x1 - x2) - (w1 + w2) / 2;
		int dy = abs(y1 - y2);
		if (abs(h1 - h2) < t1*h && dx < t2*h && dy < t3*h)
		{
			int tmp = Father[f2];
			Father[f2] = f1;
			//for (int s = 0; s < stats.rows; s++)
				//if (Father[s] == tmp)
					//Father[s] = f1;
		}
	}
}





void FindCandidatesPlateRegions(const Mat cmp, const Mat stats, vector<int>& Father)
{
	int b1, b2;
	Father.resize(stats.rows);
	// Make-Set
	//printf("\n\n*** Stats cols and rows : [%i , %i] ***\n\n", stats.cols, stats.rows);
	for (int i = 0; i < Father.size(); i++)
		Father[i] = i;
	//for (int i = 19; i < stats.rows - 20; i++)
	//	for (int j = i - 19; j < i + 20; j++)
	for (int i = 4; i < stats.rows - 4; i++)
		for (int j = i - 4; j < i + 4; j++)
		{
			b1 = i;
			b2 = j;
			Union(b1, b2, stats, Father);
		}
}

int GetRegions(const Mat stats, vector<int> Father ,vector< vector<int> >& candids_regions)
{
	int x1, y1, x2, y2, i, j, s, exist = 0;
	int max_candid = 0;
	vector< vector<int> > temp;
	for (i = 1; i < stats.rows; i++)
	{
		if (Father[i] != -1)
		{
			candids_regions.resize(max_candid + 1, vector<int>(4, 0));
			candids_regions[max_candid][0] = stats.at<int>(i, 0); //x1
			candids_regions[max_candid][1] = stats.at<int>(i, 1); //y1
			candids_regions[max_candid][2] = stats.at<int>(i, 0) + stats.at<int>(i, 2); //x2
			candids_regions[max_candid][3] = stats.at<int>(i, 1) + stats.at<int>(i, 3); //y2
			for (j = i + 1; j < stats.rows; j++)
			{
				if (Father[j] == Father[i] && Father[j] != -1)
				{
					x1 = stats.at<int>(j, 0); //x1
					y1 = stats.at<int>(j, 1); //y1
					x2 = stats.at<int>(j, 0) + stats.at<int>(j, 2); //x2
					y2 = stats.at<int>(j, 1) + stats.at<int>(j, 3); //y2
					if (x1 < candids_regions[max_candid][0]) candids_regions[max_candid][0] = x1;
					if (y1 < candids_regions[max_candid][1]) candids_regions[max_candid][1] = y1;
					if (x2 > candids_regions[max_candid][2]) candids_regions[max_candid][2] = x2;
					if (y2 > candids_regions[max_candid][3]) candids_regions[max_candid][3] = y2;
					Father[j] = -1;
				}
			}
			// Increase Candid Borders for Containing all Pixels
			max_candid++;
			Father[i] = -1;
		}
	}
	int andis = 0;
	for (int i = 0; i < candids_regions.size(); i++)
	{
		int w = abs(candids_regions[i][0] - candids_regions[i][2]) + 1;
		int h = abs(candids_regions[i][1] - candids_regions[i][3]) + 1;
		if (w >= 55 && h >= 20)
		{
			temp.resize(andis + 1);
			temp[andis] = candids_regions[i];
			andis++;
		}
	}
	candids_regions.clear();
	candids_regions = temp;
	return andis;
}


void Show_Plate_Candids(const Mat croppedImage, const vector< vector<int> > candids_regions, const bool Show_Plate_Candidate)
{
	if (Show_Plate_Candidate == true)
	{
		Mat tmp;
		croppedImage.copyTo(tmp);
		for (int i = 0; i < candids_regions.size(); i++)
			rectangle(tmp, cvPoint(candids_regions[i][0], candids_regions[i][1]), cvPoint(candids_regions[i][2], candids_regions[i][3]), CV_RGB(255, 255, 0), 3, 8, 0);
		namedWindow("Plate Candidates", WINDOW_NORMAL);
		moveWindow("Plate Candidates", 405, 0);
		//resizeWindow("Plate Candidates", croppedImage.size().width / 2, croppedImage.size().height / 2);
		resizeWindow("Plate Candidates", 300, 200);
		imshow("Plate Candidates", tmp);
		imwrite("candids_regions.jpg", tmp);
	}

}