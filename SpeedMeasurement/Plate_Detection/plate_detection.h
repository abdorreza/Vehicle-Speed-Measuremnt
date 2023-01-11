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

using namespace cv;
using namespace std;
using namespace cv::motempl;


// various tracking parameters (in seconds)
//const double MHI_DURATION = 5;
//const double MAX_TIME_DELTA = 0.5;
//const double MIN_TIME_DELTA = 0.05;
//const int DELTA_T = 1;
//const double DIFF_THRESHOLD = 0.17;
//const double FIND_HILLS_THRE = 0.1;
const double MAGNITUDE_VALUE = 2;  //t
// number of cyclic frame buffer used for motion detection
// (should, probably, depend on FPS)



void Vertical_Edge(const Mat& img, Mat& dst, const bool Show_Vertical_Edge);

int max1(int n1, int n2, int n3, int n4);

int max2(int n1, int n2, int n3, int n4, int n5, int n6);

void Dilate(const Mat& ComponentImage, Mat& dst, const bool Show_Dilate_Image);

int Single(int n1, int n2, int n3, int n4);

int Min(int n1, int n2, int n3, int n4);

void DeleteExtraRegions(Mat& dst, int max_candid, vector< vector<int> > candids_regios, const bool Show_Plate_Dilate_Candidate);

void FindRegionsColsCenter(Mat& dlt, int max_candid, vector< vector<int> > candids_regios);

int GetCenter(const Mat& dlt, int col, int row1, int row2);

int Find_Plate_Candid_Content_Text(Mat croppedImage, vector< vector<int> >& candids_regions, const bool Show_Plate_Region);

void DeleteSmallLargeComponent(Mat& dst, Mat& stats, int _short, int _long);

int FindSet(int b, vector<int> Father);

void Union(int b1, int b2, const Mat stats, vector<int>& Father);

void FindCandidatesPlateRegions(const Mat cmp, const Mat stats, vector<int>& Father);

int GetRegions(const Mat stats, vector<int> Father, vector< vector<int> >& candids_regions);

void Show_Plate_Candids(const Mat croppedImage, const vector< vector<int> > candids_regions, const bool Show_Plate_Candidate);
