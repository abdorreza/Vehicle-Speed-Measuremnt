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

using namespace cv;
using namespace std;
using namespace cv::motempl;


// various tracking parameters (in seconds)
const double MHI_DURATION = 5;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;
const int DELTA_T = 1;
const double DIFF_THRESHOLD = 0.17;
const double FIND_HILLS_THRE = 0.1;
const double MAGNITUDE_VALUE = 2;  //t
// number of cyclic frame buffer used for motion detection
// (should, probably, depend on FPS)


struct _vehicle_boundaries
{
	int left;
	int right;
	int top;
	int bottom;
};

struct _XML
{
	int lane;
	bool moto;
	bool plate;
	bool radar;
	int frame_start;
	int frame_end;
	double speed;
};


static void  update_mhi(const Mat& img, Mat& dst, int diff_threshold);

static void  Segmentation(const Mat& motion, Mat& dst);

static void  Projection(const Mat& _segment,_vehicle_boundaries &VBoundaries);

static void Find_Inclination(double const FFT);

static void	PHASES(String type);

static void Find_Hills(double const FFT);

static void ProjectionSmooth(void);

static void FindBounderies(void);

void Print_Vertical_Projection(vector<int> ver_proj);

void Get_Boundaries(vector<int> ver_proj,int lane, _vehicle_boundaries &VBoundaries);

void Open_XML_File(string FileName);

struct _XML Read_XML_File(void);

void Show_XML(_XML XML);