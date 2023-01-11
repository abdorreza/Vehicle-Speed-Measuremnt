#include "opencv2/optflow.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <conio.h>
#include <stdio.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <iostream>

#define PIXELS_PER_M 60.0

using namespace cv;
using namespace std;

struct hmatrix {
	float a0, a1, a2;
	float b0, b1, b2;
	float c0, c1, c2;
};


struct point {
	float x;
	float y;
};

struct quadrangle {
	struct point p_top_l;
	struct point p_top_r;
	struct point p_bot_l;
	struct point p_bot_r;
	/* Height of the reference rect in the real world, in meters. */
	float ref_height;
	/* Width of the reference rect in the real world, in meters. */
	float ref_width;
	/* Desired number of pixels per meter in the inverse perspective
	* mapping. */
	float pixels_per_m;
	/* Top position of the reference rect in the inverse perspective
	* mapped image. */
	float ipm_top;
	/* Left position of the reference rect in the inverse perspective
	* mapped image. */
	float ipm_left;
};


// Functions for Calculate Speed

struct KLT {
	int x;
	int y;
	double ipm_x;
	double ipm_y;
	double prv_x;
	double prv_y;
}  *KLT_FeatureList;

typedef struct _speed_i {
	double v_x;
	double v_y;
	double m_x;
	double m_y;
} speed_i;

//static const double lane_coef[3] = { 0.978, 0.930, 0.980 };
static const double lane_coef[3] = { 0.9, 0.9, 0.9 };
static const double ref_dist_meters = 1.0;
static const double ref_dist_pixels = 60;
static const double frame_interval = 0.033166;






void compute_ipm_features(int x_f, int y_f, vector<vector<KLT>> &features, int frame_no, int feature_no, int lane, hmatrix hmat[]);

speed_i compute_velocity_vector(vector<vector<KLT>> &features, int frame_i, int frame_j);

double metric_calc_velocity_ipm(speed_i speed, int lane);



// Functions for Matrix H and Perspective

Mat generate_hmatrix(struct hmatrix *hmat, const struct quadrangle *quad);

void apply_mat_to_image(const Mat in, const char *out, Mat ipm_matrix,string show);

void print_hmatrix(struct hmatrix *mat, int num);