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

#include "KLT.h"


// Functions for Matrix H and Perspective

Mat generate_hmatrix(struct hmatrix *hmat, const struct quadrangle *quad)
{
	Point2f ref_points[4];
	Point2f ipm_points[4];
	Mat ipm_matrix;

	/* Reference points in the input image.
	* We use these to perform an inverse perspective mapping. */

	ref_points[0] = Point2f(quad->p_top_l.x, quad->p_top_l.y);
	ref_points[1] = Point2f(quad->p_top_r.x, quad->p_top_r.y);
	ref_points[2] = Point2f(quad->p_bot_l.x, quad->p_bot_l.y);
	ref_points[3] = Point2f(quad->p_bot_r.x, quad->p_bot_r.y);

	/* Reference points in the inverse perspective mapping. */
	ipm_points[0] = Point2f(quad->ipm_left, quad->ipm_top);
	ipm_points[1] = Point2f(quad->ipm_left + quad->pixels_per_m * quad->ref_width, quad->ipm_top);
	ipm_points[2] = Point2f(quad->ipm_left, quad->ipm_top + quad->pixels_per_m * quad->ref_height);
	ipm_points[3] = Point2f(quad->ipm_left + quad->pixels_per_m * quad->ref_width, quad->ipm_top + quad->pixels_per_m * quad->ref_height);

	ipm_matrix = getPerspectiveTransform(ref_points, ipm_points);

	hmat->a0 = ipm_matrix.at <double>(0, 0);
	hmat->a1 = ipm_matrix.at <double>(0, 1);
	hmat->a2 = ipm_matrix.at <double>(0, 2);
	hmat->b0 = ipm_matrix.at <double>(1, 0);
	hmat->b1 = ipm_matrix.at <double>(1, 1);
	hmat->b2 = ipm_matrix.at <double>(1, 2);
	hmat->c0 = ipm_matrix.at <double>(2, 0);
	hmat->c1 = ipm_matrix.at <double>(2, 1);
	hmat->c2 = ipm_matrix.at <double>(2, 2);

	return ipm_matrix;
}


void apply_mat_to_image(const Mat in, const char *out, Mat ipm_matrix,string show)
{
	Mat src;
	in.copyTo(src);

	Mat dst = src.clone();

	warpPerspective(src, dst, ipm_matrix, dst.size());
	imwrite(out, dst);
	if (show == "SHOW")
	{
		namedWindow("Dst",WINDOW_NORMAL);
		resizeWindow("Dst", dst.size().width / 2, dst.size().height / 2);
		imshow("Dst", dst);
		waitKey(0);
	}
}



void print_hmatrix(struct hmatrix *mat, int num)
{
	cout.setf(ios_base::fixed, ios_base::floatfield);
	cout.precision(8);
	cout << "{" << endl << "\t" <<
		"/* faixa " << num << ": */" << endl << "\t" <<
		mat->a0 << ", " << mat->a1 << ", " << mat->a2 << "," <<
		endl << "\t" <<
		mat->b0 << ", " << mat->b1 << ", " << mat->b2 << "," <<
		endl << "\t" <<
		mat->c0 << ", " << mat->c1 << ", " << mat->c2 <<
		endl << "},";
}



// Functions for Calculate Speed

void compute_ipm_features(int x_f, int y_f, vector<vector<KLT>> &features, int frame_no, int feature_no, int lane, hmatrix hmat[])
{
	float x;
	float y;
	float w;
	int l;
	int i;
	x = x_f;
	y = y_f;
	w = (x * hmat[lane - 1].c0) + (y * hmat[lane - 1].c1) + hmat[lane - 1].c2;
	features[frame_no][feature_no].ipm_x = (x * hmat[lane - 1].a0 + y * hmat[lane - 1].a1 + hmat[lane - 1].a2) / w;
	features[frame_no][feature_no].ipm_y = (x * hmat[lane - 1].b0 + y * hmat[lane - 1].b1 + hmat[lane - 1].b2) / w;
	features[frame_no][feature_no].prv_x = x;
	features[frame_no][feature_no].prv_y = y;
}

speed_i compute_velocity_vector(vector<vector<KLT>> &features, int frame_i, int frame_j)
{

	speed_i speed = { 0.0, 0.0, 0.0, 0.0 };

	int nfeatures = 0;

	for (int i = 0; i < 10; i++) // n features
	{

		double act_x = features[frame_i][i].ipm_x;
		double act_y = features[frame_i][i].ipm_y;
		double prv_x = features[frame_j][i].ipm_x;
		double prv_y = features[frame_j][i].ipm_y;

		speed.v_x += (act_x - prv_x);
		speed.v_y += (act_y - prv_y);
		speed.m_x += act_x;
		speed.m_y += act_y;
		nfeatures++;
	}

	speed.v_x = speed.v_x / (double)nfeatures;
	speed.v_y = speed.v_y / (double)nfeatures;
	speed.m_x = speed.m_x / (double)nfeatures;
	speed.m_y = speed.m_y / (double)nfeatures;

	return speed;
}


double metric_calc_velocity_ipm(const speed_i speed, int lane)
{
	lane -= 1;
	assert(lane < (sizeof(lane_coef) / sizeof(double)));

	/* Displacement in pixels: */
	double dpixel = sqrt((speed.v_x * speed.v_x) + (speed.v_y * speed.v_y));
	
	/* Distance in meters: */
	double d = (ref_dist_meters * dpixel) / ref_dist_pixels;

	//cout << "d = " << d << "\n";
	if (d >= 0.30 && lane == 2) d = d*0.92;
	if (d >= 0.30 && lane == 0) d = d*0.82;

	/* Speed in meters per sec: */
	//return lane_coef[lane] * d / frame_interval;
	return d / frame_interval;
}
