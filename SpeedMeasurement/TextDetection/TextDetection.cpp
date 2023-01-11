#include "opencv2/optflow.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>
#include "TextDetection.h"

#ifndef min
template <class T> static inline T min(T x, T y) { return (x<y) ? x : y; }
#endif
#ifndef max
template <class T> static inline T max(T x, T y) { return (x>y) ? x : y; }
#endif

#ifndef max
# define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
# define min(a, b) ((a) < (b) ? (a) : (b))
#endif

template <class T> static inline void Swap(T& x, T& y) { T t = x; x = y; y = t; }
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst, (void *)src, sizeof(T)*n);
}

/*
static void print_string_stdout(const char *s)
{
	fputs(s, stdout);
	fflush(stdout);
}


static void(*svm_print_string) (const char *) = &print_string_stdout;


#if 1
static void info(const char *fmt, ...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap, fmt);
	vsprintf(buf, fmt, ap);
	va_end(ap);
	(*svm_print_string)(buf);
}
#else
static void info(const char *fmt, ...) {}
#endif
*/

double* thog(unsigned char *image, int nrows, int ncols, struct_thog sthog);

double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

using namespace cv;
using namespace std;
using namespace cv::motempl;


/* FIXME: implement the return of this function as a pointer to a newly
* allocated struct_thog */
struct_thog load_settings(const char *settings)
{
	struct_thog s;
	FILE *file;
	int ret;

	file = fopen(settings, "r");
	if (NULL == file) {
		printf("ERROR: (FIXME) FILE COULD NOT BE OPENED\n");
		return s;
	}

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	ret = fscanf(file, "%d", &s.nh);
	if (!ret)
		goto err;
	ret = fscanf(file, "%d", &s.ncx);
	if (!ret)
		goto err;
	ret = fscanf(file, "%d", &s.ncy);
	if (!ret)
		goto err;
	ret = fscanf(file, "%d", &s.bpc);
	if (!ret)
		goto err;
	ret = fscanf(file, "%d", &s.norm);
	if (!ret)
		goto err;
	ret = fscanf(file, "%s", s.wnorm);
	if (!ret)
		goto err;
	ret = fscanf(file, "%lf", &s.rad);
	if (!ret)
		goto err;
	ret = fscanf(file, "%s", s.grad);
	if (!ret)
		goto err;
	ret = fscanf(file, "%s", s.hmetric);
	if (!ret)
		goto err;
	ret = fscanf(file, "%s", s.weight_function);
	if (!ret)
		goto err;
	ret = fscanf(file, "%d", &s.debug);
	if (!ret)
		goto err;

	if (thog_print_settings) {
		printf("load_settings:\n");
		printf("\ts.nh=%d\n", s.nh);
		printf("\ts.ncx=%d\n", s.ncx);
		printf("\ts.ncy=%d\n", s.ncy);
		printf("\ts.bpc=%d\n", s.bpc);
		printf("\ts.norm=%d\n", s.norm);
		printf("\ts.wnorm=%s\n", s.wnorm);
		printf("\ts.rad=%.6f\n", s.rad);
		printf("\ts.grad=%s\n", s.grad);
		printf("\ts.hmetric=%s\n", s.hmetric);
		printf("\ts.weight_function=%s\n", s.weight_function);
		printf("\ts.debug=%d\n", s.debug);
	}

	s.deformable_weights = FALSE;
	s.noc = s.ncx * s.ncy;
	s.nob = s.noc * s.bpc;
	fclose(file);

	setlocale(LC_ALL, old_locale);
	free(old_locale);
	return s;
err:
	setlocale(LC_ALL, old_locale);
	free(old_locale);
	printf("ERROR: (FIXME) Bad return value from fscanf\n");
	return s;
}



static char* readline(FILE *input)
{
	int len;

	if (fgets(Line, max_line_len, input) == NULL)
		return NULL;

	while (strrchr(Line, '\n') == NULL)
	{
		max_line_len *= 2;
		Line = (char *)realloc(Line, max_line_len);
		len = (int)strlen(Line);
		if (fgets(Line + len, max_line_len - len, input) == NULL)
			break;
	}
	return Line;
}



svm_model *svm_load_model(const char *model_file_name)
{
	int ret;
	FILE *fp = fopen(model_file_name, "rb");
	if (fp == NULL) return NULL;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	// read parameters

	svm_model *model = Malloc(svm_model, 1);
	svm_parameter& param = model->param;
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->sv_indices = NULL;
	model->label = NULL;
	model->nSV = NULL;

	char cmd[81];
	while (1)
	{
		ret = fscanf(fp, "%80s", cmd);
		if (!ret)
			return NULL;

		if (strcmp(cmd, "svm_type") == 0)
		{
			ret = fscanf(fp, "%80s", cmd);
			int i;
			for (i = 0; svm_type_table[i]; i++)
			{
				if (strcmp(svm_type_table[i], cmd) == 0)
				{
					param.svm_type = i;
					break;
				}
			}
			if (svm_type_table[i] == NULL)
			{
				fprintf(stderr, "unknown svm type.\n");

				setlocale(LC_ALL, old_locale);
				free(old_locale);
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		}
		else if (strcmp(cmd, "kernel_type") == 0)
		{
			ret = fscanf(fp, "%80s", cmd);
			int i;
			for (i = 0; kernel_type_table[i]; i++)
			{
				if (strcmp(kernel_type_table[i], cmd) == 0)
				{
					param.kernel_type = i;
					break;
				}
			}
			if (kernel_type_table[i] == NULL)
			{
				fprintf(stderr, "unknown kernel function.\n");

				setlocale(LC_ALL, old_locale);
				free(old_locale);
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		}
		else if (strcmp(cmd, "degree") == 0)
			ret = fscanf(fp, "%d", &param.degree);
		else if (strcmp(cmd, "gamma") == 0)
			ret = fscanf(fp, "%lf", &param.gamma);
		else if (strcmp(cmd, "coef0") == 0)
			ret = fscanf(fp, "%lf", &param.coef0);
		else if (strcmp(cmd, "nr_class") == 0)
			ret = fscanf(fp, "%d", &model->nr_class);
		else if (strcmp(cmd, "total_sv") == 0)
			ret = fscanf(fp, "%d", &model->l);
		else if (strcmp(cmd, "rho") == 0)
		{
			int n = model->nr_class * (model->nr_class - 1) / 2;
			model->rho = Malloc(double, n);
			for (int i = 0; i<n; i++)
				ret = fscanf(fp, "%lf", &model->rho[i]);
		}
		else if (strcmp(cmd, "label") == 0)
		{
			int n = model->nr_class;
			model->label = Malloc(int, n);
			for (int i = 0; i<n; i++)
				ret = fscanf(fp, "%d", &model->label[i]);
		}
		else if (strcmp(cmd, "probA") == 0)
		{
			int n = model->nr_class * (model->nr_class - 1) / 2;
			model->probA = Malloc(double, n);
			for (int i = 0; i<n; i++)
				ret = fscanf(fp, "%lf", &model->probA[i]);
		}
		else if (strcmp(cmd, "probB") == 0)
		{
			int n = model->nr_class * (model->nr_class - 1) / 2;
			model->probB = Malloc(double, n);
			for (int i = 0; i<n; i++)
				ret = fscanf(fp, "%lf", &model->probB[i]);
		}
		else if (strcmp(cmd, "nr_sv") == 0)
		{
			int n = model->nr_class;
			model->nSV = Malloc(int, n);
			for (int i = 0; i<n; i++)
				ret = fscanf(fp, "%d", &model->nSV[i]);
		}
		else if (strcmp(cmd, "SV") == 0)
		{
			while (1)
			{
				int c = getc(fp);
				if (c == EOF || c == '\n') break;
			}
			break;
		}
		else
		{
			fprintf(stderr, "unknown text in model file: [%s]\n", cmd);

			setlocale(LC_ALL, old_locale);
			free(old_locale);
			free(model->rho);
			free(model->label);
			free(model->nSV);
			free(model);
			return NULL;
		}
		if (!ret)
			return NULL;
	}

	// read sv_coef and SV

	int elements = 0;
	long pos = ftell(fp);

	max_line_len = 1024;
	Line = Malloc(char, max_line_len);
	char *p, *endptr, *idx, *val;

	while (readline(fp) != NULL)
	{
		p = strtok(Line, ":");
		while (1)
		{
			p = strtok(NULL, ":");
			if (p == NULL)
				break;
			++elements;
		}
	}
	elements += model->l;

	fseek(fp, pos, SEEK_SET);

	int m = model->nr_class - 1;
	int l = model->l;
	model->sv_coef = Malloc(double *, m);
	int i;
	for (i = 0; i<m; i++)
		model->sv_coef[i] = Malloc(double, l);
	model->SV = Malloc(svm_node*, l);
	svm_node *x_space = NULL;
	if (l>0) x_space = Malloc(svm_node, elements);

	int j = 0;
	for (i = 0; i<l; i++)
	{
		readline(fp);
		model->SV[i] = &x_space[j];

		p = strtok(Line, " \t");
		model->sv_coef[0][i] = strtod(p, &endptr);
		for (int k = 1; k<m; k++)
		{
			p = strtok(NULL, " \t");
			model->sv_coef[k][i] = strtod(p, &endptr);
		}

		while (1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL)
				break;
			x_space[j].index = (int)strtol(idx, &endptr, 10);
			x_space[j].value = strtod(val, &endptr);

			++j;
		}
		x_space[j++].index = -1;
	}
	free(Line);

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0)
		return NULL;

	model->free_sv = 1;	// XXX
	return model;
}


//////////////////////////////////////////
//////////////////////////////////////////
//////////////////////////////////////////
double classify(const unsigned char *image, int nrows, int ncols, int x, int y, int w, int h, struct svm_model* model, double *prob, struct_thog sthog)
{

	int margin = (int)(0.25*h + 0.5);
	margin = 0;

	if ((y - margin) > 0 && (h + margin) < nrows) 
	{
		y = y - margin;
		h = h + 2 * margin;
	}

	unsigned char *sample = (unsigned char *)malloc((w*h) * sizeof(unsigned char));

	/*Cropping the candidate text region from the original image: */
	int i, j, k, l;
	for (j = y, k = 0; j < (y + h); j++, k++) {
		for (i = x, l = 0; i < (x + w); i++, l++) {
			sample[w * k + l] = image[ncols * j + i];
		}
	}

	double *descriptor = thog(sample, h, w, sthog);

	/*Converting the descriptor to the SVM model: */
	svm_node* s = (struct svm_node *)malloc((sthog.nob + 1) * sizeof(struct svm_node));

	for (i = 0; i < sthog.nob; i++) {
		s[i].index = i + 1;
		s[i].value = descriptor[i];
	}
	s[i].index = -1;
	s[i].value = -1;

	double predict_label = svm_predict_probability(model, s, prob);
	free(s);
	free(descriptor);
	free(sample);

	return predict_label;
}


double** alloc_dmatrix(int ncols, int nrows) {
	int i;
	double **matrix = (double **)malloc(nrows * sizeof(double *));
	for (i = 0; i < nrows; i++) {
		matrix[i] = (double *)malloc(ncols * sizeof(double));
	}
	return matrix;
}


void disalloc_dmatrix(double **matrix, int nrows)
{
	int i;
	for (i = 0; i < nrows; i++)
	{
		free(matrix[i]);
	}
	free(matrix);
}

void CalTempContrib(int start, int stop, double *tmpContrib, double *contrib) {
	double weight = 0;
	int i = 0;
	for (i = start; i <= stop; i++) {
		weight += contrib[i];
	}
	for (i = start; i <= stop; i++) {
		tmpContrib[i] = contrib[i] / weight;
	}
}

int Clip(int x) {
	if (x < 0) return 0;
	if (x > 255) return 255;
	return x;
}

int HorizontalFilter(unsigned char *bufImg, int width, int startX, int stopX, int start, int stop, int y, double *pContrib) {
	int valueRGB = 0;
	int i, j;
	for (i = startX, j = start; i <= stopX; i++, j++) {
		valueRGB += bufImg[width * y + i] * pContrib[j];
	}
	return Clip(valueRGB);//ComRGB(Clip((int) valueRed), Clip((int) valueGreen),Clip((int) valueBlue));
}

int VerticalFilter(unsigned char *pbInImage, int width, int startY, int stopY, int start, int stop, int x, double *pContrib) {
	int valueRGB = 0;
	int i, j;
	for (i = startY, j = start; i <= stopY; i++, j++) {
		valueRGB += pbInImage[width * i + x] * pContrib[j];
	}
	return Clip(valueRGB); //ComRGB(Clip((int) valueRed), Clip((int) valueGreen),Clip((int) valueBlue));
						   //return valueRGB;
}


unsigned char *HorizontalFiltering(unsigned char *bufImage, int dwInW, int dwInH, int iOutW, int nDots, int nHalfDots, double *contrib, double *tmpContrib, double *normContrib) {
	//int dwInW = bufImage.getWidth();
	//int dwInH = bufImage.getHeight();
	//BufferedImage pbOut = new BufferedImage(iOutW, dwInH, BufferedImage.TYPE_INT_RGB);
	int value = 0;
	unsigned char *pbOut = (unsigned char *)malloc((iOutW * dwInH) * sizeof(unsigned char));
	int x;
	for (x = 0; x < iOutW; x++) {
		int startX;
		int start;
		int X = (int)(((double)x) * ((double)dwInW) / ((double)iOutW) + 0.5);
		int y = 0;
		startX = X - nHalfDots;
		if (startX < 0) {
			startX = 0;
			start = nHalfDots - X;
		}
		else {
			start = 0;
		}

		int stop;
		int stopX = X + nHalfDots;
		if (stopX >(dwInW - 1)) {
			stopX = dwInW - 1;
			stop = nHalfDots + (dwInW - 1 - X);
		}
		else {
			stop = nHalfDots * 2;
		}

		if (start > 0 || stop < nDots - 1) {
			CalTempContrib(start, stop, tmpContrib, contrib);
			for (y = 0; y < dwInH; y++) {
				value = HorizontalFilter(bufImage, dwInW, startX, stopX, start, stop, y, tmpContrib);
				//pbOut.setRGB(x, y, value);
				pbOut[y *iOutW + x] = value;
			}
		}
		else {
			for (y = 0; y < dwInH; y++) {
				value = HorizontalFilter(bufImage, dwInW, startX, stopX, start, stop, y, normContrib);
				//pbOut.setRGB(x, y, value);
				pbOut[y *iOutW + x] = value;
			}
		}
	}
	return pbOut;
}

unsigned char *VerticalFiltering(unsigned char *pbImage, int iW, int iH, int iOutH, int nDots, int nHalfDots, double *contrib, double *tmpContrib, double *normContrib) {

	//int iW = pbImage.getWidth();
	//int iH = pbImage.getHeight();
	//BufferedImage pbOut = new BufferedImage(iW, iOutH,BufferedImage.TYPE_INT_RGB);
	int value = 0;
	unsigned char *pbOut = (unsigned char *)malloc((iW * iOutH) * sizeof(unsigned char));

	int y;
	for (y = 0; y < iOutH; y++) {
		int startY;
		int start;
		int Y = (int)(((double)y) * ((double)iH) / ((double)iOutH) + 0.5);
		startY = Y - nHalfDots;
		if (startY < 0) {
			startY = 0;
			start = nHalfDots - Y;
		}
		else {
			start = 0;
		}
		int stop;
		int stopY = Y + nHalfDots;
		if (stopY >(int) (iH - 1)) {
			stopY = iH - 1;
			stop = nHalfDots + (iH - 1 - Y);
		}
		else {
			stop = nHalfDots * 2;
		}
		if (start > 0 || stop < nDots - 1) {
			CalTempContrib(start, stop, tmpContrib, contrib);
			int x;
			for (x = 0; x < iW; x++) {
				value = VerticalFilter(pbImage, iW, startY, stopY, start, stop, x, tmpContrib);
				//pbOut.setRGB(x, y, value);
				pbOut[y * iW + x] = value;
			}
		}
		else {
			int x;
			for (x = 0; x < iW; x++) {
				value = VerticalFilter(pbImage, iW, startY, stopY, start, stop, x, normContrib);
				//pbOut.setRGB(x, y, value);
				pbOut[y * iW + x] = value;
			}
		}
	}
	return pbOut;
}

unsigned char *resize_gray_uchar_image_bilinear(unsigned char *pixels, int w, int h, int w2, int h2) {

	unsigned char *temp = (unsigned char *)malloc(w2*h2 * sizeof(unsigned char));
	int A, B, C, D, x, y, index, gray;
	float x_ratio = ((float)(w - 1)) / w2;
	float y_ratio = ((float)(h - 1)) / h2;
	float x_diff, y_diff;
	int offset = 0;
	int i, j;
	for (i = 0; i<h2; i++) {
		for (j = 0; j<w2; j++) {
			x = (int)(x_ratio * j);
			y = (int)(y_ratio * i);
			x_diff = (x_ratio * j) - x;
			y_diff = (y_ratio * i) - y;
			index = y*w + x;

			// range is 0 to 255 thus bitwise AND with 0xff
			A = pixels[index] & 0xff;
			B = pixels[index + 1] & 0xff;
			C = pixels[index + w] & 0xff;
			D = pixels[index + w + 1] & 0xff;

			// Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + Dwh
			gray = (int)(
				A*(1 - x_diff)*(1 - y_diff) + B*(x_diff)*(1 - y_diff) +
				C*(y_diff)*(1 - x_diff) + D*(x_diff*y_diff)
				);

			temp[offset++] = gray;
		}
	}
	return temp;
}

double lanczos(int i, int inWidth, int outWidth, double support) {
	double x = (double)i * (double)outWidth / (double)inWidth;
	return sin(x * M_PI) / (x * M_PI) * sin(x * M_PI / support) / (x * M_PI / support);
}



unsigned char *applylanczos(unsigned char *srcBi, int width, int height, int h, int w_mod, int *W) {

	double support = (double) 3.0;
	double scaleV = (double)(h) / (double)(height);
	int w = (int)(floor(width*scaleV / w_mod + 0.5))*w_mod;
	*W = w;
	assert((w % w_mod) == 0);
	double scaleH = (double)(w) / (double)(width);

	if (scaleH >= 1.0 && scaleV >= 1.0) {
		return resize_gray_uchar_image_bilinear(srcBi, width, height, w, h);
	}

	int nHalfDots = (int)((double)width * support / (double)w);
	int nDots = nHalfDots * 2 + 1;
	double *contrib = (double *)malloc(nDots * sizeof(double));
	double *normContrib = (double *)malloc(nDots * sizeof(double));
	double *tmpContrib = (double *)malloc(nDots * sizeof(double));
	int center = nHalfDots;

	if (center < 0) {
		return resize_gray_uchar_image_bilinear(srcBi, width, height, w, h);
	}

	contrib[center] = 1.0;

	double weight = 0.0;
	int i = 0;
	for (i = 1; i <= center; i++) {
		contrib[center + i] = lanczos(i, width, w, support);
		weight += contrib[center + i];
	}

	for (i = center - 1; i >= 0; i--) {
		contrib[i] = contrib[center * 2 - i];
	}

	weight = weight * 2 + 1.0;

	for (i = 0; i <= center; i++) {
		normContrib[i] = contrib[i] / weight;
	}

	for (i = center + 1; i < nDots; i++) {
		normContrib[i] = normContrib[center * 2 - i];
	}

	unsigned char *pbOut = HorizontalFiltering(srcBi, width, height, w, nDots, nHalfDots, contrib, tmpContrib, normContrib);

	unsigned char *pbFinalOut = VerticalFiltering(pbOut, w, height, h, nDots, nHalfDots, contrib, tmpContrib, normContrib);

	free(pbOut);
	free(contrib);
	free(normContrib);
	free(tmpContrib);

	return pbFinalOut;
}

int choose_gaussian_weight_size(double dev) {
	double rwt = 2 * dev;
	return (int)(ceil(rwt));
}

void compute_gaussian_weights(double *weight, int rwt) {
	double eps = 0.01;
	double a = -log(eps);
	int nwt = 2 * rwt + 1;

	int i;
	for (i = 0; i < nwt; i++) {
		double z = (i - rwt) / ((double)rwt);
		weight[i] = (1 - z*z)*exp(-a*z*z);
	}
}

void compute_binomial_weights(double *weight, int rwt) {
	int nwt = 2 * rwt + 1;
	weight[0] = 1;
	int i, j;
	for (i = 1; i < nwt; i++) {
		weight[i] = 0.0;
		for (j = i; j >= 1; j--) {
			weight[j] = (weight[j] + weight[j - 1]) / 2;
		}
		weight[0] /= 2;
	}
}

double get_grey_avg(double *grey, int w, int h, int x, int y, double *x_weight, int x_rwt, double *y_weight, int y_rwt) {
	double sum_vwt = 0.0, sum_wt = 0.0;
	int dx, dy;
	for (dy = -y_rwt; dy <= y_rwt; dy++) {
		int y1 = y + dy;
		if ((y1 >= 0) && (y1 < h)) {
			for (dx = -x_rwt; dx <= x_rwt; dx++) {
				int x1 = x + dx;
				if ((x1 >= 0) && (x1 < w)) {
					int position = y1 * w + x1;
					double v = grey[position];
					double wt = x_weight[x_rwt + dx] * y_weight[y_rwt + dy];
					sum_vwt += v * wt;
					sum_wt += wt;
				}
			}
		}
	}
	return sum_vwt / sum_wt;
}

/*Get the deviation of a pixel given a normalizing window: */
double get_grey_dev(double *grey, int w, int h, int x, int y, double *x_weight, int x_rwt, double *y_weight, int y_rwt, double AVG, double noise) {
	double sum_v2wt = 0.0, sum_wt = 0.0;
	int dx, dy;
	for (dy = -y_rwt; dy <= y_rwt; dy++) {
		int y1 = y + dy;
		if ((y1 >= 0) && (y1 < h)) {
			for (dx = -x_rwt; dx <= x_rwt; dx++) {
				int x1 = x + dx;
				if ((x1 >= 0) && (x1 < w)) {
					int position = y1 * w + x1;
					double v = grey[position] - AVG;
					double wt = x_weight[x_rwt + dx] * y_weight[y_rwt + dy];
					sum_v2wt += v * v * wt;
					sum_wt += wt;
				}
			}
		}
	}
	return sqrt(sum_v2wt / sum_wt + noise*noise);
}


double *normalize_grey_image(double *grey, int w, int h, double *x_weight, int x_rwt, double *y_weight, int y_rwt, double noise) {

	double* grel = (double *)malloc(w * h * sizeof(double));

	double AVG, DEV;

	int x, y;
	for (y = 0; y < h; y++) {
		for (x = 0; x < w; x++) {
			int position = y * w + x;
			AVG = get_grey_avg(grey, w, h, x, y, x_weight, x_rwt, y_weight, y_rwt);
			DEV = get_grey_dev(grey, w, h, x, y, x_weight, x_rwt, y_weight, y_rwt, AVG, noise);
			grel[position] = (grey[position] - AVG) / (3 * DEV) + 0.5;
			if (grel[position] < 0) { grel[position] = 0.0; }
			else if (grel[position] > 1) { grel[position] = 1.0; }
		}
	}
	return grel;
}


void convert_to_log_scale(double *grey, int w, int h, double eps) {
	int position;
	int n = w * h;
	for (position = 0; position < n; position++) {
		grey[position] = (log(grey[position] + eps) - log(eps)) / (log(1 + eps) - log(eps));
	}
}

double StepFunc(int n, int k, double z) {
	assert((k >= 0) && (k < n));
	if (z <= 0) { return (k == 0 ? 1 : 0); }
	else if (z >= 1) { return (k == (n - 1) ? 1 : 0); }
	else {
		return ((k <= z*n) && (z*n < k + 1) ? 1 : 0);
	}
}

double BernsteinPoly(int n, int k, double z) {
	assert((k >= 0) && (k <= n));
	double res = 1.0;
	int i;
	for (i = 0; i < k; i++) {
		res = (res * (n - i)) / (i + 1)*z;
	}
	return res*pow(1 - z, n - k);
}


/*Computes the Bernstein polynomial of degree {n} and index {k} for the argument {z}.*/
double Bernstein(int n, int k, double z) {
	assert((k >= 0) && (k <= n));
	if (z <= 0) { return (k == 0 ? 1 : 0); }
	else if (z >= 1) { return (k == n ? 1 : 0); }
	else {
		double zmax = ((double)k) / ((double)n);
		return BernsteinPoly(n, k, z) / BernsteinPoly(n, k, zmax);
	}
}

/**/

/*An edge-core weight function. If {n == 1} returns 1, if (n == 2) returns
*weight 1.0 near the edges, or 1.0 in the core region depending on {k}*/
double EdgeCore(int n, int k, double z) {
	assert(n == 2);
	assert((k >= 0) && (k < n));
	if ((z <= 0) || (z >= 1)) { return (k == 0 ? 1 : 0); }
	else {
		double v = 4 * z * (1 - z);
		v = v*v;
		return (k == 0 ? 1 - v : v);
	}
}

double gaussian(double z, double mu, double sigma) {
	return exp(-((z - mu)*(z - mu)) / (2 * sigma * sigma));
}


/*An exponential weight function: */
double Exp(int n, int k, double z) {

	double mu = 0.01;
	double sigma = 0.5;
	assert((k >= 0) && (k < n));
	double avg = -mu + (1 + 2 * mu) / (double)(n - 1)*k;
	double dev = sigma / n;
	return gaussian(z, avg, dev);
}


double cell_weight(char *weight_function, int ncz, int cz, int z, double zmax, double zmin) {

	if (ncz == 1) { return 1; }

	double z_star = (z - zmin) / (zmax - zmin);

	if (strcmp(weight_function, "Step") == 0) {
		return StepFunc(ncz, cz, z_star);
	}
	else if (strcmp(weight_function, "Bernstein") == 0) {
		return Bernstein(ncz - 1, cz, z_star);
	}
	else if (strcmp(weight_function, "Core") == 0) {
		return EdgeCore(ncz, cz, z_star);
	}
	else if (strcmp(weight_function, "Exp") == 0) {
		return Exp(ncz, cz, z_star);
	}
	else {
		assert(0);
		exit(1);
		return 0;
	}
}


void gradient_simple(double *image, int width, int height, int x, int y, double *grad) {

	int position_x = y * width + x;
	int position_y = y * width + x;

	if (x == 0) {
		position_x = y * width + (x + 1);
	}
	if (x == (width - 1)) {
		position_x = y * width + (x - 1);
	}

	if (y == 0) {
		position_y = (y + 1) * width + x;
	}

	if (y == (height - 1)) {
		position_y = (y - 1) * width + x;
	}

	double kxm = image[position_x - 1];
	double kxp = image[position_x + 1];
	double kym = image[position_y - width];
	double kyp = image[position_y + width];

	grad[0] = (kxp - kxm) / 2;
	grad[1] = (kyp - kym) / 2;
}

void get_bin_pos(int bins_per_cell, double dtheta, int *bin, double *factor) {

	/*Computing the bin according the gradient direction.*/
	int full_circ = 0;

	double period = (full_circ ? 2 * M_PI : M_PI);

	double a = bins_per_cell*(dtheta / period + 1);
	int ia = (int)floor(a);
	double frac = a - (double)ia;

	bin[0] = (ia + bins_per_cell) % bins_per_cell;
	bin[1] = (ia + 1) % bins_per_cell;
	factor[0] = (1 - frac);
	factor[1] = frac;

	assert((bin[0] >= 0) && (bin[0] < bins_per_cell));
	assert((bin[1] >= 0) && (bin[1] < bins_per_cell));
}


double* thog(unsigned char *image, int nrows, int ncols, struct_thog sthog)
{
	/*T-HOG settings: */
	int new_height = sthog.nh;
	int number_of_cells_x = sthog.ncx;
	int number_of_cells_y = sthog.ncy;
	int bins_per_cell = sthog.bpc;
	int image_normalization = sthog.norm;
	char *image_normalization_weight = sthog.wnorm;
	double image_normalization_weight_radius = sthog.rad;
	char *gradient_option = sthog.grad;
	char *histogram_normalization_metric = sthog.hmetric;
	char *weight_function = sthog.weight_function;
	int deformable_weights = sthog.deformable_weights;
	int debug = sthog.debug;
	/*End*/

	int i;

	int safe_margin = 1;

	double noise = 0.03;

	double black_level = 0.02; /*assumed black level of image*/

	int image_logscale = FALSE;

	int number_of_cells = number_of_cells_x * number_of_cells_y;

	int number_of_bins = number_of_cells * bins_per_cell; /*HOG bins*/

														  /*Image resizing: if the new height is negative the region is not resized.*/
	unsigned char *resized = NULL;

	/*Getting the resized dimensions: */
	int rwidth;

	int rheight = new_height;

	if (new_height > 0) {
		resized = applylanczos(image, ncols, nrows, new_height, 1, &rwidth);
	}
	else { printf("error: wrong image dimension\n"); exit(1); }

	/*number of pixels of the resized image.*/
	int n = rwidth * rheight;

	//double *dnorm = (double *)malloc(n * sizeof(double));

	//double *dtheta = (double *)malloc(n * sizeof(double));

	double *grey = (double *)malloc(n * sizeof(double));

	for (i = 0; i < n; i++) { grey[i] = (double)(resized[i]); }

	/*Creating a matrix to hold the cells histogram.*/
	double *cells_histogram = (double *)malloc(number_of_bins * sizeof(double));
	for (i = 0; i < number_of_bins; i++) {
		cells_histogram[i] = 0;
	}
	if ((rwidth <= 2) || (rheight <= 2)) {
		printf("too small dimension\n");
		return cells_histogram;
	}

	/*Computing weights {x_weight, y_weight} to normalize the resized image.*/
	int x_weight_rad = -1, y_weight_rad = -1;

	double *x_weight = NULL, *y_weight = NULL;

	if (strcmp(image_normalization_weight, "Gauss") == 0) {
		/*Gaussian weights*/
		x_weight_rad = choose_gaussian_weight_size(image_normalization_weight_radius * rheight);
		y_weight_rad = choose_gaussian_weight_size(1.0 * rheight / 3.0);

		x_weight = (double *)malloc((2 * x_weight_rad + 1) * sizeof(double));
		y_weight = (double *)malloc((2 * y_weight_rad + 1) * sizeof(double));

		compute_gaussian_weights(x_weight, x_weight_rad);
		compute_gaussian_weights(y_weight, y_weight_rad);
	}
	else if (strcmp(image_normalization_weight, "Binomial") == 0) {
		/*Binomial weights*/
		x_weight_rad = y_weight_rad = (int)image_normalization_weight_radius;

		x_weight = (double *)malloc((2 * x_weight_rad + 1) * sizeof(double));
		y_weight = (double *)malloc((2 * y_weight_rad + 1) * sizeof(double));

		compute_binomial_weights(x_weight, x_weight_rad);
		compute_binomial_weights(y_weight, y_weight_rad);
	}
	else {
		printf("error: choose a valid weight image normalization\n");
		exit(1);
	}

	/*Convert to grey scale*/

	if (debug) {
		/*Writing image norm*/
		//image_functions.write_pgm (grey, rwidth, rheight, 0.0, 1.0, "grey");
	}

	/*Image normalization*/
	double *grel = NULL;
	if (image_normalization) {
		if ((x_weight != NULL) && (y_weight != NULL)) {
			grel = normalize_grey_image(grey, rwidth, rheight, x_weight, x_weight_rad, y_weight, y_weight_rad, noise);
		}
		else {
			exit(1);
		}
	}
	else {
		for (i = 0; i < n; i++) { grel[i] = grey[i]; }
	}

	/*convert to log scale*/
	if (image_logscale) {
		convert_to_log_scale(grel, rwidth, rheight, black_level);
	}

	//get_mag_theta (grel, dnorm, dtheta, rwidth, rheight, safe_margin, noise, gradient_option, debug);

	int x, y;

	/*Getting the baselines: */
	double xleft = safe_margin;
	double xright = rwidth - safe_margin;
	double ybot = rheight - safe_margin - 1;
	double ytop = safe_margin;

	double **mwtx = alloc_dmatrix(number_of_cells_x, rwidth);
	for (x = safe_margin; x < (rwidth - safe_margin); x++) {
		int cx;
		for (cx = 0; cx < number_of_cells_x; cx++) {
			mwtx[x][cx] = cell_weight(weight_function, number_of_cells_x, cx, x, xright, xleft);
		}
	}

	double **mwty = alloc_dmatrix(number_of_cells_y, rheight);
	for (y = safe_margin; y < (rheight - safe_margin); y++) {
		int cy;
		for (cy = 0; cy < number_of_cells_y; cy++) {
			mwty[y][cy] = cell_weight(weight_function, number_of_cells_y, cy, y, ybot, ytop);
		}
	}

	double eps = sqrt(2.0)*noise; /*Assumed deviation of noise in gradient*/

	double eps2 = eps * eps;

	int* vbin = (int *)malloc(2 * sizeof(int));

	double* factor = (double *)malloc(2 * sizeof(double));

	double *grad = (double *)malloc(2 * sizeof(double));;

	for (x = safe_margin; x < (rwidth - safe_margin); x++) {

		for (y = safe_margin; y < (rheight - safe_margin); y++) {

			int position = y * rwidth + x;

			/*Computing image gradients*/
			gradient_simple(grel, rwidth, rheight, x, y, grad);

			/*Computing the gradient norm but return zero if too small*/
			double d2 = grad[0] * grad[0] + grad[1] * grad[1];

			double dnorm = 0.0;
			if (d2 > eps2) {
				dnorm = sqrt(d2 - eps2);
			}

			/*Computing the gradient direction.*/
			double dtheta = atan2(grad[1], grad[0]);

			if (dtheta < 0) {
				//dtheta[position] += Math.PI;
				dtheta += 2 * M_PI;
			}

			get_bin_pos(bins_per_cell, dtheta, vbin, factor);

			/*Computing the cells histogram and fuzzy weights: */
			int cx, cy;

			for (cx = 0; cx < number_of_cells_x; cx++) {

				for (cy = 0; cy < number_of_cells_y; cy++) {

					int c_pos = cy * number_of_cells_x + cx;

					int bin_pos1 = c_pos * bins_per_cell + vbin[0];

					int bin_pos2 = c_pos * bins_per_cell + vbin[1];

					cells_histogram[bin_pos1] += (dnorm * mwtx[x][cx] * mwty[y][cy]) * factor[0];

					cells_histogram[bin_pos2] += (dnorm * mwtx[x][cx] * mwty[y][cy]) * factor[1];
				}
			}
		}
	}

	free(grad);
	free(vbin);
	free(factor);

	/*Normalize the histogram of each cell to unit L1 or L2 norm: */
	double sum = 0.0;

	int bin;

	/*Normalization sum: */
	for (bin = 0; bin < number_of_bins; bin++) {
		if (L1) { sum += cells_histogram[bin]; }
		else { sum += cells_histogram[bin] * cells_histogram[bin]; }
	}

	double cell_norm = 0.0;
	if (L1) { cell_norm = sum + 1.0 * number_of_bins; }
	else { cell_norm = sqrt(sum + 1.0 * number_of_bins); }

	/*Descriptor normalization: */
	for (bin = 0; bin < number_of_bins; bin++) {
		cells_histogram[bin] = (float)(cells_histogram[bin] / cell_norm);
	}

	disalloc_dmatrix(mwtx, rwidth);
	disalloc_dmatrix(mwty, rheight);
	free(x_weight);
	free(y_weight);
	free(grel);
	//free(dnorm);
	//free(dtheta);
	free(grey);
	free(resized);

	return cells_histogram;
}


/******************************************************************************/
/*********************************** SVM **************************************/
/******************************************************************************/

static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for (int t = times; t>0; t /= 2)
	{
		if (t % 2 == 1) ret *= tmp;
		tmp = tmp * tmp;
	}
	return ret;
}



class QMatrix {
public:
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual ~QMatrix() {}
};

class Kernel : public QMatrix {
public:
	Kernel(int l, svm_node * const * x, const svm_parameter& param);
	virtual ~Kernel();

	static double k_function(const svm_node *x, const svm_node *y,
		const svm_parameter& param);
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const	// no so const...
	{
		Swap(x[i], x[j]);
		if (x_square) Swap(x_square[i], x_square[j]);
	}
protected:

	double (Kernel::*kernel_function)(int i, int j) const;

private:
	const svm_node **x;
	double *x_square;

	// svm_parameter
	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

	static double dot(const svm_node *px, const svm_node *py);
	double kernel_linear(int i, int j) const
	{
		return dot(x[i], x[j]);
	}
	double kernel_poly(int i, int j) const
	{
		return powi(gamma*dot(x[i], x[j]) + coef0, degree);
	}
	double kernel_rbf(int i, int j) const
	{
		return exp(-gamma*(x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
	}
	double kernel_sigmoid(int i, int j) const
	{
		return tanh(gamma*dot(x[i], x[j]) + coef0);
	}
	double kernel_precomputed(int i, int j) const
	{
		return x[i][(int)(x[j][0].value)].value;
	}
};

Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& param)
	:kernel_type(param.kernel_type), degree(param.degree),
	gamma(param.gamma), coef0(param.coef0)
{
	switch (kernel_type)
	{
	case LINEAR:
		kernel_function = &Kernel::kernel_linear;
		break;
	case POLY:
		kernel_function = &Kernel::kernel_poly;
		break;
	case RBF:
		kernel_function = &Kernel::kernel_rbf;
		break;
	case SIGMOID:
		kernel_function = &Kernel::kernel_sigmoid;
		break;
	case PRECOMPUTED:
		kernel_function = &Kernel::kernel_precomputed;
		break;
	}

	clone(x, x_, l);

	if (kernel_type == RBF)
	{
		x_square = new double[l];
		for (int i = 0; i<l; i++)
			x_square[i] = dot(x[i], x[i]);
	}
	else
		x_square = 0;
}

Kernel::~Kernel()
{
	delete[] x;
	delete[] x_square;
}

double Kernel::dot(const svm_node *px, const svm_node *py)
{
	double sum = 0;
	while (px->index != -1 && py->index != -1)
	{
		if (px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if (px->index > py->index)
				++py;
			else
				++px;
		}
	}
	return sum;
}


double Kernel::k_function(const svm_node *x, const svm_node *y,
	const svm_parameter& param)
{
	switch (param.kernel_type)
	{
	case LINEAR:
		return dot(x, y);
	case POLY:
		return powi(param.gamma*dot(x, y) + param.coef0, param.degree);
	case RBF:
	{
		double sum = 0;
		while (x->index != -1 && y->index != -1)
		{
			if (x->index == y->index)
			{
				double d = x->value - y->value;
				sum += d*d;
				++x;
				++y;
			}
			else
			{
				if (x->index > y->index)
				{
					sum += y->value * y->value;
					++y;
				}
				else
				{
					sum += x->value * x->value;
					++x;
				}
			}
		}

		while (x->index != -1)
		{
			sum += x->value * x->value;
			++x;
		}

		while (y->index != -1)
		{
			sum += y->value * y->value;
			++y;
		}

		return exp(-param.gamma*sum);
	}
	case SIGMOID:
		return tanh(param.gamma*dot(x, y) + param.coef0);
	case PRECOMPUTED:  //x: test (validation), y: SV
		return x[(int)(y->value)].value;
	default:
		return 0;  // Unreachable 
	}
}

double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
{
	int i;
	if (model->param.svm_type == ONE_CLASS ||
		model->param.svm_type == EPSILON_SVR ||
		model->param.svm_type == NU_SVR)
	{
		double *sv_coef = model->sv_coef[0];
		double sum = 0;
		for (i = 0; i<model->l; i++)
			sum += sv_coef[i] * Kernel::k_function(x, model->SV[i], model->param);
		sum -= model->rho[0];
		*dec_values = sum;

		if (model->param.svm_type == ONE_CLASS)
			return (sum>0) ? 1 : -1;
		else
			return sum;
	}
	else
	{
		int nr_class = model->nr_class;
		int l = model->l;

		double *kvalue = Malloc(double, l);
		for (i = 0; i<l; i++)
			kvalue[i] = Kernel::k_function(x, model->SV[i], model->param);

		int *start = Malloc(int, nr_class);
		start[0] = 0;
		for (i = 1; i<nr_class; i++)
			start[i] = start[i - 1] + model->nSV[i - 1];

		int *vote = Malloc(int, nr_class);
		for (i = 0; i<nr_class; i++)
			vote[i] = 0;

		int p = 0;
		for (i = 0; i<nr_class; i++)
			for (int j = i + 1; j<nr_class; j++)
			{
				double sum = 0;
				int si = start[i];
				int sj = start[j];
				int ci = model->nSV[i];
				int cj = model->nSV[j];

				int k;
				double *coef1 = model->sv_coef[j - 1];
				double *coef2 = model->sv_coef[i];
				for (k = 0; k<ci; k++)
					sum += coef1[si + k] * kvalue[si + k];
				for (k = 0; k<cj; k++)
					sum += coef2[sj + k] * kvalue[sj + k];
				sum -= model->rho[p];
				dec_values[p] = sum;

				if (dec_values[p] > 0)
					++vote[i];
				else
					++vote[j];
				p++;
			}

		int vote_max_idx = 0;
		for (i = 1; i<nr_class; i++)
			if (vote[i] > vote[vote_max_idx])
				vote_max_idx = i;

		free(kvalue);
		free(start);
		free(vote);
		return model->label[vote_max_idx];
	}
}

static double sigmoid_predict(double decision_value, double A, double B)
{
	double fApB = decision_value*A + B;
	// 1-p used later; avoid catastrophic cancellation
	if (fApB >= 0)
		return exp(-fApB) / (1.0 + exp(-fApB));
	else
		return 1.0 / (1 + exp(fApB));
}


// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
static void multiclass_probability(int k, double **r, double *p)
{
	int t, j;
	int iter = 0, max_iter = max(100, k);
	double **Q = Malloc(double *, k);
	double *Qp = Malloc(double, k);
	double pQp, eps = 0.005 / k;

	for (t = 0; t<k; t++)
	{
		p[t] = 1.0 / k;  // Valid if k = 1
		Q[t] = Malloc(double, k);
		Q[t][t] = 0;
		for (j = 0; j<t; j++)
		{
			Q[t][t] += r[j][t] * r[j][t];
			Q[t][j] = Q[j][t];
		}
		for (j = t + 1; j<k; j++)
		{
			Q[t][t] += r[j][t] * r[j][t];
			Q[t][j] = -r[j][t] * r[t][j];
		}
	}
	for (iter = 0; iter<max_iter; iter++)
	{
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp = 0;
		for (t = 0; t<k; t++)
		{
			Qp[t] = 0;
			for (j = 0; j<k; j++)
				Qp[t] += Q[t][j] * p[j];
			pQp += p[t] * Qp[t];
		}
		double max_error = 0;
		for (t = 0; t<k; t++)
		{
			double error = fabs(Qp[t] - pQp);
			if (error>max_error)
				max_error = error;
		}
		if (max_error<eps) break;

		for (t = 0; t<k; t++)
		{
			double diff = (-Qp[t] + pQp) / Q[t][t];
			p[t] += diff;
			pQp = (pQp + diff*(diff*Q[t][t] + 2 * Qp[t])) / (1 + diff) / (1 + diff);
			for (j = 0; j<k; j++)
			{
				Qp[j] = (Qp[j] + diff*Q[t][j]) / (1 + diff);
				p[j] /= (1 + diff);
			}
		}
	}
	//if (iter >= max_iter)
		//info("Exceeds max_iter in multiclass_prob\n");
	for (t = 0; t<k; t++) free(Q[t]);
	free(Q);
	free(Qp);
}

double svm_predict(const svm_model *model, const svm_node *x)
{
	int nr_class = model->nr_class;
	double *dec_values;
	if (model->param.svm_type == ONE_CLASS ||
		model->param.svm_type == EPSILON_SVR ||
		model->param.svm_type == NU_SVR)
		dec_values = Malloc(double, 1);
	else
		dec_values = Malloc(double, nr_class*(nr_class - 1) / 2);
	double pred_result = svm_predict_values(model, x, dec_values);
	free(dec_values);
	return pred_result;
}


double svm_predict_probability(const svm_model *model, const svm_node *x, double *prob_estimates)
{
	if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
		model->probA != NULL && model->probB != NULL)
	{
		int i;
		int nr_class = model->nr_class;
		double *dec_values = Malloc(double, nr_class*(nr_class - 1) / 2);
		svm_predict_values(model, x, dec_values);

		double min_prob = 1e-7;
		double **pairwise_prob = Malloc(double *, nr_class);
		for (i = 0; i<nr_class; i++)
			pairwise_prob[i] = Malloc(double, nr_class);
		int k = 0;
		for (i = 0; i<nr_class; i++)
			for (int j = i + 1; j<nr_class; j++)
			{
				pairwise_prob[i][j] = min(max(sigmoid_predict(dec_values[k], model->probA[k], model->probB[k]), min_prob), 1 - min_prob);
				pairwise_prob[j][i] = 1 - pairwise_prob[i][j];
				k++;
			}
		multiclass_probability(nr_class, pairwise_prob, prob_estimates);

		int prob_max_idx = 0;
		for (i = 1; i<nr_class; i++)
			if (prob_estimates[i] > prob_estimates[prob_max_idx])
				prob_max_idx = i;
		for (i = 0; i<nr_class; i++)
			free(pairwise_prob[i]);
		free(dec_values);
		free(pairwise_prob);
		return model->label[prob_max_idx];
	}
	else
		return svm_predict(model, x);
}

unsigned char* convert_rgb_to_gray(unsigned char *image, int nrows, int ncols) {

	int x, y;

	int size = nrows * ncols;

	unsigned char *out = (unsigned char *)malloc(size * sizeof(unsigned char));

	for (y = 0; y < nrows; y++) {
		for (x = 0; x < ncols; x++) {
			int r = image[3 * y * ncols + 3 * x + 0]; /*red*/
			int g = image[3 * y * ncols + 3 * x + 1]; /*green*/
			int b = image[3 * y * ncols + 3 * x + 2]; /*blue*/
													  //r /= 255.0; g /= 255.0; b /= 255.0;
			unsigned char gray = (unsigned char)(0.299*r + 0.587*g + 0.114*b);
			out[y * ncols + x] = gray;
		}
	}
	return out;
}

int svm_get_nr_class(const svm_model *model)
{
	return model->nr_class;
}

int svm_check_probability_model(const svm_model *model)
{
	return ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
		model->probA != NULL && model->probB != NULL) ||
		((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
			model->probA != NULL);
}


