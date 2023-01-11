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

#define TRUE 1
#define FALSE 0
#define L1 1

#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define M_PI   3.14159265358979323846264338327950288

#define SVM_MODEL_FILE "TextDetection/input/model.svm"
#define THOG_SETTINGS_FILE "TextDetection/input/1_7_9.txt"

typedef float Qfloat;
typedef signed char schar;

template <class T> static inline T min(T x, T y);
template <class T> static inline T max(T x, T y);

# define max(a, b) ((a) > (b) ? (a) : (b))

# define min(a, b) ((a) < (b) ? (a) : (b))

template <class T> static inline void Swap(T& x, T& y);
template <class S, class T> static inline void clone(T*& dst, S* src, int n);

//static void print_string_stdout(const char *s);


//static void(*svm_print_string) (const char *) = &print_string_stdout;

//#if 1
//static void info(const char *fmt, ...);
//#else
//static void info(const char *fmt, ...);
//#endif



static const bool thog_print_settings = false;

static char *Line = NULL;
static int max_line_len;


typedef struct _struct_thog {
	int nh;                    /*Height image normalization (pixels).*/
	int ncx;                   /*Number of cells in x direction.*/
	int ncy;                   /*Number of cells in y direction.*/
	int noc;                   /*Total number of cells.*/
	int bpc;                   /*Bins per cell.*/
	int nob;                   /*Total number of bins.*/
	int norm;                  /*Image normalization.*/
	char wnorm[512];           /*Image normalization weight.*/
	double rad;                /*Image normalization weight radius*/
	char grad[512];            /*Image gradient option*/
	char hmetric[512];         /*Histogram normalization metric.*/
	char weight_function[512];
	int deformable_weights;
	int debug;
} struct_thog;

struct svm_node
{
	int index;
	double value;
};

struct svm_parameter
{
	int svm_type;
	int kernel_type;
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */

					/* these are for training only */
	double cache_size; /* in MB */
	double eps;	/* stopping criteria */
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	int nr_weight;		/* for C_SVC */
	int *weight_label;	/* for C_SVC */
	double* weight;		/* for C_SVC */
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	double p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */
	int probability; /* do probability estimates */
};

struct svm_model
{
	struct svm_parameter param;	/* parameter */
	int nr_class;		/* number of classes, = 2 in regression/one class svm */
	int l;			/* total #SV */
	struct svm_node **SV;		/* SVs (SV[l]) */
	double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
	double *probA;		/* pariwise probability information */
	double *probB;
	int *sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

							/* for classification only */

	int *label;		/* label of each class (label[k]) */
	int *nSV;		/* number of SVs for each class (nSV[k]) */
					/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
					/* XXX */
	int free_sv;		/* 1 if svm_model is created by svm_load_model*/
						/* 0 if svm_model is created by svm_train */
} ;

//struct svm_model *model;

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */


double* thog(unsigned char *image, int nrows, int ncols, struct_thog sthog);

double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

static const char *svm_type_table[] =
{
	"c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

static const char *kernel_type_table[] =
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

using namespace cv;
using namespace std;
using namespace cv::motempl;


/* FIXME: implement the return of this function as a pointer to a newly
* allocated struct_thog */
struct_thog load_settings(const char *settings);

static char* readline(FILE *input);

svm_model *svm_load_model(const char *model_file_name);

//////////////////////////////////////////
//////////////////////////////////////////
//////////////////////////////////////////

double classify(const unsigned char *image, int nrows, int ncols, int x, int y, int w, int h, struct svm_model* model, double *prob, struct_thog sthog);

double** alloc_dmatrix(int ncols, int nrows);

void disalloc_dmatrix(double **matrix, int nrows);

void CalTempContrib(int start, int stop, double *tmpContrib, double *contrib);

int Clip(int x);

int HorizontalFilter(unsigned char *bufImg, int width, int startX, int stopX, int start, int stop, int y, double *pContrib);

int VerticalFilter(unsigned char *pbInImage, int width, int startY, int stopY, int start, int stop, int x, double *pContrib);

unsigned char *HorizontalFiltering(unsigned char *bufImage, int dwInW, int dwInH, int iOutW, int nDots, int nHalfDots, double *contrib, double *tmpContrib, double *normContrib);

unsigned char *VerticalFiltering(unsigned char *pbImage, int iW, int iH, int iOutH, int nDots, int nHalfDots, double *contrib, double *tmpContrib, double *normContrib);

unsigned char *resize_gray_uchar_image_bilinear(unsigned char *pixels, int w, int h, int w2, int h2);

double lanczos(int i, int inWidth, int outWidth, double support);

unsigned char *applylanczos(unsigned char *srcBi, int width, int height, int h, int w_mod, int *W);

int choose_gaussian_weight_size(double dev);

void compute_gaussian_weights(double *weight, int rwt);

void compute_binomial_weights(double *weight, int rwt);

double get_grey_avg(double *grey, int w, int h, int x, int y, double *x_weight, int x_rwt, double *y_weight, int y_rwt);

/*Get the deviation of a pixel given a normalizing window: */
double get_grey_dev(double *grey, int w, int h, int x, int y, double *x_weight, int x_rwt, double *y_weight, int y_rwt, double AVG, double noise);

double *normalize_grey_image(double *grey, int w, int h, double *x_weight, int x_rwt, double *y_weight, int y_rwt, double noise);

void convert_to_log_scale(double *grey, int w, int h, double eps);

double StepFunc(int n, int k, double z);

double BernsteinPoly(int n, int k, double z);

/*Computes the Bernstein polynomial of degree {n} and index {k} for the argument {z}.*/
double Bernstein(int n, int k, double z);

/**/

/*An edge-core weight function. If {n == 1} returns 1, if (n == 2) returns
*weight 1.0 near the edges, or 1.0 in the core region depending on {k}*/
double EdgeCore(int n, int k, double z);

double gaussian(double z, double mu, double sigma);

/*An exponential weight function: */
double Exp(int n, int k, double z);

double cell_weight(char *weight_function, int ncz, int cz, int z, double zmax, double zmin);

void gradient_simple(double *image, int width, int height, int x, int y, double *grad);

void get_bin_pos(int bins_per_cell, double dtheta, int *bin, double *factor);

double* thog(unsigned char *image, int nrows, int ncols, struct_thog sthog);

/******************************************************************************/
/*********************************** SVM **************************************/
/******************************************************************************/

static inline double powi(double base, int times);

/*
class QMatrix {
public:
	virtual Qfloat *get_Q(int column, int len);
	virtual double *get_QD();
	virtual void swap_index(int i, int j);
	virtual ~QMatrix();
};
*/

/*
class Kernel : public QMatrix
{
public:
	Kernel(int l, svm_node * const * x, const svm_parameter& param);
	virtual ~Kernel();

	static double k_function(const svm_node *x, const svm_node *y, const svm_parameter& param);
	virtual Qfloat *get_Q(int column, int len);
	virtual double *get_QD();
	virtual void swap_index(int i, int j);
protected:

	double (Kernel::*kernel_function)(int i, int j);

private:
	const svm_node **x;
	double *x_square;

	// svm_parameter
	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

	static double dot(const svm_node *px, const svm_node *py);
	double kernel_linear(int i, int j);
	double kernel_poly(int i, int j);
	double kernel_rbf(int i, int j);
	double kernel_sigmoid(int i, int j);
	double kernel_precomputed(int i, int j);
};

Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& param)
	:kernel_type(param.kernel_type), degree(param.degree),
	gamma(param.gamma), coef0(param.coef0);

Kernel::~Kernel();

double Kernel::dot(const svm_node *px, const svm_node *py);

double Kernel::k_function(const svm_node *x, const svm_node *y, const svm_parameter& param);
*/

double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values);

static double sigmoid_predict(double decision_value, double A, double B);

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
static void multiclass_probability(int k, double **r, double *p);

double svm_predict(const svm_model *model, const svm_node *x);

double svm_predict_probability(const svm_model *model, const svm_node *x, double *prob_estimates);

unsigned char* convert_rgb_to_gray(unsigned char *image, int nrows, int ncols);

int svm_get_nr_class(const svm_model *model);

int svm_check_probability_model(const svm_model *model);
