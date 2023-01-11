#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"

__global__ void _SegmentationBy_GPU(uchar* mt, uchar* motion, int h, int w);

void SegmentationGPU(const Mat& motion, Mat& dst);

__global__ void _Update_MHI_GPU(uchar*_GMHI, uchar* GMHI, uchar* GSILH, int h, int w, int MHI_DURATION);

void  update_mhiGPU(Mat& mhi, const Mat& img, Mat& dst, int diff_threshold);
