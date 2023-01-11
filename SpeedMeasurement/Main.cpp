#include <conio.h>
#include <stdio.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ratio>

//#include "opencv2/optflow/pcaflow.hpp"
#include "opencv2\core\cuda.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\optflow.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2\cudalegacy.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\xfeatures2d\cuda.hpp"
#include "opencv2\objdetect\detection_based_tracker.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\tracking.hpp"
#include "opencv2\cudaimgproc.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/video.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>


#include "Motion_Detection\motion_detection.h"
#include "Plate_Detection\plate_detection.h"
#include "TextDetection\TextDetection.h"
#include "KLT\KLT.h"

const bool GPU = false;

const string Video_File = "4.H264";
const string XML_File = "4.XML";
const int Frames_to_Track = 10;

const bool Show_Frame_Image = true;				 // Show Frame
const bool Show_MHI_Image = false;				 // Show MHI Image
const bool Show_SEG_Image = false;				 // Show Segmentation Image
const bool Show_Cropped_Image = true;            // Show Cropped Image After Select Frame
const bool Show_Vertical_Edge = false;           // Show Vertical Edge Image
const bool Show_After_Delete = false;            // Show Image After Delete Small and Large Components
const bool Show_Dilate_Image = false;            // Show Dilate Image
const bool Show_Plate_Dilate_Candidate = false;  // Show Plate Condidates in Dilated Image
const bool Show_Plate_Candidate = false;         // Show Plate Condidates
const bool Show_Plate_Region = false;            // Show Plate Condidates
const bool Show_Convert_Perspective = false;     // Show Image For Perspective
const bool Show_Tracking_Farems = false;         // Show Tracking Frames
const bool Show_Vertical_Projection = false;     // Show Vertical Projection
const bool Print_Vertical_Projections = false;   // Print Vertical Projection
const bool SHOW_TRACK_POINS = false;              // Show Good Features in Plate
const bool Show_Vehicle_Boundaries = true;       // Show a BOX around detected car
const bool SHOW_TRACK_LINES = false;				 // show tracking lines

using namespace cv;
using namespace std;
using namespace std::chrono;
using namespace cv::motempl;


using  ns = chrono::nanoseconds;
using get_time = chrono::steady_clock;

vector< vector<int> > com_attr;        //Save Components Attributes x1,y1,x2,y2
//int max_label, max_candid;
//int last = 0

// Projection
vector<int> Father;
vector< vector<int> > DilateSquares;

// temporary images
Mat MagImage, ComponentImage, DilateImage;

struct _tracking
{
	int lane;								// Which lane vehicle is
	int frame_start;						// Frame NO. detected vehicle plate
	int frame_end;							// Frame NO. finished tracking 
	bool moto;
	bool plate;
	bool radar;
	bool tracking;							// Begin Tracking or not
	vector<vector<KLT>> TrackingFeatures;   // Tracking Features
	double real_speed;   					// Vehicle Speed
	int frame_no;							// frame NO. that vehicle exist in it
	vector< vector<int> > candids_regions;
	double our_speed_KM;  // Our Speed K/H
	double our_speed_MS;  // Our Speed M/S
	int left_trim;
	int right_trim;
	int top_trim;
	int bottom_trim;
	Mat PreviusFrame;
	vector< Point2f > features[41];
	bool find_plate;

	double time_CPU_select_feature;
	double time_GPU_select_feature;
	vector<double> time_CPU_track_feature;
	vector<double> time_GPU_track_feature;
};

struct hmatrix hmat[3];

#ifdef _MSC_VER
#include <ctime>
inline double gets_time()
{
	return static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
}
#endif

void Save_Vehicle(vector<_tracking>& tracking, _XML XML, _vehicle_boundaries Vehicle_boundaries, vector< vector<int> > candids_regions, vector< Point2f > features, int real_plate, Mat PreFrame)
{
	tracking.resize(tracking.size() + 1);
	tracking.at(tracking.size()-1).lane = XML.lane;
	tracking.at(tracking.size()-1).frame_start = XML.frame_start;
	tracking.at(tracking.size()-1).frame_end = XML.frame_end;
	tracking.at(tracking.size()-1).moto = XML.moto;
	tracking.at(tracking.size()-1).radar = XML.radar;
	tracking.at(tracking.size()-1).plate = XML.plate;
	tracking.at(tracking.size()-1).real_speed = XML.speed;
	if ( candids_regions.size()>0 )
		tracking.at(tracking.size()-1).candids_regions = candids_regions;
	tracking.at(tracking.size() - 1).our_speed_KM = 0.0;
	tracking.at(tracking.size() - 1).our_speed_MS = 0.0;
	tracking.at(tracking.size()-1).frame_no = XML.frame_start+1;
	tracking.at(tracking.size()-1).top_trim = Vehicle_boundaries.top;
	tracking.at(tracking.size()-1).left_trim = Vehicle_boundaries.left;
	tracking.at(tracking.size()-1).right_trim = Vehicle_boundaries.right;
	tracking.at(tracking.size() - 1).bottom_trim = Vehicle_boundaries.bottom;
	//tracking.at(tracking.size() - 1).features->resize(41);
	tracking.at(tracking.size() - 1).features[0] = features;
	tracking.at(tracking.size() - 1).PreviusFrame = PreFrame;
	if ( real_plate<=0 )
		tracking.at(tracking.size() - 1).find_plate = false;
	else
		tracking.at(tracking.size() - 1).find_plate = true;
	//cout << " Find Plate = " << tracking.at(tracking.size() - 1).find_plate;
}


void Select_Good_Features(vector< Point2f >& features, Mat previous_frame, vector< vector<int> > candids_regions)
{
	/*
	cout << "\n\n[Found New Vehicle.]\n\n";
	cv::cvtColor(previous_frame, previous_frame, COLOR_BGR2GRAY);
	Mat mask = Mat::zeros(previous_frame.size(), CV_8UC1);
	Mat roi(mask, Rect(candids_regions[0][0], candids_regions[0][1], (candids_regions[0][2] - candids_regions[0][0]), (candids_regions[0][3] - candids_regions[0][1])));
	roi = Scalar(255, 255, 255);
	double CPU_start = gets_time();
	goodFeaturesToTrack(previous_frame, features, 10, 0.01, 1, mask, 11, false, 0.04);
	double CPU_end = gets_time();
	cerr << "\n\nCPU Select Features Time: " << CPU_end - CPU_start << " Milliseconds" << endl;  //Milliseconds
	*/
	
	//cout << "\n\n[Found New Vehicle.]\n\n";
	cv::cvtColor(previous_frame, previous_frame, COLOR_BGR2GRAY);
	Mat mask;
	Rect myROI(candids_regions[0][0], candids_regions[0][1], (candids_regions[0][2] - candids_regions[0][0]), (candids_regions[0][3] - candids_regions[0][1]));
	Mat croppedImage = previous_frame(myROI);
	//high_resolution_clock::time_point t1 = high_resolution_clock::now();
	goodFeaturesToTrack(croppedImage, features, 10, 0.01, 1, mask, 11, false, 0.04);
	//high_resolution_clock::time_point t2 = high_resolution_clock::now();
	//duration<double, std::milli> time_span = t2 - t1;
	//cout << "CPU time for select good features " << time_span.count() << " milliseconds.\n";
	for (int i = 0; i <features.size();i++)
	{
		features[i].x += candids_regions[0][0];
		features[i].y += candids_regions[0][1];
	}
}


static void download(const cuda::GpuMat& d_mat, vector<Point2f>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
	d_mat.download(mat);
}

static void download(const cuda::GpuMat& d_mat, vector<uchar>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
	d_mat.download(mat);
}


void Select_Good_Features_GPU(vector< Point2f >& features, Mat previous_frame, vector< vector<int> > candids_regions,cuda::GpuMat& Prev_Points)
{
	/*
	cout << "\n\n[Found New Vehicle.]\n\n";
	cv::cvtColor(previous_frame, previous_frame, COLOR_BGR2GRAY);

	Mat mask = Mat::zeros(previous_frame.size(), CV_8UC1);
	Mat roi(mask, Rect(candids_regions[0][0], candids_regions[0][1], (candids_regions[0][2] - candids_regions[0][0]), (candids_regions[0][3] - candids_regions[0][1])));
	roi = Scalar(255, 255, 255);

	cuda::GpuMat GPUmask;
	GPUmask.upload(mask);
	
	cuda::GpuMat G_FirstFrame;

	G_FirstFrame.upload(previous_frame);

	double CPU_start = gets_time();
	Ptr<cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cuda::SparsePyrLKOpticalFlow::create(Size(11, 11), 5, 1);
	Ptr<cuda::CornersDetector> detector = cuda::createGoodFeaturesToTrackDetector(G_FirstFrame.type(), 10, 0.01, 1);
	detector->detect(G_FirstFrame, Prev_Points, GPUmask);
	double CPU_end = gets_time();
	cerr << "\n\nGPU Select Features Time: " << CPU_end - CPU_start << " Milliseconds" << endl;  //Milliseconds

	download(Prev_Points, features);

	cout << "\nCols = " << Prev_Points.cols;
	cout << "\nRows = " << Prev_Points.rows<<"\n\n";

	Mat myMat;
	Prev_Points.download(myMat);
	cout << "\nCols = " << myMat.cols;
	cout << "\nRows = " << myMat.rows<<"\n\n";
	*/

	//cout << "\n\n[Found New Vehicle.]\n\n";

	cv::cvtColor(previous_frame, previous_frame, COLOR_BGR2GRAY);

	Mat mask;
	Rect myROI(candids_regions[0][0], candids_regions[0][1], (candids_regions[0][2] - candids_regions[0][0]), (candids_regions[0][3] - candids_regions[0][1]));
	Mat croppedImage = previous_frame(myROI);

	cuda::GpuMat G_FirstFrame;

	G_FirstFrame.upload(croppedImage);

	//high_resolution_clock::time_point t1 = high_resolution_clock::now();
	Ptr<cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cuda::SparsePyrLKOpticalFlow::create(Size(11, 11), 5, 1);
	Ptr<cuda::CornersDetector> detector = cuda::createGoodFeaturesToTrackDetector(G_FirstFrame.type(), 10, 0.01, 1);
	detector->detect(G_FirstFrame, Prev_Points);
	//high_resolution_clock::time_point t2 = high_resolution_clock::now();
	//duration<double, std::milli> time_span = t2 - t1;
	//cout << "GPU time for select good features " << time_span.count() << " milliseconds.\n";

	download(Prev_Points, features);

	for (int i = 0; i < features.size();i++)
	{
		features[i].x += candids_regions[0][0];
		features[i].y += candids_regions[0][1];
	}

	//cout << "\nCols = " << Prev_Points.cols;
	//cout << "\nRows = " << Prev_Points.rows << "\n\n";

	Mat myMat;
	Prev_Points.download(myMat);
	//cout << "\nCols = " << myMat.cols;
	//cout << "\nRows = " << myMat.rows << "\n\n";
	
}


void Set_Vehicle_Plate_To_ImgSpeed(vector< vector<int> >& candids_regions, _vehicle_boundaries Vehicle_boundaries, int height)
{
	for (int i = 0; i < candids_regions.size(); i++)
	{
		candids_regions[i][1] += Vehicle_boundaries.top;
		candids_regions[i][3] += Vehicle_boundaries.top;
		candids_regions[i][0] += Vehicle_boundaries.left;
		candids_regions[i][2] += Vehicle_boundaries.left;

	}
}

void Track_Good_Features(Mat previous_frame, Mat current_frame, vector< Point2f > features, vector< Point2f >& features1)
{
	cv::cvtColor(previous_frame, previous_frame, COLOR_BGR2GRAY);
	cv::cvtColor(current_frame, current_frame, COLOR_BGR2GRAY);
	Mat status, err;
	TermCriteria tc = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);
	//high_resolution_clock::time_point t1 = high_resolution_clock::now();
	calcOpticalFlowPyrLK(previous_frame, current_frame, features, features1, status, err, Size(11, 11), 5, tc, 0, 0.0001);
	//high_resolution_clock::time_point t2 = high_resolution_clock::now();
	//duration<double, std::milli> time_span = t2 - t1;
	//cout << "CPU time for track good features " << time_span.count() << " milliseconds.\n";
}

void Track_Good_Features_GPU(Mat previous_frame, Mat current_frame, vector< Point2f > features, vector< Point2f >& features1)
{
	Mat status, err;
	cuda::GpuMat p_frame, c_frame;
	cuda::GpuMat Prev_Points, Next_Points;
	cuda::GpuMat d_status;

	cv::cvtColor(previous_frame, previous_frame, COLOR_BGR2GRAY);
	cv::cvtColor(current_frame, current_frame, COLOR_BGR2GRAY);

	p_frame.upload(previous_frame);
	c_frame.upload(current_frame);
	Mat m(1, 20, CV_32FC2, (void*)&features[0]);  // Copy features to Mat

	Prev_Points.upload(m);

	//high_resolution_clock::time_point t1 = high_resolution_clock::now();
	///
	Ptr<cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cuda::SparsePyrLKOpticalFlow::create(Size(11, 11), 5, 1);
	//Ptr<cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cuda::SparsePyrLKOpticalFlow::create(Size(11, 11), 5, 1);
	d_pyrLK_sparse->calc(p_frame, c_frame, Prev_Points, Next_Points, d_status);
    //high_resolution_clock::time_point t2 = high_resolution_clock::now();
	//duration<double, std::milli> time_span = t2 - t1;
	//cout << "GPU time for track good features " << time_span.count() << " milliseconds.\n";

	vector<Point2f> nextPoints(Next_Points.cols);
	download(Next_Points, nextPoints);

	features1 = nextPoints;
	features1.resize(10);

	/*for (int i = 0; i < features1.size(); i++)
		cout << features1[i] << "\n";

	cout << "Feature Size = " << features1.size() << "\n";
	waitKey(0);*/
}


void Save_Tracking(vector<_tracking>& tracking, int andis, Mat current_frame, vector< Point2f > newFeatures)
{
	tracking[andis].frame_no++;
	tracking[andis].PreviusFrame = current_frame;
	tracking[andis].features[(tracking[andis].frame_no-tracking[andis].frame_start)-1] = newFeatures;
}


void Halculate_H_Matrix(void)
{
	Mat ipm_matrix[3];  //H Mtarix for 3 Lines
	Mat Hframe;
	struct quadrangle rect[3];
	FILE *fhmatrix;

	VideoCapture capH(Video_File);
	capH >> Hframe;
	cvtColor(Hframe, Hframe, COLOR_BGR2GRAY);

	// Lane 1
	line(Hframe, Point(530, 50), Point(500, 270), CV_RGB(255, 255, 255), 2, 8, 0);  // right
	line(Hframe, Point(320, 70), Point(250, 290), CV_RGB(255, 255, 255), 2, 8, 0);  // left
	line(Hframe, Point(250, 290), Point(500, 270), CV_RGB(255, 255, 255), 2, 8, 0);  // bottom
	line(Hframe, Point(320, 70), Point(530, 50), CV_RGB(255, 255, 255), 2, 8, 0);  // top

    // Line 2
	line(Hframe, Point(670, 270), Point(920, 260), CV_RGB(255, 255, 255), 2, 8, 0);  // bottom
	line(Hframe, Point(870, 50), Point(920, 260), CV_RGB(255, 255, 255), 2, 8, 0);  // right
	line(Hframe, Point(675, 50), Point(670, 270), CV_RGB(255, 255, 255), 2, 8, 0);  // left
	line(Hframe, Point(675, 50), Point(870, 50), CV_RGB(255, 255, 255), 2, 8, 0);  // top

	// Line 3
	line(Hframe, Point(1200, 255), Point(1450, 260), CV_RGB(255, 255, 255), 2, 8, 0);  // bottom
	line(Hframe, Point(1100, 50), Point(1200, 255), CV_RGB(255, 255, 255), 2, 8, 0);  // left
	line(Hframe, Point(1300, 50), Point(1450, 260), CV_RGB(255, 255, 255), 2, 8, 0);  // right
	line(Hframe, Point(1100, 50), Point(1300, 50), CV_RGB(255, 255, 255), 2, 8, 0);  // top

	if (Show_Convert_Perspective == true)
	{
		namedWindow("Box For Convert", WINDOW_NORMAL);
		resizeWindow("Box For Convert", Hframe.cols / 2, Hframe.rows / 2);
		imshow("Box For Convert", Hframe);
		waitKey(0);
	}
	// First Road Line 
	rect[0].p_top_l.x = 320.0;  // Top Left
	rect[0].p_top_l.y = 70.0;
	rect[0].p_top_r.x = 530.0;	// Top Right
	rect[0].p_top_r.y = 50.0;
	rect[0].p_bot_l.x = 250.0;	// Bottom Left
	rect[0].p_bot_l.y = 290.0;
	rect[0].p_bot_r.x = 500.0;	// Bottom Right
	rect[0].p_bot_r.y = 270.0;
	rect[0].pixels_per_m = PIXELS_PER_M;
	rect[0].ipm_left = 0.0;
	rect[0].ipm_top = 0.0;
	rect[0].ref_height = 4.8;
	rect[0].ref_width = 2.0;


	ipm_matrix[0] = generate_hmatrix(&hmat[0], &rect[0]);
	apply_mat_to_image(Hframe, "line1.png", ipm_matrix[0], "NO-SHOW");
	//print_hmatrix(&hmat[0], 1);



	// Second Road Line 
	rect[1].p_top_l.x = 675.0;  // Top Left
	rect[1].p_top_l.y = 50.0;
	rect[1].p_top_r.x = 870.0;	// Top Right
	rect[1].p_top_r.y = 50.0;
	rect[1].p_bot_l.x = 670.0;	// Bottom Left
	rect[1].p_bot_l.y = 270.0;
	rect[1].p_bot_r.x = 920.0;	// Bottom Right
	rect[1].p_bot_r.y = 260.0;
	rect[1].pixels_per_m = PIXELS_PER_M;
	rect[1].ipm_left = 0.0;
	rect[1].ipm_top = 0.0;
	rect[1].ref_height = 4.8;
	rect[1].ref_width = 2.0;

	ipm_matrix[1] = generate_hmatrix(&hmat[1], &rect[1]);
	apply_mat_to_image(Hframe, "line2.png", ipm_matrix[1], "NO-SHOW");
	//print_hmatrix(&hmat[1], 2);


	// Third Road Line 
	rect[2].p_top_l.x = 1100.0;  // Top Left
	rect[2].p_top_l.y = 50.0;
	rect[2].p_top_r.x = 1300.0;	// Top Right
	rect[2].p_top_r.y = 50.0;
	rect[2].p_bot_l.x = 1200.0;	// Bottom Left
	rect[2].p_bot_l.y = 255.0;
	rect[2].p_bot_r.x = 1450.0;	// Bottom Right
	rect[2].p_bot_r.y = 260.0;
	rect[2].pixels_per_m = PIXELS_PER_M;
	rect[2].ipm_left = 0.0;
	rect[2].ipm_top = 0.0;
	rect[2].ref_height = 4.8;
	rect[2].ref_width = 2.0;

	ipm_matrix[2] = generate_hmatrix(&hmat[2], &rect[2]);
	apply_mat_to_image(Hframe, "line3.png", ipm_matrix[2], "NO-SHOW");
	//print_hmatrix(&hmat[2], 2);

}



void Save_Tracking_To_File(vector<_tracking>& tracking, int andis, ofstream& tracking_file)
{
	//fwrite(&tracking[andis], sizeof(struct _tracking), 1, tracking_file);
	//tracking_file.write(reinterpret_cast<char*>(&tracking[andis]), sizeof(tracking[andis]));
	tracking_file << tracking[andis].frame_start << endl;
	tracking_file << tracking[andis].frame_end << endl;
	tracking_file << tracking[andis].lane << endl;
	tracking_file << tracking[andis].moto << endl;
	tracking_file << tracking[andis].plate << endl;
	tracking_file << tracking[andis].radar << endl;
	tracking_file << tracking[andis].real_speed << endl;
	tracking_file << tracking[andis].find_plate << endl;
	tracking_file << tracking[andis].our_speed_MS << endl;
	tracking_file << tracking[andis].our_speed_KM << endl;
	tracking.erase(tracking.begin() + andis);
	//cout << "\n\n[Save Vehicle Information To File.]\n\n";
}


void Calculate_Speed(vector<_tracking>& tracking, int andis)
{
	vector<vector<KLT>> NewFeatures;
	speed_i speed;

	NewFeatures.resize(41, vector<KLT>(10));

	for (int i = 0; i < Frames_to_Track; i++) // n Frame Tracking
		for (int j = 0; j < tracking[andis].features[i].size(); j++) // n Features
			compute_ipm_features(tracking[andis].features[i][j].x, tracking[andis].features[i][j].y, NewFeatures, i, j, tracking[andis].lane, hmat);

	double s = 0.0, _s = 0.0;
	int tracks = 0;
	for (int i = 0; i < Frames_to_Track-1; i++) // n Frame Tracking
	{
		speed = compute_velocity_vector(NewFeatures, i, i + 1);
		//speed = compute_velocity_vector(NewFeatures, i, i - 1);
		_s = metric_calc_velocity_ipm(speed, tracking[andis].lane);
		s += _s;
		if (_s > 0)
			tracks++;
	}
	tracking[andis].our_speed_MS = (s / (Frames_to_Track - 1));
	tracking[andis].our_speed_KM = (s / (Frames_to_Track - 1))*(3.6);
	cout << "\n> Lane               : " << tracking[andis].lane << "\n\n";
	cout << "> Our Speed k/h      : " << tracking[andis].our_speed_KM << "\n\n";
	cout << "> Real Speed k/h     : " << tracking[andis].real_speed << "\n\n";
	cout << "> Speed Difference   : " << (tracking[andis].real_speed - tracking[andis].our_speed_KM) << "\n\n";
	cout << "----------------------------------------" << "\n\n";
}


void PrintDetectedVehicles(const double All_Vehicles, const double Detected_Vehicles, const double Undetected_Vehicles)
{
	//All_Vehicles, Detected_Vehicles, Undetected_Vehicles, (100 * Detected_Vehicles) / All_Vehicles
	//cout << "\n\n>> From [" << All_Vehicles << "] vehicles we detected [" << Detected_Vehicles << "] Vehicles (%" << (100 * Detected_Vehicles) / All_Vehicles << ")\n\n";
}


void TrackingInfo(void)
{
	int cars = 0;
	_tracking trk;
	ifstream in("tracking_file.txt");
	in >> trk.frame_start;
	in >> trk.frame_end;
	in >> trk.lane;
	in >> trk.moto;
	in >> trk.plate;
	in >> trk.radar;
	in >> trk.real_speed;
	in >> trk.find_plate;
	in >> trk.our_speed_MS;
	in >> trk.our_speed_KM;
	while (!in.eof())
	{
		cout << "*************************************************\n";
		cout << "Frame Start : " << trk.frame_start << "\n";
		cout << "Frame End : " << trk.frame_end << "\n";
		cout << "Lane : " << trk.lane << "\n";
		cout << "Moto : " << trk.moto << "\n";
		cout << "Plate : " << trk.plate << "\n";
		cout << "Radar : " << trk.radar << "\n";
		cout << "Find Plate : " << trk.find_plate << "\n";
		cout << "Our Speed (m/s) : " << trk.our_speed_MS << "\n";
		cout << "Our Speed (k/h) : " << trk.our_speed_KM << "\n";
		cout << "Real Spead : " << trk.real_speed << "\n";
		cout << "*************************************************\n";
		cars++;
		in >> trk.frame_start;
		in >> trk.frame_end;
		in >> trk.lane;
		in >> trk.moto;
		in >> trk.plate;
		in >> trk.radar;
		in >> trk.real_speed;
		in >> trk.find_plate;
		in >> trk.our_speed_MS;
		in >> trk.our_speed_KM;
	}
	in.close();
	//cout << "\n\n" << cars << " Cars\n";
}


///////////////////////////////// Main /////////////////////////////////

void Get_Vehicle_Box(const Mat mhi, int lane, _vehicle_boundaries &VBoundaries)
{
	// Get Vehicle Box
	int top, bottom, left, right, start, end;
	top = 1000;
	left = 2000;
	bottom = right = 0;
	if (lane == 3) { start = 1200; end = mhi.cols; }
	else
		if (lane == 2) { start = 450; end = 1200; }
		else
			if (lane == 1) { start = 0; end = 500; }
	for (int x = start; x < end; x += 2)
		for (int y = 500; y < mhi.rows; y += 2)
		{
			if (mhi.at<uchar>(y, x) != 0 && y < top)
				top = y;
			if (mhi.at<uchar>(y, x) != 0 && y > bottom)
				bottom = y;
			if (mhi.at<uchar>(y, x) != 0 && x < left)
				left = x;
			if (mhi.at<uchar>(y, x) != 0 && x > right)
				right = x;
		}
	if (lane == 3) right = mhi.cols;
	VBoundaries.left = left;
	VBoundaries.right = right;
	VBoundaries.top = top;
	VBoundaries.bottom = bottom;
}


void main(void)
{
	_XML XML;
	_vehicle_boundaries Vehicle_boundaries;
	int frame_number = 0, components, max_label, max_candid;
	ifstream inFile;
	vector<int> ver_proj;
	vector<_tracking> tracking;
	vector< vector<int> > candids_regions;  //Save Plates Cindids Regions
	vector< Point2f > features, FeaturesList[41];

	cuda::GpuMat Prev_Points, Next_Points;



	//////////////////////////////////////
	/////////////////////////////////////
	/////////////////////////////////////
	/*
	Mat cmp = Mat::zeros(Size(20,20),CV_8U);
	Mat stats, centroids, cmp1;
	cmp.at<uchar>(5, 5) = 1;
	cmp.at<uchar>(6, 5) = 1;
	cmp.at<uchar>(7, 5) = 1;
	cmp.at<uchar>(8, 5) = 1;
	cmp.at<uchar>(5, 6) = 1;
	cmp.at<uchar>(6, 6) = 1;
	cmp.at<uchar>(7, 6) = 1;
	cmp.at<uchar>(8, 6) = 1;
	cmp.at<uchar>(5, 7) = 1;
	cmp.at<uchar>(6, 7) = 1;
	cmp.at<uchar>(7, 7) = 1;
	cmp.at<uchar>(8, 7) = 1;

	cmp.at<uchar>(4, 10) = 1;
	cmp.at<uchar>(4, 11) = 1;
	cmp.at<uchar>(4, 12) = 1;
	cmp.at<uchar>(4, 13) = 1;
	cmp.at<uchar>(5, 10) = 1;
	cmp.at<uchar>(5, 11) = 1;
	cmp.at<uchar>(5, 12) = 1;
	cmp.at<uchar>(5, 13) = 1;
	cmp.at<uchar>(6, 10) = 1;
	cmp.at<uchar>(6, 11) = 1;
	cmp.at<uchar>(6, 12) = 1;
	cmp.at<uchar>(6, 13) = 1;

	cmp.at<uchar>(10, 13) = 1;
	cmp.at<uchar>(10, 14) = 1;
	cmp.at<uchar>(11, 13) = 1;
	cmp.at<uchar>(11, 14) = 1;

	cmp.at<uchar>(12, 16) = 1;
	cmp.at<uchar>(12, 17) = 1;
	cmp.at<uchar>(13, 16) = 1;
	cmp.at<uchar>(13, 17) = 1;
	cmp.at<uchar>(12, 18) = 1;
	cmp.at<uchar>(13, 18) = 1;
	cmp.at<uchar>(11, 16) = 1;
	cmp.at<uchar>(11, 17) = 1;
	cmp.at<uchar>(11, 18) = 1;

	cmp.at<uchar>(15, 10) = 1;
	cmp.at<uchar>(16, 10) = 1;
	cmp.at<uchar>(17, 10) = 1;
	cmp.at<uchar>(18, 10) = 1;
	cmp.at<uchar>(15, 11) = 1;
	cmp.at<uchar>(16, 11) = 1;
	cmp.at<uchar>(17, 11) = 1;
	cmp.at<uchar>(18, 11) = 1;
	cmp.at<uchar>(15, 12) = 1;
	cmp.at<uchar>(16, 12) = 1;
	cmp.at<uchar>(17, 12) = 1;
	cmp.at<uchar>(18, 12) = 1;
	cmp.at<uchar>(15, 13) = 1;
	cmp.at<uchar>(16, 13) = 1;
	cmp.at<uchar>(17, 13) = 1;
	cmp.at<uchar>(18, 13) = 1;
	cmp.at<uchar>(15, 14) = 1;
	cmp.at<uchar>(16, 14) = 1;
	cmp.at<uchar>(17, 14) = 1;
	cmp.at<uchar>(18, 14) = 1;
	cmp.at<uchar>(15, 15) = 1;
	cmp.at<uchar>(16, 15) = 1;
	cmp.at<uchar>(17, 15) = 1;
	cmp.at<uchar>(18, 15) = 1;
	cmp.at<uchar>(15, 16) = 1;
	cmp.at<uchar>(16, 16) = 1;
	cmp.at<uchar>(17, 16) = 1;
	cmp.at<uchar>(18, 16) = 1;
	
	
	ofstream fout1("cmp1.txt");
	for (int i = 0; i<cmp.rows; i++)
	{
		for (int j = 0; j<cmp.cols; j++)
		{
			fout1 << (int)cmp.at<uchar>(i, j) << "\t";
		}
		fout1 << endl;
	}
	fout1.close();
	
	connectedComponentsWithStats(cmp, cmp1, stats, centroids, 4);

	DeleteSmallLargeComponent(cmp1, stats, 3, 10);

	ofstream fout2("cmp2.txt");
	for (int i = 0; i<cmp1.rows; i++)
	{
		for (int j = 0; j<cmp1.cols; j++)
		{
			fout2 << cmp1.at<int>(i, j) << "\t";
		}
		fout2 << endl;
	}
	fout2.close();
	*/
	/////////////////////////////////////
	/////////////////////////////////////
	/////////////////////////////////////

	//////////////////////////
	/*
	Mat src_host = imread("test.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cuda::GpuMat dst, src;
	src.upload(src_host);

	threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);

	Mat result_host;
	dst.download(result_host);

	cout << "number of CUDA devices: " << cuda::getCudaEnabledDeviceCount() << endl;
	int device_id = 0;
	cuda::printCudaDeviceInfo(device_id);
	cout << "CUDA device info: " << endl;

	imshow("Result", result_host);
	waitKey(0);	
	*/
	//////////////////////////

	VideoCapture cap(Video_File);


	//cap.set(CV_CAP_PROP_POS_FRAMES, 105);

	buf.resize(2);
	Mat image, motion, mhi, segment, tmp, previous_frame, image1, image2, ImgSpeed;

	namedWindow("Frame", WINDOW_NORMAL);
	resizeWindow("Frame", 400, 300);
	moveWindow("Frame", 0, 0);
	if (Show_MHI_Image == true)
	{
		namedWindow("MHI", WINDOW_NORMAL);
		resizeWindow("MHI", 400, 300);
		moveWindow("MHI", 405, 0);
	}
	int frame_no = 0;
	int frame_selected = 55;
	int Prev_Frame_NO = 0;

	inFile=Open_XML_File(XML_File);  // Open XML File
	XML = Read_XML_File(inFile);   // Read from XML file
	//Show_XML(XML);  //Shoow XML readed data

	Halculate_H_Matrix();

	double All_Vehicles = 0.0, Detected_Vehicles = 0.0, Undetected_Vehicles = 0.0;

	ofstream tracking_file("tracking_file.txt");


	////high_resolution_clock::time_point t1 = high_resolution_clock::now();

	for (;;)
	{
		if (XML.frame_start != Prev_Frame_NO)
		{
			cap >> image;
			if (Show_Frame_Image == true)
				imshow("Frame", image);
			Mat ppp = image;
			imwrite("frame.jpg", image);
			frame_no++;
			//printf("Frame NO. : %i\n", frame_no);
			for (int i = 0; i < tracking.size(); i++)
			{
				if (tracking[i].radar == false || tracking[i].find_plate == false || tracking[i].plate==false)
				{
					Save_Tracking_To_File(tracking,i, tracking_file);
					if (tracking.size() <= 0) break;
				}
				//if (tracking[i].radar == true && tracking[i].frame_no <= tracking[i].frame_end && tracking[i].find_plate == true)
				if (tracking[i].radar == true && tracking[i].frame_no <= tracking[i].frame_start+ Frames_to_Track && tracking[i].find_plate == true)
				{
					Mat current_frame;  
					image.copyTo(current_frame);
					vector< Point2f > newFeatures;
					if (GPU == false)
					{
						Track_Good_Features(tracking[i].PreviusFrame, current_frame, tracking[i].features[(tracking[i].frame_no - tracking[i].frame_start) - 1], newFeatures);
					}
					////// GPU //////
					if (GPU == true)
					{
						Track_Good_Features_GPU(tracking[i].PreviusFrame, current_frame, tracking[i].features[(tracking[i].frame_no - tracking[i].frame_start) - 1], newFeatures);
					}
					/////////////////
					Save_Tracking(tracking,i,current_frame, newFeatures);

					// Show Tracking Points
					if (GPU == false)
					{
						if (SHOW_TRACK_POINS == true)
						{
							for (int j = 0; j < tracking[i].features[(tracking[i].frame_no - tracking[i].frame_start) - 1].size(); j++)
								circle(image, tracking[i].features[(tracking[i].frame_no - tracking[i].frame_start) - 1][j], 3, CV_RGB(255, 255, 0), -1);
							imshow("Frame", image);
						}
					}
					if (GPU == true)
					{
						if (SHOW_TRACK_POINS == true)
						{
							for (int j = 0; j < tracking[i].features[(tracking[i].frame_no - tracking[i].frame_start) - 1].size(); j++)
								circle(image, tracking[i].features[(tracking[i].frame_no - tracking[i].frame_start) - 1][j], 3, CV_RGB(0, 255, 255), -1);
							imshow("Frame", image);
						}
					}
				}
				//if (tracking[i].radar == true && tracking[i].frame_no>tracking[i].frame_end && tracking[i].find_plate == true)
				if (tracking[i].radar == true && tracking[i].frame_no >= tracking[i].frame_start + Frames_to_Track && tracking[i].find_plate == true)
				{
					if (SHOW_TRACK_LINES == true)
					{
						for (int i = 0; i < tracking.size(); i++)
							for (int j = 0; j < tracking[i].features[i].size(); j++)
							{
								circle(image, tracking[i].features[0][j], 3, CV_RGB(255, 0, 0), 2);
								line(ppp, tracking[i].features[0][j], tracking[i].features[1][j], CV_RGB(0, 255, 255), 2, 8, 0);
								circle(image, tracking[i].features[1][j], 3, CV_RGB(255, 0, 0), 2);
								line(ppp, tracking[i].features[1][j], tracking[i].features[2][j], CV_RGB(0, 255, 255), 2, 8, 0);
								circle(image, tracking[i].features[2][j], 3, CV_RGB(255, 0, 0), 2);
								line(ppp, tracking[i].features[2][j], tracking[i].features[3][j], CV_RGB(0, 255, 255), 2, 8, 0);
								circle(image, tracking[i].features[3][j], 3, CV_RGB(255, 0, 0), 2);
								line(ppp, tracking[i].features[3][j], tracking[i].features[4][j], CV_RGB(0, 255, 255), 2, 8, 0);
								circle(image, tracking[i].features[4][j], 3, CV_RGB(255, 0, 0), 2);
								line(ppp, tracking[i].features[4][j], tracking[i].features[5][j], CV_RGB(0, 255, 255), 2, 8, 0);
								circle(image, tracking[i].features[5][j], 3, CV_RGB(255, 0, 0), 2);
								line(ppp, tracking[i].features[5][j], tracking[i].features[6][j], CV_RGB(0, 255, 255), 2, 8, 0);
								circle(image, tracking[i].features[6][j], 3, CV_RGB(255, 0, 0), 2);
								line(ppp, tracking[i].features[6][j], tracking[i].features[7][j], CV_RGB(0, 255, 255), 2, 8, 0);
								circle(image, tracking[i].features[7][j], 3, CV_RGB(255, 0, 0), 2);
								line(ppp, tracking[i].features[7][j], tracking[i].features[8][j], CV_RGB(0, 255, 255), 2, 8, 0);
								circle(image, tracking[i].features[8][j], 3, CV_RGB(255, 0, 0), 2);
								line(ppp, tracking[i].features[8][j], tracking[i].features[9][j], CV_RGB(0, 255, 255), 2, 8, 0);
							}
						namedWindow("ppp", WINDOW_NORMAL);
						moveWindow("ppp", 405,0);
						resizeWindow("ppp", 400, 300);
						imshow("ppp", ppp);
						imwrite("TRACK_LINES.jpg", image);
						//////waitKey(0);
					}
					//high_resolution_clock::time_point t1 = high_resolution_clock::now();
					Calculate_Speed(tracking, i);
					Calculate_Speed(tracking, i);
					//high_resolution_clock::time_point t2 = high_resolution_clock::now();
					//duration<double, std::milli> time_span = t2 - t1;
					//cout << "*** CPU time for Speed Measuring [" << time_span.count() << "] milliseconds.\n";

					//high_resolution_clock::time_point t2 = high_resolution_clock::now();
					//duration<double, std::milli> time_span = t2 - t1;
					//cout << "*** CPU time for Speed Calculation [" << time_span.count() << "] milliseconds.\n";
					Save_Tracking_To_File(tracking, i, tracking_file);
					i--;
					if (tracking.size() <= 0) break;
				}
			}
		}
		if (image.empty())
			break;

		//cout << "mhi.size    = " << mhi.size() << "\n";
		//cout << "image.size  = " << image.size() << "\n";
		//cout << "motion.size = " << motion.size() << "\n";
		if (GPU == false)
		{
			//high_resolution_clock::time_point t1 = high_resolution_clock::now();
			update_mhi(mhi, image, motion, 60);
			//high_resolution_clock::time_point t2 = high_resolution_clock::now();
			//duration<double, std::milli> time_span = t2 - t1;
			//cout << "CPU MHI time [" << time_span.count() << "] milliseconds.\n";
		}
		else
		{
			//high_resolution_clock::time_point t1 = high_resolution_clock::now();
			update_mhi_GPU(mhi, image, motion, 60);
			//high_resolution_clock::time_point t2 = high_resolution_clock::now();
			//duration<double, std::milli> time_span = t2 - t1;
			//cout << "GPU MHI time [" << time_span.count() << "] milliseconds.\n";
		}

		if (GPU == false)
		{
			//high_resolution_clock::time_point t1 = high_resolution_clock::now();
			Segmentation(motion, segment);
			//high_resolution_clock::time_point t2 = high_resolution_clock::now();
			//duration<double, std::milli> time_span = t2 - t1;
			//cout << "CPU Segmentation time [" << time_span.count() << "] milliseconds.\n";
		}
		else
		{
			//high_resolution_clock::time_point t1 = high_resolution_clock::now();
			SegmentationGPU(motion, segment);
			//high_resolution_clock::time_point t2 = high_resolution_clock::now();
			//duration<double, std::milli> time_span = t2 - t1;
			//cout << "GPU Segmentation time [" << time_span.count() << "] milliseconds.\n";
		}

		if (frame_no == XML.frame_start)
		{
			if (XML.radar == true && XML.plate == true)
			{
				ver_proj = Projection(segment, Vehicle_boundaries, Show_Vertical_Projection); // Vertical Projection
				ProjectionSmooth(ver_proj);
				Print_Vertical_Projection(ver_proj, Print_Vertical_Projections);
				Get_Boundaries(ver_proj, XML.lane, Vehicle_boundaries);
				//Get_Vehicle_Box(mhi, XML.lane, Vehicle_boundaries);
				//cout << "Left   " << Vehicle_boundaries.left << "\n";
				//cout << "Right  " << Vehicle_boundaries.right << "\n";
				//cout << "Top    " << Vehicle_boundaries.top << "\n";
				//cout << "Bottom " << Vehicle_boundaries.bottom << "\n";
				Rect myROI(Vehicle_boundaries.left, Vehicle_boundaries.top, Vehicle_boundaries.right - Vehicle_boundaries.left, Vehicle_boundaries.bottom - Vehicle_boundaries.top);
				Mat croppedImage = image(myROI);
				//**************************************************************
				
				//high_resolution_clock::time_point t1 = high_resolution_clock::now();

				Vertical_Edge(croppedImage, MagImage, Show_Vertical_Edge);
				//Mat v = MagImage * 255;
				//imwrite("Vertical_Edges.jpg", v);

				Mat stats, centroids;
				connectedComponentsWithStats(MagImage, ComponentImage, stats, centroids, 4);

				DeleteSmallLargeComponent(ComponentImage, stats, 4, 120);
				//imwrite("After_Delete_Edges.jpg", ComponentImage);
				
				Dilate(ComponentImage, DilateImage, Show_Dilate_Image);
				//imwrite("Dilate.jpg", DilateImage);
				connectedComponentsWithStats(DilateImage, ComponentImage, stats, centroids, 4);


				//////////////////////////////////
				/*
				ofstream fout("cmp.txt");
				for (int i = 0; i<ComponentImage.rows; i++)
				{
					for (int j = 0; j<ComponentImage.cols; j++)
					{
						fout << ComponentImage.at<int>(i, j) << "\t";
					}
					fout << endl;
				}
				fout.close();
				*/
				//////////////////////////////////

				//high_resolution_clock::time_point t1 = high_resolution_clock::now();

				FindCandidatesPlateRegions(ComponentImage, stats, Father);

				//high_resolution_clock::time_point t2 = high_resolution_clock::now();
				//duration<double, std::milli> time_span = t2 - t1;
				//cout << "*** CPU time for Find Candid Plates Regions [" << time_span.count() << "] milliseconds.\n";

				max_candid = GetRegions(stats, Father, candids_regions);

				//waitKey(0);

				////// Show //////
				Show_Plate_Candids(croppedImage, candids_regions, Show_Plate_Candidate);
				/////////////////

				imwrite("TestCandids.jpg", croppedImage);
				croppedImage.release();
				croppedImage = imread("TestCandids.jpg");

				croppedImage.copyTo(tmp);

				//high_resolution_clock::time_point t1 = high_resolution_clock::now();

				int real_plate = Find_Plate_Candid_Content_Text(croppedImage, candids_regions, Show_Plate_Region);

				//high_resolution_clock::time_point t2 = high_resolution_clock::now();
				//duration<double, std::milli> time_span = t2 - t1;
				//cout << "*** CPU time for Find Plates Regions [" << time_span.count() << "] milliseconds.\n";

				if (XML.moto == false && XML.plate == true && XML.radar==true )
				{
					All_Vehicles++;
					if (real_plate >= 1) Detected_Vehicles++; else Undetected_Vehicles++;
					//t1 = high_resolution_clock::now();

				}

				Mat ImgSpeed;
				image.copyTo(ImgSpeed);

				if (XML.radar == true && XML.plate == true && real_plate > 0 && XML.radar==true)
				{
					Set_Vehicle_Plate_To_ImgSpeed(candids_regions, Vehicle_boundaries, image.size().height);
					if (real_plate > 0)
					{
						if (GPU == false)
						{
							Select_Good_Features(features, ImgSpeed, candids_regions);
						}
						////// GPU //////
						if (GPU == true)
						{
							Select_Good_Features_GPU(features, ImgSpeed, candids_regions, Prev_Points);
							for (int j = 0; j < 10; j++)
								circle(image, features[j], 3, CV_RGB(0, 255, 255), -1);
						}

						////////////////////////////////
					}
					Save_Vehicle(tracking, XML, Vehicle_boundaries, candids_regions, features, real_plate, ImgSpeed);
					// Show Tracking Points
					if (SHOW_TRACK_POINS == true)
					{
						for (int j = 0; j < features.size(); j++)
							circle(image, features[j], 3, CV_RGB(255, 255, 0), -1);
						imshow("Frame", image);
						imwrite("track_points.jpg", image);
					}
				}
				//*****************************************************************

				if (Show_Cropped_Image == true)
				{
					namedWindow("Cropped Image", WINDOW_NORMAL);
					resizeWindow("Cropped Image", 200, 150);
					moveWindow("Cropped Image", 405, 0);
					imshow("Cropped Image", croppedImage);
					imwrite("Cropped_Image.jpg", croppedImage);
				}
				if (Show_Vehicle_Boundaries == true)
				{
					rectangle(image, Point(Vehicle_boundaries.left, Vehicle_boundaries.top), Point(Vehicle_boundaries.right, Vehicle_boundaries.bottom), CV_RGB(255, 255, 255), 4, 8, 0);
					for (int i = 0; i < candids_regions.size(); i++)
						rectangle(image, cvPoint(candids_regions[i][0], candids_regions[i][1]), cvPoint(candids_regions[i][2], candids_regions[i][3]), CV_RGB(255, 255, 255), 2, 8, 0);
					Rect Shot(Vehicle_boundaries.left, Vehicle_boundaries.top, Vehicle_boundaries.right - Vehicle_boundaries.left, Vehicle_boundaries.bottom - Vehicle_boundaries.top);
					Mat ShotImage = image(Shot);
					string fni = "shots\\frame" + to_string(XML.frame_start) +"-lane"+ to_string(XML.lane) + ".jpg";
					imwrite(fni, ShotImage);
					if (Show_Frame_Image == true)
						imshow("Frame", image);
					imwrite("vehicle_late.jpg", image);
				}
			}
			Prev_Frame_NO = XML.frame_start;
			XML = Read_XML_File(inFile);   // Read from XML file

			PrintDetectedVehicles(All_Vehicles, Detected_Vehicles, Undetected_Vehicles);
		}

		if (Show_SEG_Image == true)
		{
			namedWindow("Segment", WINDOW_NORMAL);
			imshow("Segment", segment * 255);
		}
		if (Show_MHI_Image == true)
		{
			imshow("MHI", motion * 255);
			Mat joe = motion * 255;
			imwrite("MHI.JPG", joe);
		}
		////waitKey(0);
		if (waitKey(1) == 27)
			break;
		frame_number++;

	}

	inFile.close();  // Close XML File
	tracking_file.close();
	TrackingInfo();
	///waitKey(0);
}



/*
#ifdef _MSC_VER
#include <ctime>
inline double gets_time()
{
return static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
}
#endif
// Time Strat
if (frame_no >1)
{
double tt = 100000.0;
for (int cc = 0; cc < 10; cc++)
{
	clock_t begin = clock();
	for (int i = 1; i < 41; i++) // n Frame Tracking
	{
		speed = compute_velocity_vector(NewFeatures, i, i - 1);
		_s = metric_calc_velocity_ipm(speed, 2);
		s += _s;
		if (_s != 0)
			tracks++;
	}
	tracking[andis].our_speed_MS = (s / tracks);
	tracking[andis].our_speed_KM = (s / tracks)*(3.6);
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	if (elapsed_secs < tt)
		tt = elapsed_secs;
}
cout << "\n\nTime(ms) : " << tt << "\n\n";
printf("\nElasped time is %.2lf seconds.", tt);
imshow("TIME", image);
waitKey(0);
}
//Time End
*/



/*
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>
#include <Windows.h>

int main()
{
using namespace std::chrono;

high_resolution_clock::time_point t1 = high_resolution_clock::now();

std::cout << "printing out 1000 stars...\n";
for (int i = 0; i<1000; ++i) std::cout << "*";
std::cout << std::endl;

high_resolution_clock::time_point t2 = high_resolution_clock::now();

duration<double, std::milli> time_span = t2 - t1;

std::cout << "It took me " << time_span.count() << " milliseconds.";
std::cout << std::endl;

Sleep(100000);

return 0;
}
*/