// CameraFaceRecognizer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include <dlib/opencv/cv_image.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/iosockstream.h>
#include <sstream>
#include "trainer.h"
#include "web_server.h"
#include "Dweet.h"
#include "Coordinate.h"
#include<chrono>
#include<future>
#include<concurrent_vector.h>
#include<thread>
#include <ctime>
using namespace dlib;
using namespace std;
using namespace cv;
using namespace concurrency;
//-------------------------------------------
//	Neural Network Stuff
//-------------------------------------------------------------------------
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;
//-------------------------------------------------------------------------

//---------------------------------------------
//	Image Jitter
//-------------------------------------------------------------------------
std::vector<matrix<rgb_pixel>> jitter_image(
	const matrix<rgb_pixel>& img
);
//-------------------------------------------------------------------------
//---------------------------------------------
//	Current Date and Time
//-------------------------------------------------------------------------
const std::string currentDateTime() {
	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(&now);
	
	strftime(buf, sizeof(buf), "%Y-%m-%d\t%X", &tstruct);

	return buf;
}
//-------------------------------------------------------------------------

int main()
{
    //Initialize and open video capture device;
	//VideoCapture vid_cap("rtsp://192.168.1.12:554/11");
	VideoCapture vid_cap(0,CAP_DSHOW);

	////Creating and initializing new webserver
	//web_server webServer;
	//webServer.set_listening_port(80);
	//webServer.start_async();
	

	if (!vid_cap.isOpened())
	{
		cout << "Video capture device could not be initialised." << endl;
		cout << "Please check if the devices is in use or working properly." << endl;
		return -1;
	}

	//Facedetector
	frontal_face_detector f_detector = get_frontal_face_detector();

	//Shape Predictor
	shape_predictor sp;
	deserialize("models/shape_predictor_5_face_landmarks.dat")>>sp;

	//Deep Neural Network
	anet_type net;
	deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> net;

	//Create a new window object
	//image_window window;

	//-----------------------------------------------
	trainer _trainer("known_faces");
	auto face_desc = _trainer.Get_Face_Descriptors();
	//-----------------------------------------------
	while (true)
	{
		//Initialise a matrix object to store the frame;
		Mat frame;
		//Sore frame from camera in frame object
		vid_cap >> frame;

		//Convert frame to a DLIB frame
		dlib::cv_image<rgb_pixel> dlib_frame(frame);
		
		//Store copies of Face Image in Vector
		concurrent_vector<matrix<rgb_pixel>> faces;

		//Store cordinates of respective face_image
		concurrent_vector<pair<int,int>> xy_coords;
		concurrent_vector<pair<int, int>>width_heigt;



		auto detections = f_detector(dlib_frame);
		
		
		dlib::parallel_for(0, detections.size(), [&, detections](long i)
			{
				auto det_face = detections[i];
				
				////Get the Shape details from the face
				packaged_task < dlib::full_object_detection()>sd_task(bind(sp,dlib_frame,det_face));
				auto sd_task_res = sd_task.get_future();
				sd_task();
				//auto shape = ;

				//Extract Face Image from frame
				matrix<rgb_pixel> face_img;
				extract_image_chip(dlib_frame, get_face_chip_details(sd_task_res.get(), 150, 0.25), face_img);
				faces.push_back(face_img);
				//Add the coordinates to the coordinates vector
				xy_coords.push_back(std::pair<int, int>((int)det_face.left(), (int)det_face.bottom()));
				width_heigt.push_back(pair<int, int>((int)det_face.width(), (int)det_face.height()));
				
			});

		
		//Creating a vector to store face descriptors
		std::vector<matrix<float, 0, 1>> face_descriptors;

		//Create a Dweet object for dweeting
		Dweet myDweet("MFCassim_LiveCamFaceRecognition");

		//Only run if faces are found
		if (faces.size() != 0)
		{
			face_descriptors = net(faces);

			
			//Print Face Descriptors to Screen
			dlib::parallel_for(0, face_descriptors.size(), [&](long i) 
			{
				
					auto descriptor = face_descriptors[i];

					string fName = _trainer.Get_Face_Label(face_desc, descriptor);
					//dw_det_faceName.append(fName + ",");
					int x_coord = xy_coords[i].first;
					int y_coord = xy_coords[i].second;

					putText(frame, //target image
						fName, //text
						cv::Point(x_coord, y_coord+ 25), //top-left position
						cv::FONT_HERSHEY_DUPLEX,
						1.0,
						CV_RGB(0, 255, 0), //font color
						2);
					Point p1(x_coord, y_coord);
					Point p2(x_coord + width_heigt[i].first, y_coord - width_heigt[i].second);
					cv::rectangle(frame, p1, p2, cv::Scalar(0, 255, 0));

					cout <<"Time Stamp: "<<currentDateTime()<< "\tFace Name: " << fName << "\tx_coord: " << x_coord << "\ty_coord: " << y_coord << endl;

					//Add Face to Dweet
					Coordinate coord(x_coord, y_coord);
					//myDweet.AddFace(fName,coord);
				
			});
			
			//Send Dweet to Dweet Server
			//myDweet.SendDweet();
		}
		
		//Show Frame in OpenCv Window
		cv::imshow("Face Recognition", frame);
		cv::waitKey(2);
		frame.release();
	}

	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
