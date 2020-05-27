
#define DDLIB_JPEG_SUPPORT
#include "trainer.h"
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include<algorithm>
#include <dlib/image_processing/frontal_face_detector.h>


using namespace dlib;
using namespace std;

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

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2,
	dlib::input_rgb_image_sized<150>
	>>>>>>>>>>>>;
//-------------------------------------------------------------------------

dlib::matrix<float, 0, 1> trainer::Face_Descriptor(string path)
{
	std::vector<dlib::matrix<float, 0, 1>> face_descriptor;

	//Frontal Face Detectpr
	dlib::frontal_face_detector f_detector = dlib::get_frontal_face_detector();

	//Shape Predictor
	dlib::shape_predictor s_predictor;
	dlib::deserialize("models/shape_predictor_5_face_landmarks.dat") >> s_predictor;

	// DNN Neural Network
	anet_type d_net;
	dlib::deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> d_net;

	//Dlib matrix to store image
	dlib::matrix<dlib::rgb_pixel> face_image;

	// (1) Get ROI.
	// (2) Crop ROI.
	// (3) Get Face Descriptor.
	// (4) Return Face Descriptor.

	//----------------------------------------------------------------

	//Loading image to face_image
	dlib::load_jpeg(face_image, path);

	std::vector<matrix<rgb_pixel>> faces;

	dlib::resize_image(0.75, face_image);

	//Iterating through detected faces;
	for (auto face : f_detector(face_image))
	{
		if (face.is_empty())
		{
			return dlib::matrix<float, 0, 1>();
		}
		std::cout << "Face Found! " << "Path: \t" << path<< endl;

		//Dlib matrix to croped faces;
		dlib::matrix<dlib::rgb_pixel> croped_face;

		//Apply shape predictor
		auto shape = s_predictor(face_image, face);

		//Extract cropped face
		dlib::extract_image_chip(face_image, dlib::get_face_chip_details(shape, 150, 0.25), croped_face);

		faces.push_back(croped_face);

	}

	if (faces.size() > 0)
	{
		//Get Face Discriptor for Croped Face
		face_descriptor = d_net(faces);
	}

	//----------------------------------------------------------------
	
	
	return (face_descriptor.size() > 0) ? face_descriptor[0] : dlib::matrix<float, 0, 1>();
}

std::vector<pair<string, dlib::matrix<float, 0, 1>>> trainer::Get_Face_Descriptors()
{
	//A map to store face_descriptors with their respective labels
	std::vector<pair< string, dlib::matrix<float, 0, 1>>> label_fDescriptor;

	//Initialise recursive directory iterators
	recursive_directory_iterator dir_itr(this->rootFolderPath);
	recursive_directory_iterator dir_end;

	//Create a string vector to store directories
	std::vector<string> directories;


	//Get Available Directories
	while (dir_itr != dir_end)
	{
		error_code ec;
		if (is_directory(dir_itr->path()))
		{
			//dir_itr.disable_recursion_pending();
		}
		else
		{
			std::cout << dir_itr->path().string() << endl;
			directories.push_back(dir_itr->path().string());
		}

		dir_itr.increment(ec);

		if (ec)
		{
			std::cout << "Error acessing directory: " << ec.message() << endl;
		}

	}

	std::cout << "Traning data consists of " << directories.size() << " images." << endl;

	// (1) Extract "Image Lable" and "Image Path" from directories vector
	// (2) Process each image and give a Face Descriptor
	// (3) Add to <label,face descriptor> map 
	// (4) return map. 
	
	parallel_for_(Range(0, directories.size()), [&](const Range & range)
		{
			for (int r = range.start; r < range.end; r++)
			{
				std::vector<string> tokenized_data;
				stringstream ss(directories[r]);
				string token;

				while (getline(ss, token, '\\'))
				{

					if (token != "known_faces")
					{
						tokenized_data.push_back(token);
					}

				}

				string name = tokenized_data[0];
				string i_path = directories[r];

				//***************************************
				//	Process Image here
				//***************************************
				
			
				auto f_descriptor = Face_Descriptor(i_path);

				if (f_descriptor != dlib::matrix<float, 0, 1>())
				{
					
					label_fDescriptor.push_back(pair<string,dlib::matrix<float, 0, 1>>(name, f_descriptor));
					
					std::cout << "Face Name:\t" << name << endl;
				}
				
				//--------------------------------------
				tokenized_data.clear();
			}
		
	});
	

	std::cout << "Total images trained: " << label_fDescriptor.size() << endl;

	return label_fDescriptor;
}

string trainer::Get_Face_Label(std::vector<pair<string, dlib::matrix<float, 0, 1>>>& training_data, dlib::matrix<float, 0, 1> face_descriptor)
{
	std::vector <float>differences;

	//Finding the difference between Current Face Descriptor
	// and all trained face descriptors;
	for (int i = 0; i < training_data.size(); i++)
	{
		auto fDescriptor_diff = length(training_data[i].second - face_descriptor);
		differences.push_back(fDescriptor_diff);
	}
	
	//Finding the minimum difference;
	auto min_diff = min_element(begin(differences), end(differences));

	if (*min_diff < 0.6)
	{
		//Initialising an iterator to find the index of the minimum value element
		std::vector<float>::iterator it = std::find(differences.begin(), differences.end(), *min_diff);
		//Geting the index
		int index = std::distance(differences.begin(), it);

		return training_data[index].first;
	}
	else
	{
		return "Unrecognized_Person";
	}
}

trainer::~trainer()
{
}
