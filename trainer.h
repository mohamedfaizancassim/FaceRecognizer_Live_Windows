#pragma once


#include<string>
#include <iostream>
#include <vector>
#include <filesystem>
#include<sstream>
#include<map>
#include<opencv2/opencv.hpp>
#include<dlib/matrix.h>
using namespace std;
using namespace experimental::filesystem;
using namespace cv;


class trainer
{
private:
	string rootFolderPath{ NULL };
	
public:
	trainer(string folderPath) :rootFolderPath(folderPath) {};
	dlib::matrix<float, 0, 1> Face_Descriptor(string path);
	std::vector<pair<string,dlib::matrix<float, 0, 1>>> Get_Face_Descriptors();
	string Get_Face_Label(std::vector<pair<string, dlib::matrix<float, 0, 1>>> &training, dlib::matrix<float, 0, 1> face_descriptor);
	~trainer();
};

