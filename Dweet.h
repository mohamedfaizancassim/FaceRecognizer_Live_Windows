#pragma once
#include <iostream>
#include<sstream>
#include <vector>
#include <dlib/iosockstream.h>
#include "Coordinate.h"
#include <thread>
#include <ctime>
using namespace std;
using namespace dlib;
class Dweet
{
private:
	string dw_thingName{};
	std::vector<pair<string,Coordinate>> vFaceNames;
	string currentDateTime();
public:
	Dweet(string thingName) :dw_thingName(thingName) {};
	void SendDweet();
	void AddFace(string &faceName,Coordinate &coord);
	~Dweet();
};

