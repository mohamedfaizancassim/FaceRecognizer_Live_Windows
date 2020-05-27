#pragma once
#include "dlib/server.h"
#include <iostream>
#include <sstream>
#include <string>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace dlib;

class web_server : public server_http
{
public:
	string content;

	const std::string on_request(const incoming_things& incoming,outgoing_things& outgoing)
	{
		ostringstream sout;
		// We are going to send back a page that contains an HTML form with two text input fields.
		// One field called name.  The HTML form uses the post method but could also use the get
		// method (just change method='post' to method='get').
		
		sout << "<body><h1>Smart CCTV Face Recognizer</h1>";
		sout << " <img src='C:\\Users\\hp\\Pictures\\26th_Birthday\\IMG_4219_SS.jpg' alt='Webcam Feed'> ";
		sout << "</body>";
		sout << "</html>";

		return sout.str();
	}
};

