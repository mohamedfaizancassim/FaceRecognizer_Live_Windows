#include "Dweet.h"



string Dweet::currentDateTime()
{
	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(&now);

	strftime(buf, sizeof(buf), "%Y-%m-%d\t%X", &tstruct);

	return buf;

	return string();
}

void Dweet::SendDweet()
{

	thread thSendDweet([&]()
		{
			stringstream ssFaceName;
			for (auto face : this->vFaceNames)
			{
				ssFaceName << face.first << " , ";
				ssFaceName << " (" << face.second.getXCoord() << "," << face.second.getYCoord() << ") | ";
			}


			//Initialise IO Stream
			iosockstream dweetStream("dweet.io:80");

			//Assemble Get Request
			dweetStream << "GET ";
			dweetStream << "/dweet/for/" << this->dw_thingName << "?";
			dweetStream << "CurrentDateTime=" << this->currentDateTime();
			dweetStream << "&Faces=" << ssFaceName.str();
			dweetStream << " TTP/1.0\r\n\r\n";

			//Print result to console
			while (dweetStream.peek() != EOF)
			{
				cout << (char)dweetStream.get();
			}

			//Clear Vectors and String Stream
			ssFaceName.clear();
			vFaceNames.clear();
		});
}

void Dweet::AddFace(string& faceName, Coordinate& coord)
{
	vFaceNames.push_back(pair<string, Coordinate>(faceName, coord));
}

Dweet::~Dweet()
{
}
