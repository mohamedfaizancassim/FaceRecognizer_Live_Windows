#pragma once
#include<iostream>

using namespace std;
/*
A simple class to store coordinate points;
*/
class Coordinate
{
private:
	int x_coord{ 0 };
	int y_coord{ 0 };

public:
	Coordinate(int x, int y) :x_coord(x), y_coord(y) {};
	int getXCoord()
	{
		return x_coord;
	}
	int getYCoord()
	{
		return y_coord;
	}
};