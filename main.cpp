#include "paralel_device.h"
#include <opencv2/opencv.hpp>

using namespace cv;

void main()
{	
	VideoCapture cap(0);
	Mat input, output;
	output = Mat(Size(640, 480), CV_8UC1);
	ParalelDevice dev;
	while (1)
	{
		cap >> input;
		cvtColor(input, input, CV_BGR2GRAY);
		dev.process(input.data, output.data);
		//imshow("image1", input);
		imshow("image", output);
		waitKey(33);
	}
}