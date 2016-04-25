#include "paralel_device.h"
#include <opencv2/opencv.hpp>

using namespace cv;

void main()
{	
	VideoCapture cap(0);
	Mat input, output;
	output = Mat(Size(640, 480), CV_8UC1);
	ParalelDevice dev;
	int range = 1;
	while (1)
	{
		cap >> input;
		cvtColor(input, input, CV_BGR2GRAY);
		dev.process(input.data, output.data, range);
		//imshow("image1", input);
		imshow("image", output);
		if (waitKey(32) == 2490368) range++;
		if (waitKey(32) == 2621440) range--;
		
		waitKey(33);
	}
}