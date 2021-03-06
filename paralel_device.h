#include <fstream>
#include <iostream>
#include <CL/cl.hpp>


class ParalelDevice
{
	int devId;
	int width, height, size, defaultHeight = 480, defaultWidth = 640;

	std::string kernelFile, defaultKernelFile="kernels.cl";
	cl::Kernel *defaultKernel;
	cl::CommandQueue *defaultDeviceQueue;


	int err;	
	
	void init();
public:
	ParalelDevice();
	ParalelDevice(std::string _kernelFile, int _width, int _height);
	void process(cl_uchar* inLeft, cl_uchar* out, int range);
};