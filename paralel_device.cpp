#include "paralel_device.h"
ParalelDevice::ParalelDevice()
{
	kernelFile = defaultKernelFile;
	width = defaultWidth;
	height = defaultHeight;
	init();
}
	
ParalelDevice::ParalelDevice(std::string _kernelFile, int _width, int _height)
{
	kernelFile = _kernelFile;
	width = _width;
	height = _height;
	init();
}
void ParalelDevice::init()
{
	size = width*height;
	std::ifstream file(kernelFile);
	std::string 	programCode(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
	
	
	cl::Program program(programCode);
	err=program.build();
	cl_int buildErr;
	std::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault(), &err);
	std::cout << str;
	defaultDeviceQueue=new cl::CommandQueue(cl::Context::getDefault(), cl::Device::getDefault());
	
	defaultKernel = new cl::Kernel(program, "gradient", &err);
	
	
}

void ParalelDevice::process(cl_uchar* input, cl_uchar* out)
{
	cl::Buffer inBuffer(CL_MEM_READ_WRITE, size);
	cl::enqueueWriteBuffer(inBuffer, CL_TRUE, 0, size, input);
	cl::Event event;
	cl::Buffer outBuffer(CL_MEM_READ_WRITE, size*sizeof(cl_uchar));
	defaultKernel->setArg(0, inBuffer);
	defaultKernel->setArg(1, outBuffer);
	err=defaultDeviceQueue->enqueueNDRangeKernel(*defaultKernel, cl::NullRange, cl::NDRange(640, 480));
	enqueueReadBuffer(outBuffer, CL_TRUE, 0, size*sizeof(cl_uchar), out);
}