
__kernel void gradient(__global const char* input, __global char* output)
{
	int y = get_global_id(1);
	int x = get_global_id(0);
	int offset=y*640+x;
	for (int i = 0; i < 10; i++){
		output[i] = 255;
	}
	
}