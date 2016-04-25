
char averagecol(char mid, char topleft, char topmid, char topright, char midleft, char midright, char lowleft, char lowmid, char lowright)
{
	return((mid + topleft + topmid + topright + midleft + midright + lowleft + lowmid + lowright) / 9);
}

__kernel void gradient(__global const char* input, __global char* output, int _range)
{
	int lx=get_local_id(0);
	int ly=get_local_id(1);
	int gx=get_group_id(0);
	int gy=get_group_id(1);
	int lsx=get_local_size(0);
	int lsy=get_local_size(1);
	int y=lsy*gy+ly;
	int x=lsx*gx+lx;
	if (x > 2 && y > 2)
	{
		int offset = y * 640 + x;
		int k = 0;
		for (int i = -1; i < 2; i++)
			for (int j = -1; j < 2; j++)
				//k += (abs(averagecol(input[y*640 + x], input[(y - 1)*640 + x - 1], input[(y - 1)*640 + x], input[(y - 1)*640 + x + 1], input[y*640 + x - 1], input[y*640 + x + 1], input[(y + 1)*640 + x - 1], input[(y + 1)*640 + x], input[(y + 1)*640 + x + 1]) - averagecol(input[(y + i)*640 + x + j], input[(y - 1 + i)*640 + x - 1 + j], input[(y - 1 + i)*640 + x + j], input[(y - 1 + i)*640 + x + 1 + j], input[(y + i) *640 + x - 1 + j], input[(y + i)*640 + x + 1 + j], input[(y + 1 + i)*640 + x - 1 + j], input[(y + 1 + i)*640 + x + j], input[(y + 1 + i)*640 + x + 1 + j])));
				if (abs(averagecol(input[y*640 + x], input[(y - 1)*640 + x - 1], input[(y - 1)*640 + x], input[(y - 1)*640 + x + 1], input[y*640 + x - 1], input[y*640 + x + 1], input[(y + 1)*640 + x - 1], input[(y + 1)*640 + x], input[(y + 1)*640 + x + 1]) - averagecol(input[(y + i)*640 + x + j], input[(y - 1 + i)*640 + x - 1 + j], input[(y - 1 + i)*640 + x + j], input[(y - 1 + i)*640 + x + 1 + j], input[(y + i) *640 + x - 1 + j], input[(y + i)*640 + x + 1 + j], input[(y + 1 + i)*640 + x - 1 + j], input[(y + 1 + i)*640 + x + j], input[(y + 1 + i)*640 + x + 1 + j])) > _range)
					k++;
		output[offset] = k * 28;
	}
}