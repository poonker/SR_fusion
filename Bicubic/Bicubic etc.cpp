#include<iostream>
#include<opencv2/opencv.hpp>

void main()
{
	using namespace cv;
	Mat img;
	img=imread("IN PATH");
	int nCols = 1600;
	int nRows = 1200;
	Mat img2(nRows, nCols, img.type());
	resize(img, img2, img2.size(), 0, 0, INTER_CUBIC);
	imwrite("OUT  PATH",img2);
	waitKey(0);
}