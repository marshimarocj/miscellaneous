#ifndef MAT_OPERATION
#define MAT_OPERATION

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <fstream>
using namespace std;
#include "Image_Operation.h"



////////////////////////////////////////////////////////////////////////////
//函数名：ShowHistogram
//作用  ： 基于给定的直方图,输出他的图像显示^_^
//参数  ：
//char * display          输出图像的文件名 
//显示   :   统计矩阵中元素的范围,归一至0-255的灰度图像
////////////////////////////////////////////////////////////////////////////
void ShowHistogram(double * data,int datas, char * display)
{
	IplImage * img = cvCreateImage(cvSize(320,240),IPL_DEPTH_8U,3);
	
	
	for (int i=0;i<img->height;++i)
		for (int j=0;j<img->width;++j)
		{
			setPix(i,j,2,img)=255;
			setPix(i,j,1,img)=255;
			setPix(i,j,0,img)=255;
		}

	for (int i=0;i<datas;++i)
	{
		int width = (int)(320.0 / (double)(datas+1) * (double)i);
		int height = (int)(240.0 * data[i] / 1.0);
		cvLine(  img, cvPoint(width,240-height), 
		cvPoint (width,240), CV_RGB(255,0,0));
	}

	cvSaveImage(display,img);
	//cout<<min<<" "<<max<<endl;
}

////////////////////////////////////////////////////////////////////////////
//函数名：ShowHistogram
//作用  ： 基于给定的一维直方图,返回他的图像显示^_^
//参数  ： hist_size 一维直方图的区域数量
////////////////////////////////////////////////////////////////////////////
IplImage * ShowHistogram(CvHistogram * hist,int hist_size)
{
	/** 直方图归一化 */
	cvNormalizeHist(hist,1.0);
	/** 创建直方图的显示图像 */
	IplImage * histimg = cvCreateImage( cvSize(320,240), 8, 3 );
	cvZero( histimg );
	int bin_w = histimg->width / hist_size;
	for(int i = 0; i < hist_size; i++ )
	{
		int val = cvRound( cvGetReal1D(hist->bins,i)*histimg->height);
		CvScalar color = hsv2rgb(i*180.f/hist_size);
		cvRectangle( histimg, cvPoint(i*bin_w,histimg->height),
			cvPoint((i+1)*bin_w,histimg->height - val),
			color, -1, 8, 0 );
	}
	return histimg;
}

IplImage * ShowHistogram(double * hist,int hist_size)
{
	/** 创建直方图的显示图像 */
	IplImage * histimg = cvCreateImage( cvSize(320,240), 8, 3 );
	cvZero( histimg );
	int bin_w = histimg->width / hist_size;
	for(int i = 0; i < hist_size; i++ )
	{
		int val = cvRound(hist[i]*histimg->height);
		CvScalar color = hsv2rgb(i*180.f/hist_size);
		cvRectangle( histimg, cvPoint(i*bin_w,histimg->height),
			cvPoint((i+1)*bin_w,histimg->height - val),
			color, -1, 8, 0 );
	}
	return histimg;
}

////////////////////////////////////////////////////////////////////////////
//函数名：ShowMat
//作用  ： 基于给定的矩阵,输出他的图像显示^_^
//参数  ：
//char * display          输出图像的文件名 
//显示   :   统计矩阵中元素的范围,归一至0-255的灰度图像
////////////////////////////////////////////////////////////////////////////
IplImage * ShowMat(CvMat * in)
{
	double max = -1e30;
    double min = 1e30;
	for (int i = 0; i < in->height; ++i)
		for (int j = 0; j < in->width; ++j)
		{
			double t = cvmGet(in,i,j);
			if (t > max) max = t;
			if (t < min) min = t;
		}

	double alpha= 255 / (max-min);
	IplImage * img = cvCreateImage(cvSize(in->width,in->height), 8, 3);
	
	for (int i = 0; i < in->height; ++i)
		for (int j = 0; j < in->width; ++j)
		{
			int output = (int) ((cvmGet(in,i,j) - min) * alpha);
            cvSet2D(img, i, j, cvScalar(output, output, output, 0));
		}
	return img;
}


void ShowMat(CvMat * in, char * display)
{
    IplImage * img = ShowMat(in);
	cvSaveImage(display,img);
    cvReleaseImage(&img);
}

IplImage * getMatImage(CvMat * in)
{
	double max=-1e50;double min=1e50;double t;
	for (int i=0;i<in->height;++i)
		for (int j=0;j<in->width;++j)
		{
			t=cvmGet(in,i,j);
			if (t>max) max=t;
			if (t<min) min=t;
		}

	double alpha= 255 / (max-min);

	IplImage * img = cvCreateImage(cvSize(in->width,in->height),IPL_DEPTH_8U,3);
	
	double output;
	for (int i=0;i<in->height;++i)
	{
		for (int j=0;j<in->width;++j)
		{
			output=(cvmGet(in,i,j)-min) * alpha;
			setPix(i,j,2,img)=(int) output;
			setPix(i,j,1,img)=(int) output;
			setPix(i,j,0,img)=(int) output;
		}
	}
	return img;
}

////////////////////////////////////////////////////////////////////////////
//函数名：ShowMat2
//作用  ： 基于给定的矩阵,输出他的图像显示^_^
//参数  ：
//char * display          输出图像的文件名 
//显示   :   离散的显示矩阵的图像信息
////////////////////////////////////////////////////////////////////////////
void ShowMat2(CvMat * in, char * display)
{
	IplImage * img = cvCreateImage(cvSize(in->width,in->height),IPL_DEPTH_8U,3);

	int color[10][3];
	color[0][0]=255;color[0][1]=0;color[0][2]=0;
	color[1][0]=0;color[1][1]=255;color[1][2]=0;
	color[2][0]=0;color[2][1]=0;color[2][2]=255;
	color[3][0]=255;color[3][1]=255;color[3][2]=0;
	color[4][0]=0;color[4][1]=255;color[4][2]=255;

	int output;
	for (int i=0;i<in->height;++i)
		for (int j=0;j<in->width;++j)
		{
			output=(int) cvmGet(in,i,j);
			output %=5;
			setPix(i,j,2,img)=color[output][0];
			setPix(i,j,1,img)=color[output][1];
			setPix(i,j,0,img)=color[output][2];
		}

	cvSaveImage(display,img);
}

////////////////////////////////////////////////////////////////////////////
//函数名：ShowMat
//作用  ： 基于给定的矩阵,输出其值为tar的分布图像
//参数  ：
//char * display          输出图像的文件名 
//显示   :   离散的显示矩阵的图像信息
////////////////////////////////////////////////////////////////////////////
void ShowMat(CvMat * in, char * display,int tar,IplImage * ori=NULL)
{
	IplImage * img = cvCreateImage(cvSize(in->width,in->height),IPL_DEPTH_8U,3);

	int output;
	for (int i=0;i<in->height;++i)
		for (int j=0;j<in->width;++j)
		{
			output=(int) cvmGet(in,i,j);
			if (output == tar) 
			{
				if (ori!=NULL)
				{
					setPix(i,j,2,img)=getPix(i,j,2,ori);
					setPix(i,j,1,img)=getPix(i,j,1,ori);
					setPix(i,j,0,img)=getPix(i,j,0,ori);
				}
				else
				{
					setPix(i,j,2,img)=0;
					setPix(i,j,1,img)=0;
					setPix(i,j,0,img)=0;
				}
			}
			else
			{
				setPix(i,j,2,img)=255;
				setPix(i,j,1,img)=255;
				setPix(i,j,0,img)=255;
			}
		}
	cvSaveImage(display,img);
}


////////////////////////////////////////////////////////////////////////////
//函数名：ShowMat
//作用  ： 基于给定的三维数组,输出Channel信道上的图像
//参数  ：
//char * display          输出图像的文件名 
//显示   :   离散的显示矩阵的图像信息
////////////////////////////////////////////////////////////////////////////
void ShowMat(double *** in,int height,int width, char * display,int Channel)
{
	IplImage * img = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,3);

	CvMat * mat = cvCreateMat(height,width,CV_64FC1);
	for (int i=0;i<height;++i)
		for (int j=0;j<width;++j)
			cvmSet(mat,i,j,in[i][j][Channel]);

	ShowMat(mat,display);
	cvReleaseMat(&mat);
}




////////////////////////////////////////////////////////////////////////////
//函数名：Normalize()
//作用  ： 对矩阵统计元素的最大值最小值,从而将每个元素正规化至0-1
////////////////////////////////////////////////////////////////////////////
void Normalize(CvMat * in)
{
	double min,max;
	min=1e20;
	max=-1e20;

	for (int i=0;i<in->height;++i)
		for (int j=0;j<in->width;++j)
		{
			double t=cvmGet(in,i,j);
			if (t>max) max=t;
			if (t<min) min=t;
		}
	double range=max-min;
		
	for (int i=0;i<in->height;++i)
		for (int j=0;j<in->width;++j)
			cvmSet(in,i,j , (cvmGet(in,i,j) - min)/range);
}

////////////////////////////////////////////////////////////////////////////
//函数名：PrintMat
//作用  ： 打印一个矩阵的元素信息
////////////////////////////////////////////////////////////////////////////
void PrintMat(CvMat * in)
{
	for (int i=0;i<in->height;++i)
	{
		for (int j=0;j<in->width;++j) printf("%lf ",cvmGet(in,i,j));
		printf("\n");
	}
}



////////////////////////////////////////////////////////////////////////////
//函数名：Mat_Minus
//作用  ： 将两个输入矩阵做差,结果返回
////////////////////////////////////////////////////////////////////////////
CvMat * Mat_Minus(CvMat * a, CvMat * b)
{
	if (a->height != b->height || a->width != b->width) return NULL;
	CvMat * out = cvCreateMat(a->height,b->width,CV_64FC1);

	for (int i=0;i<a->height;++i)
		for (int j=0;j<a->width;++j)
		{
			double t= cvmGet(a,i,j) - cvmGet(b,i,j);
			cvmSet(out,i,j,t);
		}
	return out;
}


////////////////////////////////////////////////////////////////////////////
//函数名：Local_Max_Extrema_Detection
//作用  ： 实现 Distinctive Image Features from Scale-Invariant Keypoints 
//             中的一个函数, 给定三个矩阵,检测矩阵Detect中的每个元素是否是相邻
//             26个元素中的极大值,在返回矩阵中体现
////////////////////////////////////////////////////////////////////////////
CvMat * Local_Max_Extrema_Detection(CvMat * detect, CvMat * up,CvMat * down)
{
	CvMat * out = cvCreateMat(detect->height,detect->width,CV_64FC1);
	for (int i=0;i<out->height;++i)
		for (int j=0;j<out->width;++j)
			cvmSet(out,i,j,0.0);

	for (int i=1;i<out->height-1;++i)
		for (int j=1;j<out->width-1;++j)
		{
			double now=cvmGet(detect,i,j);
			bool ok=true;
			for (int x=-1;x<=1;++x)
				for (int y=-1;y<=1;++y) 
				{
					if (cvmGet(detect,i+x,j+y)>now) ok=false;
					if (cvmGet(up,i+x,j+y)>now) ok=false;
					if (cvmGet(down,i+x,j+y)>now) ok=false;
				}
			if (ok) cvmSet(out,i,j,1.0);
		}
	return out;
}



///////////////////////////////////////////////////////////////////////
// ComputeRange 
///////////////////////////////////////////////////////////////////////
void ComputeRange(CvMat * in , double * target , int targets ,int & minx,int & maxx,int & miny, int & maxy)
{
	minx = 9999;
	miny = 9999;
	maxx = 0;
	maxy = 0;

	for (int i=0;i<in->height;++i)
		for (int j=0;j<in->width;++j)
		{
			for (int k=0;k<targets;++k)
				if (cvmGet(in,i,j) == target[k])
				{
					if (i<minx) minx = i;
					if (i>maxx) maxx = i;
					if (j<miny) miny = j;
					if (j>maxy) maxy = j;
				}
		}
	if (minx==9999 && miny==9999 && maxx==0 && maxy==0)
	{ minx=miny=maxx=maxy=0; }
	return ;
}

#endif
