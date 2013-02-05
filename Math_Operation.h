#ifndef MATH_OPERATION
#define MATH_OPERATION

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <fstream>
#include <cmath>
#include <set>

const double PI = acos(-1.0);

////////////////////////////////////////////////////////////////////////////
//函数名：Gaussian
//作用  ： 根据参数生成一个3*3的Gauss卷积矩阵
//参数  ：
////////////////////////////////////////////////////////////////////////////
CvMat * Gaussian(double tou)
{
	CvMat * out = cvCreateMat(3,3,CV_64FC1);
	int cx=1,cy=1;
	double total=0.0;
	for (int x=-1;x<=1;++x)
		for (int y=-1;y<=1;++y)
		{
			double k=1.0/2.0/PI/tou/tou;
			double e=exp( - (x*x+y*y) / 2/ tou /tou);
			cvmSet(out,x+cx,y+cy,k*e);
			total+=k*e;
		}
	for (int i=0;i<3;++i)
		for (int j=0;j<3;++j)
		{
			double t = cvmGet(out,i,j);
			t/=total;
			cvmSet(out,i,j,t);
		}
	return out;
}

////////////////////////////////////////////////////////////////////////////
//函数名：Convolution
//作用  ： 将一个3*3的卷积con作用在矩阵in上,结果返回,不改变in的值
////////////////////////////////////////////////////////////////////////////
CvMat * Convolution(CvMat * in , CvMat * con)
{
	CvMat * out = cvCreateMat(in->height,in->width,CV_64FC1);
	for (int i=0;i<in->height;++i)
		for (int j=0;j<in->width;++j) cvmSet(out,i,j,0.0);

	for (int i=1;i<in->height-1;++i)
		for (int j=1;j<in->width-1;++j)
		{
			double now=0.0;
			for (int x=0;x<3;++x)
				for (int y=0;y<3;++y)
					now+= cvmGet(con,x,y) * cvmGet(in,i+x-1,j+y-1);
			cvmSet(out,i,j,now);
		}	

	return out;
}

////////////////////////////////////////////////////////////////////////////
//函数名：Convolution
//作用  ： 将一个5*5的卷积con作用在矩阵in的坐标(x,y)位置上
//             结果直接修改在输入矩阵中
////////////////////////////////////////////////////////////////////////////
void ConvolutionOnSinglePoint(CvMat * & in , CvMat * con,int x,int y)
{
	if (x-2<0 || x+2 >= in->height || y-2<0 || y+2>=in->width) return;

	double value=0.0;
	for (int i=0;i<5;++i)
		for (int j=0;j<5;++j)
			value += cvmGet(in,x+i-2,y+j-2) * cvmGet(con,i,j);
	cvmSet(in,x,y,value);
}

template <class T>
double normal2(vector<T> & f)
{
	double ret = 0.0;
	for (int i=0;i<f.size();++i)
		ret+= f[i] * f[i];
	ret = sqrt(ret);
	return ret;
}

template <class T>
double dot(vector<T> & f1 ,vector<T> & f2)
{
	double ret = 0.0;
	for (int i=0;i<f1.size();++i)
		ret += f1[i] * f2[i];
	return ret;
}

template <class T>
double cosSim(vector<T> & f1,vector<T> & f2)
{
	double n1 = normal2(f1);
	double n2 = normal2(f2);
	if (fabs(n1)<1e-6 || fabs(n2)<1e-6) return 1e20;
	return dot(f1,f2) / n1 / n2;
}

template <class T>
vector<T> operator+(vector<T> & f1 , vector<T> & f2)
{
	vector<T> ret(f1.size());
	for (int i = 0; i < f1.size(); ++i)
		ret[i] = f1[i] + f2[i];
	return ret;
}


#endif

