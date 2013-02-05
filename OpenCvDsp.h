//#define DSP
//#define MYOPENCV
//#define MAC

#ifndef DSP
#ifndef MAC
#include "cv.h"
#include "highgui.h"
#include "cxcore.h"
#endif
#ifdef MAC
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#endif
#endif

#ifndef OPENCV_DSP_H
#define OPENCV_DSP_H

#include <iostream>
using namespace std;

#ifdef DSP

const int CV_BGR2GRAY = 1;
const int CV_BGR2Lab = 2;
//const int IPL_DEPTH_32F = 32;
const int CV_32FC1 = 32;

///////////////////////
// BASIC STRUCTURE
///////////////////////
struct CvScalar
{
	double val[4];
};

CvScalar cvScalar(double a, double b, double c)
{ 
	CvScalar ret;
	ret.val[0] = a;
	ret.val[1] = b;
	ret.val[2] = c;
	return ret;
}

struct CvPoint
{
	int x, y;
};

CvPoint cvPoint(int x, int y) 
{
	CvPoint ret;
	ret.x = x;
	ret.y = y;
	return ret;
}

struct CvRect
{
	int x, y;
	int height, width;
};

CvRect cvRect(int x, int y, int w, int h)  
{
	CvRect ret; ret.x = x; ret.y = y; ret.height = h; ret.width = w;
	return ret;
}

struct CvSize
{
	int width;
	int height;
	CvSize(int w, int h)
	{ width = w; height = h; }
};
CvSize cvSize(int w, int h) { return CvSize(w, h); }


///////////////////////////////////////
// CVMAT
///////////////////////////////////////
struct CvMat
{
	int type;
    int step;

    /* for internal use only */
    int* refcount;
    int hdr_refcount;

    union
    {
        unsigned char * ptr;
        short* s;
        int* i;
        float* fl;
        double* db;
    } data;

    union
    {
        int rows;
        int height;
    };

    union
    {
        int cols;
        int width;
    };


	CvMat(int h, int w)
	{
		data.fl = new float[h * w];
		height = h;
		width = w;
		rows = h;
		cols = w;
	}

	~CvMat()
	{
		delete [] data.fl;
	}
};

CvMat * cvCreateMat(int h, int w, int depth)
{
	CvMat * mat = new CvMat(h, w);
	return mat;
}

void cvReleaseMat(CvMat ** mat) { delete *mat; }

inline double cvmGet(CvMat * mat, int x, int y) { return mat->data.fl[x * mat->width + y]; }
inline void cvmSet(CvMat * mat, int x, int y, double value) { mat->data.fl[x * mat->width + y] = value; }
void cvZero(CvMat * mat) { for (int i = 0; i < mat->height * mat->width; ++i) mat->data.fl[i] = 0; }

/////////////////////////////
// IPLIMAGE
/////////////////////////////
struct IplImage
{
	unsigned char * imageData;
	int height;
	int width;
	int widthStep;
	int nChannels;
	int depth;
	IplImage (CvSize size, int depth, int channels)
	{
		height = size.height;
		width = size.width;
		nChannels = channels;
		widthStep = channels * width;
		imageData = new unsigned char[height * width * channels];
	}

	~IplImage()
	{
		delete [] imageData;
	}
};

void cvZero(IplImage * img) { for (int i = 0; i < img->height * img->width * img->nChannels; ++i) img->imageData[i] = 0; }
void cvSet2D(IplImage * img, int i, int j, CvScalar s) {
	int step = img->widthStep;
	img->imageData[i * step + j * 3 + 0] = s.val[0];
	img->imageData[i * step + j * 3 + 1] = s.val[1];
	img->imageData[i * step + j * 3 + 2] = s.val[2];
}

IplImage * cvCreateImage(CvSize size, int depth, int channels)
{
	IplImage * img = new IplImage(size, depth, channels);
	return img;
}

void cvReleaseImage(IplImage ** img) { delete *img; }
IplImage * cvLoadImage(const char * data) { IplImage * img; return img; }
void cvSaveImage(const char * data, IplImage * img) {}
CvSize cvGetSize(IplImage * img) { return cvSize(img->width, img->height); }
void cvWaitKey(int x) {}
void cvLine(IplImage * img, CvPoint p1, CvPoint p2, CvScalar color, int x, int y, int z) {}
//////////////////////////
// ADVANCED
//////////////////////////
void cvShowImage(char * name, IplImage * srcImage) {}

CvScalar CV_RGB(int r, int g, int b) { return cvScalar(r, g, b); }

void cvRectangle(IplImage * img, CvRect rect, CvScalar color, int w)
{
	return;
}

#endif

#ifdef MYOPENCV
void cvResize(IplImage * srcImage, IplImage * dstImage)
{
	double srcHeight = srcImage->height;
	double srcWidth = srcImage->width;
	double dstHeight = dstImage->height;
	double dstWidth = dstImage->width;

	double revHeightScale = srcHeight / dstHeight;
	double revWidthScale = srcWidth / dstWidth;

	unsigned char * srcData = (unsigned char *) srcImage->imageData;
	unsigned char * dstData = (unsigned char *) dstImage->imageData;
	int srcStep= srcImage->widthStep;
	int dstStep = dstImage->widthStep;
	int nC = srcImage->nChannels;
	for (int k = 0; k < nC; ++k)
		for (int i = 0; i < dstImage->height; ++i)
			for (int j = 0; j < dstImage->width; ++j)
			{
				int value = 0;
				double u = (double) i * revHeightScale;
				double v = (double) j * revWidthScale;

				double x = u - floor(u);
				double y = v - floor(v);
				double minDis = 1e10;
				int x1 = (int)u;
				int x2 = (int) ceil(u); x2 = min(x2, srcImage->height);
				int y1 = (int)v;
				int y2 = (int) ceil(v); y2 = min(y2, srcImage->width);
				int f00 = srcData[x1 * srcStep + y1 * nC + k];
				int f01 = srcData[x1 * srcStep + y2 * nC + k];
				int f10 = srcData[x2 * srcStep + y1 * nC + k];
				int f11 = srcData[x2 * srcStep + y2 * nC + k];
				value = f00 * (1 - x) * (1 - y) + f10 * x * (1 - y) + f01 * (1 -  x) * y + f11 * x * y;

				dstData[i * dstStep + j * nC + k] = value;
			}
}


void convolution(IplImage * src, CvMat * dst, CvMat * mask)
{
	int height = src->height;
	int width = src->width;
	int step = src->widthStep;
	unsigned char * srcData = (unsigned char * ) src->imageData;

	int r = mask->rows;
	int c = mask->cols;
	if (r % 2 == 1)
	{
		for (int i = 0; i < height; ++i)
			for (int j = 0; j < width; ++j)
			{
				double ele = 0;
				for (int x = max(0, i-r/2); x <= min(height-1, i+r/2); x++)
					for (int y = max(0, j-c/2); y <= min(width-1, j+c/2); y++)
						ele += srcData[x*step+y] * cvmGet(mask, x-i+r/2, y-j+c/2);
				cvmSet(dst, i, j, fabs(ele));
			}
	} 
	else
	{
		for(int i = 0; i < height-r; ++i)
			for(int j = 0; j < width-c; ++j)
			{
				double ele = 0;
				for(int x = i; x < i+r; x++)
					for(int y = j; y < j+c; y++)
						ele += srcData[x*step+y] * cvmGet(mask, x-i, y-j);
				cvmSet(dst, i, j, fabs(ele) );
			}
	}
}

CvMat * laplacian = NULL;
CvMat * robert45 = NULL;
CvMat * robert135 = NULL;
CvMat * robert = NULL;
CvMat * sobelX = NULL;
CvMat * sobelY = NULL;
CvMat * prewittX = NULL;
CvMat * prewittY = NULL;

void cvSobel(IplImage * img, CvMat * mat, int dx, int dy, int d)
{
	static bool first = true;
	if (first)
	{
		first = false;
		laplacian = cvCreateMat(3, 3, CV_32FC1);
		cvZero(laplacian);
		cvmSet(laplacian, 0, 1, -1);
		cvmSet(laplacian, 1, 0, -1);
		cvmSet(laplacian, 1, 1, 4);
		cvmSet(laplacian, 1, 2, -1);
		cvmSet(laplacian, 2, 1, -1);

		robert45 = cvCreateMat(3, 3, CV_32FC1);
		cvZero(robert45);
		cvmSet(robert45, 0, 2, -1);
		cvmSet(robert45, 1, 1, 1);

		robert135 = cvCreateMat(3, 3, CV_32FC1);
		cvZero(robert135);
		cvmSet(robert135, 0, 0, -1);
		cvmSet(robert135, 1, 1, 1);

		robert = cvCreateMat(3, 3, CV_32FC1);
		cvZero(robert);
		cvmSet(robert, 0, 0, -1);
		cvmSet(robert, 0, 2, -1);
		cvmSet(robert, 1, 1, 2);

		sobelX = cvCreateMat(3, 3, CV_32FC1);
		cvZero(sobelX);
		cvmSet(sobelX, 0, 0, 1);
		cvmSet(sobelX, 1, 0, 2);
		cvmSet(sobelX, 2, 0, 1);
		cvmSet(sobelX, 0, 2, -1);
		cvmSet(sobelX, 1, 2, -2);
		cvmSet(sobelX, 2, 2, -1);

		sobelY = cvCreateMat(3, 3, CV_32FC1);
		cvZero(sobelY);
		cvmSet(sobelY, 0, 0, -1);
		cvmSet(sobelY, 0, 1, -2);
		cvmSet(sobelY, 0, 2, -1);
		cvmSet(sobelY, 2, 0, 1);
		cvmSet(sobelY, 2, 1, 2);
		cvmSet(sobelY, 2, 2, 1);

		prewittX = cvCreateMat(3, 3, CV_32FC1);
		cvZero(prewittX);
		cvmSet(prewittX, 0, 0, 0.33);
		cvmSet(prewittX, 1, 0, 0.34);
		cvmSet(prewittX, 2, 0, 0.33);
		cvmSet(prewittX, 0, 2, -0.33);
		cvmSet(prewittX, 1, 2, -0.34);
		cvmSet(prewittX, 2, 2, -0.33);

		prewittY = cvCreateMat(3, 3, CV_32FC1);
		cvZero(prewittY);
		cvmSet(prewittY, 0, 0, -0.33);
		cvmSet(prewittY, 0, 1, -0.34);
		cvmSet(prewittY, 0, 2, -0.33);
		cvmSet(prewittY, 2, 0, 0.33);
		cvmSet(prewittY, 2, 1, 0.34);
		cvmSet(prewittY, 2, 2, 0.33);
	}

	if (dx == 1)
		convolution(img, mat, sobelX);
	else
		convolution(img, mat, sobelY);
}

// Cvt COLOR 
#include <cassert>
#define  CV_CAST_8U(t)  (unsigned char)(!((t) & ~255) ? (t) : (t) > 0 ? 255 : 0)
#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))
#define fix(x,n)      (int)((x)*(1 << (n)) + 0.5)
#define  labXr_32f  0.433953f /* = xyzXr_32f / 0.950456 */
#define  labXg_32f  0.376219f /* = xyzXg_32f / 0.950456 */
#define  labXb_32f  0.189828f /* = xyzXb_32f / 0.950456 */

#define  labYr_32f  0.212671f /* = xyzYr_32f */
#define  labYg_32f  0.715160f /* = xyzYg_32f */ 
#define  labYb_32f  0.072169f /* = xyzYb_32f */ 

#define  labZr_32f  0.017758f /* = xyzZr_32f / 1.088754 */
#define  labZg_32f  0.109477f /* = xyzZg_32f / 1.088754 */
#define  labZb_32f  0.872766f /* = xyzZb_32f / 1.088754 */

#define  labRx_32f  3.0799327f  /* = xyzRx_32f * 0.950456 */
#define  labRy_32f  (-1.53715f) /* = xyzRy_32f */
#define  labRz_32f  (-0.542782f)/* = xyzRz_32f * 1.088754 */

#define  labGx_32f  (-0.921235f)/* = xyzGx_32f * 0.950456 */
#define  labGy_32f  1.875991f   /* = xyzGy_32f */ 
#define  labGz_32f  0.04524426f /* = xyzGz_32f * 1.088754 */

#define  labBx_32f  0.0528909755f /* = xyzBx_32f * 0.950456 */
#define  labBy_32f  (-0.204043f)  /* = xyzBy_32f */
#define  labBz_32f  1.15115158f   /* = xyzBz_32f * 1.088754 */

#define  labT_32f   0.008856f

#define labT   fix(labT_32f*255,lab_shift)

#undef lab_shift
#define lab_shift 10
#define labXr  fix(labXr_32f,lab_shift)
#define labXg  fix(labXg_32f,lab_shift)
#define labXb  fix(labXb_32f,lab_shift)
                            
#define labYr  fix(labYr_32f,lab_shift)
#define labYg  fix(labYg_32f,lab_shift)
#define labYb  fix(labYb_32f,lab_shift)
                            
#define labZr  fix(labZr_32f,lab_shift)
#define labZg  fix(labZg_32f,lab_shift)
#define labZb  fix(labZb_32f,lab_shift)

#define labSmallScale_32f  7.787f
#define labSmallShift_32f  0.13793103448275862f  /* 16/116 */
#define labLScale_32f      116.f
#define labLShift_32f      16.f
#define labLScale2_32f     903.3f

#define labSmallScale fix(31.27 /* labSmallScale_32f*(1<<lab_shift)/255 */,lab_shift)
#define labSmallShift fix(141.24138 /* labSmallScale_32f*(1<<lab) */,lab_shift)
#define labLScale fix(295.8 /* labLScale_32f*255/100 */,lab_shift)
#define labLShift fix(41779.2 /* labLShift_32f*1024*255/100 */,lab_shift)
#define labLScale2 fix(labLScale2_32f*0.01,lab_shift)

/* 1024*(([0..511]./255)**(1./3)) */
static unsigned short icvLabCubeRootTab[] = {
       0,  161,  203,  232,  256,  276,  293,  308,  322,  335,  347,  359,  369,  379,  389,  398,
     406,  415,  423,  430,  438,  445,  452,  459,  465,  472,  478,  484,  490,  496,  501,  507,
     512,  517,  523,  528,  533,  538,  542,  547,  552,  556,  561,  565,  570,  574,  578,  582,
     586,  590,  594,  598,  602,  606,  610,  614,  617,  621,  625,  628,  632,  635,  639,  642,
     645,  649,  652,  655,  659,  662,  665,  668,  671,  674,  677,  680,  684,  686,  689,  692,
     695,  698,  701,  704,  707,  710,  712,  715,  718,  720,  723,  726,  728,  731,  734,  736,
     739,  741,  744,  747,  749,  752,  754,  756,  759,  761,  764,  766,  769,  771,  773,  776,
     778,  780,  782,  785,  787,  789,  792,  794,  796,  798,  800,  803,  805,  807,  809,  811,
     813,  815,  818,  820,  822,  824,  826,  828,  830,  832,  834,  836,  838,  840,  842,  844,
     846,  848,  850,  852,  854,  856,  857,  859,  861,  863,  865,  867,  869,  871,  872,  874,
     876,  878,  880,  882,  883,  885,  887,  889,  891,  892,  894,  896,  898,  899,  901,  903,
     904,  906,  908,  910,  911,  913,  915,  916,  918,  920,  921,  923,  925,  926,  928,  929,
     931,  933,  934,  936,  938,  939,  941,  942,  944,  945,  947,  949,  950,  952,  953,  955,
     956,  958,  959,  961,  962,  964,  965,  967,  968,  970,  971,  973,  974,  976,  977,  979,
     980,  982,  983,  985,  986,  987,  989,  990,  992,  993,  995,  996,  997,  999, 1000, 1002,
    1003, 1004, 1006, 1007, 1009, 1010, 1011, 1013, 1014, 1015, 1017, 1018, 1019, 1021, 1022, 1024,
    1025, 1026, 1028, 1029, 1030, 1031, 1033, 1034, 1035, 1037, 1038, 1039, 1041, 1042, 1043, 1044,
    1046, 1047, 1048, 1050, 1051, 1052, 1053, 1055, 1056, 1057, 1058, 1060, 1061, 1062, 1063, 1065,
    1066, 1067, 1068, 1070, 1071, 1072, 1073, 1074, 1076, 1077, 1078, 1079, 1081, 1082, 1083, 1084,
    1085, 1086, 1088, 1089, 1090, 1091, 1092, 1094, 1095, 1096, 1097, 1098, 1099, 1101, 1102, 1103,
    1104, 1105, 1106, 1107, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1117, 1118, 1119, 1120, 1121,
    1122, 1123, 1124, 1125, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1138, 1139,
    1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1154, 1155, 1156,
    1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172,
    1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188,
    1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204,
    1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1215, 1216, 1217, 1218, 1219,
    1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1230, 1231, 1232, 1233, 1234,
    1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249,
    1250, 1251, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1259, 1260, 1261, 1262, 1263,
    1264, 1265, 1266, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1273, 1274, 1275, 1276, 1277,
    1278, 1279, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1285, 1286, 1287, 1288, 1289, 1290, 1291
};

void rgb2lab(int r, int g, int b, int &Lab_L, int &Lab_a, int &Lab_b)
{
	int x, y, z, f;
	int L, a;
	x = x = b*labXb + g*labXg + r*labXr;
	y = b*labYb + g*labYg + r*labYr;
	z = b*labZb + g*labZg + r*labZr;

	f = x > labT;
	x = CV_DESCALE( x, lab_shift );

	if( f )
		assert( (unsigned)x < 512 ), x = icvLabCubeRootTab[x];
	else
		x = CV_DESCALE(x*labSmallScale + labSmallShift,lab_shift);

	f = z > labT;
	z = CV_DESCALE( z, lab_shift );

	if( f )
		assert( (unsigned)z < 512 ), z = icvLabCubeRootTab[z];
	else
		z = CV_DESCALE(z*labSmallScale + labSmallShift,lab_shift);

	f = y > labT;
	y = CV_DESCALE( y, lab_shift );

	if( f )
	{
		assert( (unsigned)y < 512 ), y = icvLabCubeRootTab[y];
		L = CV_DESCALE(y*labLScale - labLShift, 2*lab_shift );
	}
	else
	{
		L = CV_DESCALE(y*labLScale2,lab_shift);
		y = CV_DESCALE(y*labSmallScale + labSmallShift,lab_shift);
	}

	a = CV_DESCALE( 500*(x - y), lab_shift ) + 128;
	b = CV_DESCALE( 200*(y - z), lab_shift ) + 128;

	Lab_L = CV_CAST_8U(L);
	Lab_a = CV_CAST_8U(a);
	Lab_b = CV_CAST_8U(b);
}

void cvCvtColor(IplImage * img, IplImage * gray, int code)
{
	if (code == CV_BGR2GRAY)
	{
		int stepSrc = img->widthStep;
		int step = gray->widthStep;

		unsigned char * srcData = (unsigned char *) img->imageData;
		unsigned char * data = (unsigned char *) gray->imageData;
		for (int i = 0; i < gray->height; ++i)
			for (int j = 0; j < gray->width; ++j)
			{
				float r = 0.299f * srcData[i * stepSrc + j * 3 + 2];
				float g = 0.587f * srcData[i * stepSrc + j * 3 + 1];
				float b = 0.114f * srcData[i * stepSrc + j * 3];
				data[i * step + j] = (unsigned char )(r + g + b);
			}
	}
	else
	{
		int stepSrc = img->widthStep;
		int step = gray->widthStep;

		unsigned char * srcData = (unsigned char *) img->imageData;
		unsigned char * data = (unsigned char *) gray->imageData;
		for (int i = 0; i < gray->height; ++i)
			for (int j = 0; j < gray->width; ++j)
			{
				int r = srcData[i * stepSrc + j * 3 + 2];
				int g = srcData[i * stepSrc + j * 3 + 1];
				int b = srcData[i * stepSrc + j * 3];
				int ll, aa, bb;
				rgb2lab(r, g, b, ll, aa, bb);
				data[i * step + j * 3 + 0] = ll;
				data[i * step + j * 3 + 1] = aa;
				data[i * step + j * 3 + 2] = bb;
			}
	}
}
#endif

#endif