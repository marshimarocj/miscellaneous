#ifndef IMAGE_EDITING
#define IMAGE_EDITING

#include "stdafx.h"
#include <string>
#include <iostream>
#include <algorithm>
#include "geometry.h"
#include "CvRegion.h"
#include "Research_EditingModel.h"

using namespace std;

// 实现参见论文 Coordinates for Instant Image Cloning
class ImageEditing
{
public:

	// 给定一个mask矩阵，获得他的边缘点，边缘按照一定方向排序
	static void getContourPoints(IplImage * mask, vector<CvPoint> & contours, int maxsize = -1)
	{
		IplImage * temp = cvCreateImage(cvGetSize(mask), 8, 1);
		cvCopy(mask, temp);

		CvMemStorage * storage = cvCreateMemStorage(0);
		CvSeq * pc = 0;
		cvFindContours(mask, storage, &pc, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cvPoint(0,0));

		contours.clear();
		if (pc != NULL)
			for (int i = 0; i < pc->total; i++)
			{
				CvPoint * p = (CvPoint *) cvGetSeqElem(pc , i);
				contours.push_back(*p);
			}

    random_shuffle(contours.begin(), contours.end());
    if (maxsize >= 0 && contours.size() > maxsize) contours.resize(maxsize);
		cvReleaseMemStorage(&storage);
		cvReleaseImage(&temp);
	}

	// 获得lamda系数
	static vector<double> getLamda(vector<CvPoint> & contours, CvPoint x)
	{
		int N = contours.size();
		vector<double> w(N);
		for (int i = 0; i < N; ++i)
			w[i] = 1.0 / sqrt(sqr(x.x - contours[i].x) + sqr(x.y - contours[i].y));

		double total = 0;
		for (int i = 0; i < N; ++i) total += w[i];

		vector<double> ret(N);
		for (int i = 0; i < N; ++i) ret[i] = w[i] / total;
		return ret;
	}

	// 图像编辑模型
	static void imageEditing2(IplImage * src, IplImage * dst, IplImage * mask, IplImage * ret)
	{
		//cvShowImage("src", src);
		//cvShowImage("dst", dst);
		//cvShowImage("mask", mask);
		//cvWaitKey(0);
		int height = src->height;
		int width = src->width;
		int step = src->widthStep;
		unsigned char * srcData = (unsigned char *) src->imageData;
		unsigned char * dstData = (unsigned char *) dst->imageData;
		unsigned char * retData = (unsigned char *) ret->imageData;
		unsigned char * maskData = (unsigned char *) mask->imageData;

		vector<vector<bool> > onMask(height, vector<bool>(width, false));
		vector<vector<bool> > onContour(height, vector<bool>(width, false));
		vector<CvPoint> contours;
		getContourPoints(mask, contours);
		for (int i = 0; i < contours.size(); ++i)
			onContour[contours[i].y][contours[i].x] = true;

		for (int i = 0; i < height; ++i)
			for (int j = 0; j < width; ++j)
				if (maskData[i * step + j] > 0) 
					onMask[i][j] = true;

		vector<vector<double> > f0(height, vector<double>(width));
		vector<vector<double> > f1(height, vector<double>(width));
		for (int i = 0; i < height; ++i)
			for (int j = 0; j < width; ++j)
			{
				f0[i][j] = (srcData[i * step + j]) / 256.0;
				f1[i][j] = (dstData[i * step + j]) / 256.0;
			}

		//EditingModel model(f0, f1, vpq_cloning, f1p_cloning, onMask, onContour);
		//model.genMatrix();
		//model.outputMatrix("D:\\A.txt", "D:\\b.txt");
		//system("pause");
	}

	// 将原始图像src，嵌入到 目标图像dst中，只替换mask中的区域，结果图像存在ret 中
	static void imageEditing(IplImage * src, IplImage * dst, IplImage * mask, IplImage * ret, 
    vector<CvPoint> & contours, 
    vector<vector<bool> > & onContour, 
    vector<vector<vector<double> > > & lamdas)
	{
		int height = src->height;
		int width = src->width;
		int step = src->widthStep;
		unsigned char * srcData = (unsigned char *) src->imageData;
		unsigned char * dstData = (unsigned char *) dst->imageData;
		unsigned char * retData = (unsigned char *) ret->imageData;
		unsigned char * maskData = (unsigned char *) mask->imageData;

    /**
		vector<vector<bool> > onContour(height, vector<bool>(width, false));
		vector<CvPoint> contours;
		getContourPoints(mask, contours, -1);

		IplImage * conImage = cvCreateImage(cvGetSize(mask), 8, 3);
		cvZero(conImage);
		for (int i = 0; i < contours.size(); ++i)
		{
			onContour[contours[i].y][contours[i].x] = true;
			cvCircle(conImage, contours[i], 1, CV_RGB(255, 0, 0), 1);
		}

		cvShowImage("contour", conImage, 64, 64, 512, 0);
		cvShowImage("mask", mask, 64, 64, 640, 0);
		//cvWaitKey(0);
		debug1(contours.size());
    */

		vector<double> deltaInContour(contours.size());
		for (int k = 0; k < contours.size(); ++k)
		{
			int i = contours[k].y;
			int j = contours[k].x;
			deltaInContour[k] = dstData[i * step + j] - srcData[i * step + j];
		}
		
		/** image editing */
		for (int i = 0; i < height; ++i)
		{
			for (int j = 0; j < width; ++j)
			{
				if (maskData[i * step + j] > 0 && !onContour[i][j])
				{
					CvPoint x = cvPoint(j, i);
					vector<double> & lamda = lamdas[i][j];
          //debug2(x, lamda.size());
					double r = 0;
					for (int k = 0; k < lamda.size(); ++k)
						r += lamda[k] * deltaInContour[k];
					int fstar = srcData[i * step + j] + r;
					if (fstar < 0) fstar = 0;
					if (fstar > 255) fstar = 255;

					retData[i * step + j] = fstar;
				}
				else 
					retData[i * step + j] = dstData[i * step + j];
			}
		}
	}

  static void calContourAndLamda(IplImage * mask, vector<CvPoint> & contours, vector<vector<bool> > & onContour, 
    vector<vector<vector<double> > > & lamdas)
  {
    int height = mask->height; int width = mask->width;
    onContour = vector<vector<bool> >(height, vector<bool>(width, false));
    lamdas = vector<vector<vector<double> > >(height, vector<vector<double> >(width));

    // 计算Contour
		contours.clear();
		getContourPoints(mask, contours, 50);

		IplImage * conImage = cvCreateImage(cvGetSize(mask), 8, 3);
		cvZero(conImage);
		for (int i = 0; i < contours.size(); ++i)
		{
			onContour[contours[i].y][contours[i].x] = true;
      for (int j = 1; j <= 5; ++j)
			  cvCircle(conImage, contours[i], j, CV_RGB(255, 0, 0), 1);
		}

		//cvShowImage("contour", conImage, 64, 64, 512, 0);
    cvReleaseImage(&conImage);

    // 计算Lamda
    unsigned char * maskData = (unsigned char *) mask->imageData;
    int step = mask->widthStep;

    for (int i = 0; i < height; ++i)
			for (int j = 0; j < width; ++j)
				if (maskData[i * step + j] > 0 && !onContour[i][j])
				{
					CvPoint x = cvPoint(j, i);
					lamdas[i][j] = getLamda(contours, x);
          //debug2(x, lamdas[i][j].size());
				}
  }

	// 将三通道图像嵌入另一图像 
	// src dst 大小可以不同 mask 是作用在src上的mask, x, y 是相对于dst的坐标
	static void imageEditing3(IplImage * srcRGB, IplImage * dstRGB, IplImage * mask, int x, int y, IplImage * ret)
	{
    // 缩放mask
    IplImage * realsrc = cvCreateImage(cvGetSize(dstRGB), 8, 3);
		IplImage * realmask = cvCreateImage(cvGetSize(dstRGB), 8, 1);
		cvZero(realmask);
    cvZero(realsrc);
    int binarystep = realmask->widthStep;
    int step = realsrc->widthStep;
		for (int i = 0; i < mask->height; ++i)
			for (int j = 0; j < mask->width; ++j)
			{
				int ni = i + y;
				int nj = j + x;
				realmask->imageData[ni * binarystep + nj] = mask->imageData[i * mask->widthStep + j];
			}

    for (int i = 0; i < srcRGB->height; ++i)
      for (int j = 0; j < srcRGB->width; ++j)
        for (int k = 0; k < 3; ++k)
        {
          int ni = i + y;
          int nj = j + x;
          realsrc->imageData[ni * step + nj * 3 + k] = srcRGB->imageData
            [i * srcRGB->widthStep + j * 3 + k];
        }

    //cvShowImage("realsrc", realsrc, 64, 64, 512, 0);
		//cvShowImage("realmask", realmask, 64, 64, 640, 0);

    vector<CvPoint> contours;
    vector<vector<bool> > onContour;
    vector<vector<vector<double> > > lamdas;
    int start = clock();
    calContourAndLamda(realmask, contours, onContour, lamdas);
    int stop = clock();
    //debug2("contour lamda", stop - start);

    start = clock();
		IplImage * realsrcR = cvCreateImage(cvGetSize(realsrc), 8, 1);
		IplImage * realsrcG = cvCreateImage(cvGetSize(realsrc), 8, 1);
		IplImage * realsrcB = cvCreateImage(cvGetSize(realsrc), 8, 1);
		cvCvtPixToPlane(realsrc, realsrcB, realsrcG, realsrcR, NULL);

		IplImage * dstR = cvCreateImage(cvGetSize(dstRGB), 8, 1);
		IplImage * dstG = cvCreateImage(cvGetSize(dstRGB), 8, 1);
		IplImage * dstB = cvCreateImage(cvGetSize(dstRGB), 8, 1);
		cvCvtPixToPlane(dstRGB, dstB, dstG, dstR, NULL);

		IplImage * retR = cvCreateImage(cvGetSize(dstRGB), 8, 1);
		IplImage * retG = cvCreateImage(cvGetSize(dstRGB), 8, 1);
		IplImage * retB = cvCreateImage(cvGetSize(dstRGB), 8, 1);

		ImageEditing::imageEditing(realsrcR, dstR, realmask, retR, contours, onContour, lamdas);
		ImageEditing::imageEditing(realsrcG, dstG, realmask, retG, contours, onContour, lamdas);
		ImageEditing::imageEditing(realsrcB, dstB, realmask, retB, contours, onContour, lamdas);
		cvMerge(retB, retG, retR, NULL, ret);

    stop = clock();
    //debug2("editing", stop - start);

		cvReleaseImage(&realsrcR);
		cvReleaseImage(&realsrcG);
		cvReleaseImage(&realsrcB);
		cvReleaseImage(&dstR);
		cvReleaseImage(&dstG);
		cvReleaseImage(&dstB);
		cvReleaseImage(&retR);
		cvReleaseImage(&retR);
		cvReleaseImage(&retR);
    cvReleaseImage(&realmask);
    cvReleaseImage(&realsrc);
	}

	// 根据要嵌入的图像的轮廓和边缘自动产生mask
	static IplImage * genMask(IplImage * embedd)
	{
    static int cnt = 0;
		IplImage * canny = CvExt::getCannyDetection(embedd);
    //cvShowImage(string("genmask") + StringOperation::toString(cnt++), canny, 64, 64, cnt * 128, 6000);
		cvDilate(canny, canny);
		cvDilate(canny, canny);

		//cvNamedWindow("canny");
    //cvShowImage("canny", canny);
    //cvWaitKey(0);
    //cvShowImage(string("genmask") + StringOperation::toString(cnt++), canny, 64, 64, cnt * 128, 6000);
		//cvWaitKey(0);

		// 计算联通分量去除边缘
		vector<vector<int> > comID;
		vector<int> comSize;
		floodFill(canny, comID, comSize, 4);

		int height = canny->height;
		int width = canny->width;
		unsigned char * cannyData = (unsigned char *) canny->imageData;
		int step = canny->widthStep;

		set<int> remove;
		for (int i = 0; i < height; ++i)
		{
			if (cannyData[i * step + 0] == 0) remove.insert(comID[i][0]);
			if (cannyData[i * step + width - 1] == 0) remove.insert(comID[i][width - 1]);
		}

		for (int j = 0; j < width; ++j)
		{
			if (cannyData[j] == 0) remove.insert(comID[0][j]);
			if (cannyData[(height - 1) * step + j] == 0) remove.insert(comID[height - 1][j]);
		}

		for (int i = 0; i < height; ++i)
			for (int j = 0; j < width; ++j)
				if (remove.find(comID[i][j]) != remove.end())
					cannyData[i * step + j] = 0;
				else
					cannyData[i * step + j] = 255;

		// 平滑之后，保留最大的联通分量
		//cvNamedWindow("canny");
    //cvShowImage(string("genmask") + StringOperation::toString(cnt++), canny, 64, 64, cnt * 128, 600);
		//cvWaitKey(0);

		cvSmooth(canny, canny, CV_MEDIAN, 5);
		cvSmooth(canny, canny, CV_MEDIAN);

		floodFill(canny, comID, comSize, 4);
		int maxValidSize = 0;
		int maxValidID = -1;
		for (int i = 0; i < height; ++i)
			for (int j = 0; j < width; ++j)
				if (cannyData[i * step + j])
				{
					int id = comID[i][j];
					if (comSize[id] > maxValidSize)
					{
						maxValidSize = comSize[id];
						maxValidID = id;
					}
				}

		for (int i = 0; i < height; ++i)
			for (int j = 0; j < width; ++j)
			{
				int id = comID[i][j];
				if (id != maxValidID) cannyData[i * step + j] = 0;
			}

    int cannyCount = 0;
    for (int i = 0; i < height; ++i)
      for (int j = 0; j < width; ++j)
        if (cannyData[i * step + j] > 0)
          cannyCount++;

    double cannyRatio = (double) cannyCount / (double) height / (double) width;
    if (cannyRatio < 0.3)
    {
      for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
          cannyData[i * step + j] = 255;
    }

    //cvShowImage(string("genmask") + StringOperation::toString(cnt++), canny, 64, 64, cnt * 128, 600);
		//cvNamedWindow("canny2");
		//cvShowImage("canny2", canny);
		//cvWaitKey(0);
		return canny;
	}

	// 从图像中抹去一块区域，自动填充合适的背景
	static void removeRegion(IplImage * img, CvRect rect)
	{
		IplImage * subImage = CvExt::getSubImage(img, rect);
		IplImage * mask = genMask(subImage);
		cvDilate(mask, mask);
		cvDilate(mask, mask);
		cvDilate(mask, mask);

		//cvShowImage("removemask", mask, 128, 128, 640, 0);
		
		IplImage * bigmask = cvCreateImage(cvGetSize(img), 8, 1);
		unsigned char * maskData = (unsigned char *) mask->imageData;
		unsigned char * bmaskData = (unsigned char *) bigmask->imageData;
		int mstep = mask->widthStep;
		int bmstep = bigmask->widthStep;

		cvZero(bigmask);
		for (int i = 0; i < mask->height; ++i)
			for (int j = 0; j < mask->width; ++j)
				if (maskData[i * mstep + j] > 0) 
					bmaskData[(i + rect.y) * bmstep + j + rect.x] = 255; 

		//cvDilate(bigmask, bigmask);
		//cvShowImage("bigremovemask", bigmask, 128, 128, 640, 0);
		//cvWaitKey(0);

		unsigned char * data = (unsigned char *) img->imageData;
		int step = img->widthStep;

		int si = rect.y; int ei = rect.y + rect.height -1;
		int sj = rect.x; int ej = rect.x + rect.width - 1;
		for (int k = 0; k < 3; ++k)
		{
			for (int i = si; i <= ei; ++i)
				for (int j = sj; j <= ej; ++j)
					if (bmaskData[i * bmstep + j] > 0)
					{
						int now = 0;
						int cnt = 0;
						for (int d1 = -3; d1 <= 3; ++d1)
							for (int d2 = -rect.width / 2; d2 <= rect.width / 2; ++d2)
							{
								if (d1 == 0 && d2 == 0) continue;
								int ti = i + d1;
								int tj = j + d2;
                if (ti < 0 || tj < 0 || ti >= img->height || tj >= img->width) continue;
								if (ti < si || ti > ei || tj < sj - 5 || tj > ej + 5) continue;
								if (bmaskData[ti * bmstep + tj] > 0) continue;

								now += data[ti * step + tj * 3 + k];
								cnt++;								
							}
							if (cnt == 0) continue;
							data[i * step + j * 3 + k] = now / cnt;
					}
		}

		cvReleaseImage(&mask);
		cvReleaseImage(&subImage);
		cvReleaseImage(&bigmask);
	}
};

void testedit()
{
	cvNamedWindow("ret");
	//cvNamedWindow("src");
	//cvNamedWindow("dst");
	cvNamedWindow("mask");
	//cvNamedWindow("temp");

	cvMoveWindow("src", 200, 0);
	cvMoveWindow("dst", 400, 0);
	cvMoveWindow("mask", 600, 0);
	cvMoveWindow("ret", 800, 0);
	cvMoveWindow("temp", 1000, 0);

	string src = "D:\\workspace_matlab\\poissonSolver\\pic\\3_.jpg";
	string dst = "D:\\workspace_matlab\\poissonSolver\\pic\\3.jpg";

	IplImage * srcImage = cvLoadImage(src.data());
	IplImage * dstImage = cvLoadImage(dst.data());
	IplImage * mask = cvCreateImage(cvGetSize(srcImage), 8, 1);
	
	cvZero(mask);
	unsigned char * srcData = (unsigned char *) srcImage->imageData;
	int step = srcImage->widthStep;
	for (int i = 0; i < mask->height; ++i)
		for (int j = 0; j < mask->width; ++j)
		{
			int r = srcData[i * step + j * 3 + 2];
			int g = srcData[i * step + j * 3 + 1];
			int b = srcData[i * step + j * 3 + 0];
			if (r < 255 || g < 255 || b < 255) 
				mask->imageData[i * mask->widthStep + j] = 255;
		}
	cvErode(mask, mask);
	cvErode(mask, mask);
	IplImage * ret = cvCreateImage(cvGetSize(dstImage), 8, 3);

	ImageEditing::imageEditing3(srcImage, dstImage, mask, 10, 10, ret);

	cvShowImage("ret", ret);
	cvShowImage("mask", mask);
	cvWaitKey(0);
	return;
}


#endif