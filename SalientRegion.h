#ifndef SALIENT_REGION_H
#define SALIENT_REGION_H

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>

#include "imageSegmentation.h"
#include "CvImageOperationExt.h";

class SalientRegion
{
private:

static inline double Distance(double x1,double y1,double z1,double x2,double y2,double z2)
{
	double ret = 0.0;
	ret += (x1-x2) * (x1 - x2);
	ret += (y1-y2) * (y1 - y2);
	ret += (z1-z2) * (z1 - z2);
	return sqrt(ret);
}

/** Segment 之后 comID 从1开始 */
static void ChooseSalientRegion(vector<vector<int> > & salientMask, vector<vector<int> > & cluster, 
	CvMat * salientScore, int clusters, int choosen = 5)
{
    vector<bool> ok(clusters + 1, true);
    vector<int> sizes(clusters + 1, 0);
    vector<double> scores(clusters + 1, 0.0);

	int height = cluster.size();
	int width = cluster[0].size();

	salientMask.resize(height);
	for (int i = 0; i < height; ++i)
		salientMask[i].resize(width);

	for (int i = 0; i < height; ++i)
		for (int j = 0; j < width; ++j)
		{
            int tar = cluster[i][j];
			sizes[tar]++;
            scores[tar] += cvmGet(salientScore, i, j);
			if (i <= 5 || i >= height - 5 || 
                    j <= 5 || j >= width - 5) 
				ok[cluster[i][j]] = false;
		}

    /** cluster salient score & clusterID */
    vector<pair<double, int> > clusterInfos;
	for (int i = 1; i <= clusters; ++i) 
        if (ok[i]) {
            double score = 0.0;
            if (sizes[i] > 0) score = scores[i] / (double)sizes[i];
            clusterInfos.push_back(make_pair(score, i));
        }

	sort(clusterInfos.begin(), clusterInfos.end(), greater<pair<double, int> >());
    set<int> choosenIDs;
    for (int i = 0; i < clusterInfos.size(); ++i)
        if (choosenIDs.size() < choosen)
            choosenIDs.insert(clusterInfos[i].second);

	for (int i = 0; i < height; ++i)
		for (int j = 0; j < width; ++j) {
            int tar =cluster[i][j];
            int isSalientRegionScore = 0;
            if (choosenIDs.find(tar) != choosenIDs.end())
                isSalientRegionScore = 1;
			salientMask[i][j] = isSalientRegionScore;
		}
}

//////////////////////////////////////////////////////////////////////////////
// Contrast Detection Filter
//////////////////////////////////////////////////////////////////////////////
static CvMat * Contrast_Detection_Filter(IplImage * in, int wR2)
{
    int height = in->height;
    int width = in->width;
    CvMat * ret = cvCreateMat(height, width, CV_64FC1);
    int delta = wR2 / 2;

    static double sum1[1500][1500];
    static double sum2[1500][1500];
    static double sum3[1500][1500];
    sum1[0][0] = cvGet2D(in, 0, 0).val[0];
    sum2[0][0] = cvGet2D(in, 0, 0).val[1];
    sum3[0][0] = cvGet2D(in, 0, 0).val[2];

    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j) {
            if (i == 0 && j == 0) 
                continue;
            if (j == 0) {
                sum1[i][j] = sum1[i-1][j] + cvGet2D(in, i, j).val[0];
                sum2[i][j] = sum2[i-1][j] + cvGet2D(in, i, j).val[1];
                sum3[i][j] = sum3[i-1][j] + cvGet2D(in, i, j).val[2];
                continue;
            }
            sum1[i][j] = sum1[i][j-1] + cvGet2D(in, i, j).val[0];
            sum2[i][j] = sum2[i][j-1] + cvGet2D(in, i, j).val[1];
            sum3[i][j] = sum3[i][j-1] + cvGet2D(in, i, j).val[2];
            if (i > 0) {
                sum1[i][j] += sum1[i-1][j] - sum1[i-1][j-1];
                sum2[i][j] += sum2[i-1][j] - sum2[i-1][j-1];
                sum3[i][j] += sum3[i-1][j] - sum3[i-1][j-1];
            }
        }

    double t1[3];
    double t2[3];
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            t1[0] = cvGet2D(in, i, j).val[0];
            t1[1] = cvGet2D(in, i, j).val[1];
            t1[2] = cvGet2D(in, i, j).val[2];

            int startX = max(0, i - delta);
            int startY = max(0, j - delta);
            int stopX = min(height - 1, i + delta);
            int stopY = min(width - 1, j + delta);

            t2[0] = sum1[stopX][stopY];
            t2[1] = sum2[stopX][stopY];
            t2[2] = sum3[stopX][stopY];

            if (startX > 0) {
                t2[0] -= sum1[startX -1][stopY];
                t2[1] -= sum2[startX -1][stopY];
                t2[2] -= sum3[startX -1][stopY];
            }
            if (startY > 0) {
                t2[0] -= sum1[stopX][startY - 1];
                t2[1] -= sum2[stopX][startY - 1];
                t2[2] -= sum3[stopX][startY - 1];
            }
            if (startX > 0 && startY > 0) {
                t2[0] += sum1[startX - 1][startY - 1];
                t2[1] += sum2[startX - 1][startY - 1];
                t2[2] += sum3[startX - 1][startY - 1];
            }

            int times = (stopX - startX + 1) * (stopY - startY + 1);
            t2[0] /= (double)times;
            t2[1] /= (double)times;
            t2[2] /= (double)times;
            cvmSet(ret,i,j,Distance(t1[0],t1[1],t1[2],t2[0],t2[1],t2[2]));
        }
    }
    return ret;
}


////////////////////////////////////////////////////////////
// 计算Salient Map
////////////////////////////////////////////////////////////
static CvMat * Salient_Map(IplImage * img)
{
    IplImage * imgLab = cvCreateImage(cvGetSize(img), 8, 3);
    cvCvtColor(img, imgLab, CV_BGR2Lab);

	int range = img->height;
	if (img->width < range) range = img->width;

	CvMat * con1 = Contrast_Detection_Filter(imgLab, range / 8);
	CvMat * con2 = Contrast_Detection_Filter(imgLab, range / 4);
	CvMat * con3 = Contrast_Detection_Filter(imgLab, range / 2);

	CvMat * ret = cvCreateMat(img->height, img->width, CV_64FC1);
	for (int i = 0; i < img->height; ++i)
		for (int j = 0; j <img->width; ++j)
		{
			double t = 0.0;
			t += cvmGet(con1, i, j);
			t += cvmGet(con2, i, j);
			t += cvmGet(con3, i, j);
			cvmSet(ret, i, j, t);
		}

    cvReleaseImage(&imgLab);
	cvReleaseMat(&con1);
	cvReleaseMat(&con2);
	cvReleaseMat(&con3);

	return ret;
}

public:

/////////////////////////////////////////////////////////////////////////////////////
// 给定输入图片，返回 Salient Mask
// salientMask[i][j] != 0 是 Salient Region
/////////////////////////////////////////////////////////////////////////////////////
static CvMat* getImageSalientRegion(IplImage * img , IplImage *& segment, 
        IplImage *& salientMap, IplImage *& salientImage, vector<vector<int> > & salientMask, int select = 10)
{
	// Compute Salient Map
	CvMat * salient = Salient_Map(img);
	salientMap = CvExt::getMatImage(salient);

	// Compute Segmentation
	vector<vector<int> > comID;
	int clusters;
	segment = GraphBasedImageSegmentation::GraphBasedImageSeg(img, comID, clusters, 300, 30);	

	// Compute Salient Region
	ChooseSalientRegion(salientMask, comID, salient, clusters, select);

  // Get Salient Region Image
  salientImage = cvCreateImage(cvSize(img->width, img->height), 8, 3);

	for (int i = 0; i < salientImage->height; ++i)
		for (int j = 0; j < salientImage->width; ++j)
		{
			if (salientMask[i][j] == 0)
				cvSet2D(salientImage, i, j, cvScalar(255, 255, 255));
			else
				cvSet2D(salientImage, i, j, cvGet2D(img, i, j));
		}

	//cvReleaseMat(&salient);
	return salient;
}

static CvMat* getImageSalientMap(IplImage * img)
{
  // Compute Salient Map
	CvMat * salient = Salient_Map(img);
  return salient;
}

};

#endif