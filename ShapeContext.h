#ifndef SHAPE_H
#define SHAPE_H

#include "stdafx.h"
#include <vector>
#include <queue>
#include <map>
#include <cmath>
#include <set>
#include <map>

#include "geometry.h"
#include "Algo_BipartiteMaxMatch.h"

double getdist(CvPoint & p1, CvPoint & p2)
{	return sqrt(sqr(p1.x - p2.x) + sqr(p1.y - p2.y)); }

const int radiusBins = 5;
const int angleBins = 12;
const double radiusMin = 0.125;
const double radiusMax = 2.0;

class ShapeContext
{
public:
	vector<CvPoint> points;

	vector<vector<double> > angleMatrix;

	vector<vector<double> > radiusMatrix;
	vector<double> radiusSplits;

	// 每个采样点计算直方图 Shape Context
	vector<vector<double> > histograms;

	// 初始化参数
	void init()
	{
		radiusSplits = vector<double>(radiusBins);
		double l = log(radiusMin);
		double r = log(radiusMax);

		for (int i = 0; i < radiusBins; ++i)
		{
			double t = l + (r - l) / radiusBins * i;
			radiusSplits[i] = exp(t);
		}
	}

	// 计算角度矩阵
	void calAngleMatrix()
	{
		double pi2 = pi * 2;

		angleMatrix = vector<vector<double> >(points.size(), vector<double>(points.size()));
		for (int i = 0; i < points.size(); ++i)
			for (int j = 0; j < points.size(); ++j)
			{
				angleMatrix[i][j] = atan2((double)points[j].y - points[i].y, (double)points[j].x - points[i].x);
				if (angleMatrix[i][j] < 0) angleMatrix[i][j] += pi2;
			}
	}

	// 计算距离矩阵
	void calRadiusMatrix()
	{
		double sumdist = 0;
		radiusMatrix = vector<vector<double> >(points.size(), vector<double>(points.size()));

		for (int i = 0; i < points.size(); ++i)
			for (int j = i; j < points.size(); ++j)
			{
				double dis = getdist(points[i], points[j]);
				radiusMatrix[i][j] = radiusMatrix[j][i] = dis;
				sumdist += dis * 2;
			}

		double averdist = sumdist / points.size() / points.size();
		for (int i = 0; i < points.size(); ++i)
			for (int j = 0; j < points.size(); ++j)
				radiusMatrix[i][j] /= averdist;
	}

	// 计算直方图
	void calHistogram()
	{
		int bins = radiusBins * angleBins;
		histograms.resize(points.size());
		for (int i = 0; i < points.size(); ++i)
			histograms[i] = vector<double>(bins, 0);

		double angleFactor = pi * 2 / angleBins;

		for (int i = 0; i < points.size(); ++i)
		{
			double total = 0;
			for (int j = 0; j < points.size(); ++j)
				if (j != i)
				{
					int anglebin = (int) (angleMatrix[i][j] / angleFactor);
					if (anglebin < 0) anglebin = 0;
					if (anglebin >= angleBins) anglebin = angleBins - 1;

					int radiusbin = 4;
					for (int k = 0; k < radiusBins; ++k)
						if (angleMatrix[i][j] < radiusSplits[k]) 
						{
							radiusbin = k;
							break;
						}

					histograms[i][radiusbin * 12 + anglebin]++;
					total++;
				}

			for (int j = 0; j < bins; ++j)
				histograms[i][j] /= total;
		}
	}

	ShapeContext(vector<CvPoint> & ps)
	{
		points = ps;
		init();
		calAngleMatrix();
		calRadiusMatrix();
		calHistogram();
	}

	static vector<int> randomSample(int N, int rev)
	{
		vector<int> id(N);
		for (int i = 0; i < N; ++i)
			id[i] = i;

		for (int i = 0; i < N; ++i)
			swap(id[i], id[i + rand() % (N - i)]);

		if (rev > N) rev = N;
		vector<int> ret(rev);
		for (int i = 0; i < rev; ++i)
			ret[i] = id[i];
		return ret;
	}

	// 计算两个直方图的距离
	static double calHistogramDist(vector<double> & h1, vector<double> & h2)
	{
		double ret = 0;
		for (int i = 0; i < h1.size(); ++i)
			if (h1[i] + h2[i] > 0)
				ret += sqr(h1[i] - h2[i]) / (h1[i] + h2[i]);
		ret /= 2;
		return ret;
	}

	// Shape Context Matching
	static vector<pair<CvPoint, CvPoint> > shapeContextMatch(ShapeContext & s1, ShapeContext & s2)
	{
		vector<int> leftid = randomSample(s1.points.size(), 300);
		vector<int> rightid = randomSample(s2.points.size(), 300);
		int leftSize = leftid.size();
		int rightSize = rightid.size();
		vector<vector<int> > score(leftSize, vector<int>(rightSize));

		for (int i = 0; i < leftSize; ++i)
			for (int j = 0; j < rightSize; ++j)
				score[i][j] = (int) ((1.0 - calHistogramDist(s1.histograms[leftid[i]], s2.histograms[rightid[j]])) * 100);

		static BipartiteMaxMatch * match = new BipartiteMaxMatch();
		vector<pair<int, int> > matches = match->getMatch(leftSize, rightSize, score);

		vector<pair<CvPoint, CvPoint> > ret;
		for (int k = 0; k < matches.size(); ++k)
		{
			int i = k;
			int j = matches[k].first;
			if (j >= 0) ret.push_back(make_pair(s1.points[leftid[i]], s2.points[rightid[j]]));
		}
		return ret;
	}

	// 根据匹配找到投影矩阵
	static CvMat * findHomography(vector<pair<CvPoint, CvPoint> > & mps)
	{
		CvMat* src = cvCreateMat(mps.size(), 2, CV_64FC1);
		CvMat* dst = cvCreateMat(mps.size(), 2, CV_64FC1);
		for (int i = 0; i < mps.size(); ++i)
		{
			cvmSet(src, i, 0, mps[i].first.x);
			cvmSet(src, i, 1, mps[i].first.y);
			cvmSet(dst, i, 0, mps[i].second.x);
			cvmSet(dst, i, 1, mps[i].second.y);
		}
		CvMat * H = cvCreateMat(3, 3, CV_64FC1);
		CvMat * status = cvCreateMat(1, mps.size(), CV_8UC1);
		cvFindHomography(src, dst, H, CV_RANSAC, 10.0, status);

		int cnt = 0;
		for(int i = 0; i < mps.size(); i++)
			if((int)status->data.ptr[i]) cnt++;
		debug1(cnt);

		cvReleaseMat(&src);
		cvReleaseMat(&dst);
		cvReleaseMat(&status);
		return H;
	}

	static CvPoint getHomographyTrans(CvPoint & src, CvMat * H)
	{	
		double x = src.x; 
		double y = src.y;
		double Z = 1.0 / ((cvmGet(H, 2, 0) * x + cvmGet(H, 2, 1) * y + cvmGet(H, 2, 2)));
		double X = (cvmGet(H, 0, 0) * x + cvmGet(H, 0, 1) * y + cvmGet(H, 0, 2)) * Z;
        double Y = (cvmGet(H, 1, 0) * x + cvmGet(H, 1, 1) * y + cvmGet(H, 1, 2)) * Z;

		CvPoint p = cvPoint(X, Y);
		return p;
	}

	static double getMatchError(vector<pair<CvPoint, CvPoint> > & mps, CvMat * H)
	{
		double ret = 0;
		for (int i = 0; i < mps.size(); ++i)
		{
			CvPoint pp = getHomographyTrans(mps[i].first, H);
			ret += dist(pp, mps[i].second);
		}
		return ret / mps.size();
	}
};


#endif