#ifndef IMAGE_ANALYZER
#define IMAGE_ANALYZER

#include "stdafx.h"
#include <string>
#include <iostream>

using namespace std;

#include "Util.h"
#include "imageSegmentation.h"
#include "SalientRegion.h"
#include "CvRegion.h"

class ImageAnalyzer
{
public:
	static const int width = 320;
	static const int height = 240;
	IplImage * oriImage;
	IplImage * srcImage;
	IplImage * gray;
	string filePath;
	string fileName;
	string saveDir;

	vector<IplImage *> images;
	vector<string> saveFiles;
	map<string, IplImage *> ext2image;

	ImageAnalyzer(string file, string save = "")
	{
		oriImage = cvLoadImage(file.data());
		if (oriImage == NULL) return;

		srcImage = cvCreateImage(cvSize(width, height), 8, 3);
		cvResize(oriImage, srcImage);

		filePath = file;
		fileName = Util::getFileTrueName(filePath);
		saveDir = Util::getFilePath(filePath);
		if (save != "") saveDir = save;

		gray = cvCreateImage(cvGetSize(srcImage), 8, 1);
		cvCvtColor(srcImage, gray, CV_BGR2GRAY);
		startAnalysis();
		saveAnalysis();
	}

	~ImageAnalyzer()
	{
		for (int i = 0; i < images.size(); ++i)
			cvReleaseImage(&images[i]);

		if (srcImage != NULL)
			cvReleaseImage(&srcImage);
		if (oriImage != NULL)
			cvReleaseImage(&oriImage);
	}

	string getSaveFilePath(string ext)
	{
		return saveDir + "\\" + fileName + "_" + ext + ".jpg";
	}

	void addImage(IplImage * img, string ext)
	{
		images.push_back(img);
		saveFiles.push_back(getSaveFilePath(ext));
		ext2image[ext] = img;
	}

	/**
	 * 获取图像分割
	 */
	void getSegmentation()
	{
		int coms;
		vector<vector<int> > feature;
		IplImage * image = GraphBasedImageSegmentation::GraphBasedImageSeg(srcImage, feature, coms, 350, 20);

		addImage(image, "seg");
	}

	/** 
	 * 获取图像敏感区域
	 */
	void getSalientRegion()
	{
		IplImage * segment;
		IplImage * salientMap;
		IplImage * salientImage;
		vector<vector<int> > salientMask;
		SalientRegion::getImageSalientRegion(srcImage, segment, salientMap, salientImage, salientMask, 20);

		addImage(segment, "seg");
		addImage(salientMap, "salientMap");
		addImage(salientImage, "salientImage");
	}

	/** 
	 * 获取图像灰度  
	 */
	void getGray()
	{
		addImage(gray, "gray");
	}

	/**
	 * 获取图像 Canny 边缘 
	 */
	void getCanny()
	{
		IplImage * canny = cvCreateImage(cvGetSize(gray), 8, 1);
		cvCanny(gray, canny, 50, 200);

		addImage(canny, "canny");
	}

	/** 
	 * 获得二值化图像
	 */
	void getBinary()
	{
		IplImage * binaryImage = cvCreateImage(cvGetSize(gray), 8, 1);
		cvAdaptiveThreshold(gray, binaryImage, 255,
                          CV_ADAPTIVE_THRESH_MEAN_C,
                          CV_THRESH_BINARY,
                          11, 5 );

		addImage(binaryImage, "binary");
	}

	/** 获得Sobel 边缘 */
	void getSobel()
	{
		IplImage * xSobel = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_32F, 1);
		IplImage * ySobel = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_32F, 1);
		cvSobel(gray, xSobel, 1, 0, 3);
		cvSobel(gray, ySobel, 0, 1, 3);

		addImage(xSobel, "sobelX");
		addImage(ySobel, "sobelY");

		IplImage * sobel = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_32F, 1);
		for (int i = 0; i < srcImage->height; ++i)
			for (int j = 0; j < srcImage->width; ++j)
			{
				double eng1 = cvGet2D(xSobel, i, j).val[0];
				double eng2 = cvGet2D(ySobel, i, j).val[0];
				double eng = eng1 * eng1 + eng2 * eng2;
				eng = sqrt(eng);
				cvSet2D(sobel, i, j, cvScalar(eng));
			}
		addImage(sobel, "sobel");
	}

	/** 获得 Hough 直线 */
	void getHoughLine()
	{
		if (ext2image.find("canny") != ext2image.end())
		{
			vector<pair<CvPoint, CvPoint> > ret;
			IplImage * hough = CvExt::getHoughLines(ext2image["canny"], ret);
			addImage(hough, "hough");
		}
	}

	/** 获得不同区域的表示 */
	void getRegions()
	{
		int coms;
		vector<vector<int> > feature;
		IplImage * image = GraphBasedImageSegmentation::GraphBasedImageSeg(srcImage, feature, coms, 350, 20);
		cvReleaseImage(&image);

		set<string> cals;
		cals.insert("gray");
		cals.insert("hog");
		cals.insert("sobel");
		cals.insert("lab");
		cals.insert("rgb");
		cals.insert("geometry");
		vector<CvRegion> regions = CvRegion::getRegionFromSegment(srcImage, feature, coms, cals);

		IplImage * regionImage = cvCreateImage(cvGetSize(srcImage), 8, 3);
		cvCopy(srcImage, regionImage);

		for (int i = 0; i < regions.size(); ++i)
			regions[i].drawRegionConvexHull(regionImage);

		addImage(regionImage, "region");
	}

	void startAnalysis()
	{
		getSegmentation();
		getSalientRegion();
		getGray();
		getCanny();
		getBinary();
		getSobel();
		getHoughLine();

		getRegions();
	}

	void saveAnalysis()
	{
		for (int i = 0; i < images.size(); ++i)
			cvSaveImage(saveFiles[i].data(), images[i]);
	}
};


#endif