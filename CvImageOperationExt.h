#ifndef CV_IMAGE_OPERATION_EXT_H
#define CV_IMAGE_OPERATION_EXT_H

#include "stdafx.h"

namespace CvExt
{

  using namespace std;

  bool skelNow[100 * 100];
  bool skelErase[100 * 100];

  /*/////////////////////////////////////////////////////////////////////////////////////// 
  名称:      GetSubImage 
  功能:      求输入图像的子图像 
  算法:       
  参数: 
  image  - 输入图像 
  roi    - 子图像的定义区域，region of interests 
  返回: 
  如果成功，返回创建好的子图像 
  注意事项: 
  子图像在函数中创建，用完后需要释放内存. 
  //*///////////////////////////////////////////////////////////////////////////////////////  
  IplImage* getSubImage(IplImage * image, CvRect roi)  
  {  
    if (roi.x < 0) roi.x = 0;
    if (roi.y < 0) roi.y = 0; 
    if (roi.x + roi.width >= image->width) roi.width = image->width - roi.x;
    if (roi.y + roi.height >= image->height) roi.height = image->height - roi.y;

    IplImage * result = cvCreateImage(cvSize(roi.width, roi.height), image->depth, image->nChannels);
    int step = result->widthStep;
    int stepSrc = image->widthStep;
    unsigned char * data = (unsigned char *) result->imageData;
    for (int i = 0; i < result->height; ++i)
      for (int j = 0; j < result->width; ++j)
        for (int k = 0; k < result->nChannels; ++k)
          data[i * step + j * result->nChannels + k] = 
          image->imageData[(i + roi.y) * stepSrc + (j + roi.x) * result->nChannels + k];
    return result;  
  }  

  /////////////////////////////////////////////////////////////////
  // 设置输入图像的子图像 
  /////////////////////////////////////////////////////////////////
  void setSubImage(IplImage * src, IplImage * embedd, int x, int y)
  {
    unsigned char * srcData = (unsigned char *) src->imageData;
    unsigned char * embData = (unsigned char *) embedd->imageData;
    int srcStep = src->widthStep;
    int embStep = embedd->widthStep;
    int c = src->nChannels;
    int ce = embedd->nChannels;

    for (int i = 0; i < embedd->height; ++i)
      for (int j = 0; j < embedd->width; ++j)
      {
        int ni = i + y;
        int nj = j + x;
        if (ni < src->height && nj < src->width)
        {
          for (int k = 0; k < min(c, ce); ++k)
            srcData[ni * srcStep + nj * c + k] = embData[i * embStep + j * ce + k];
          for (int l = ce; l < c; ++l)
            srcData[ni * srcStep + nj * c + l] = srcData[ni * srcStep + nj * c];
        }
      }
  }


  /////////////////////////////////////////////////
  // 按行列排版图片，组成一张大图
  /////////////////////////////////////////////////
  IplImage * combineImageRowCol(vector<IplImage*> images, int W)
  {
    vector<CvPoint> posis;
    int nowx = 0;
    int nowy = 0;
    int nowmaxH = 0;
    int maxW = 0;
    int i;
    for (i = 0; i < images.size(); ++i) maxW = max(maxW, images[i]->width);
    maxW = max(maxW, W);

    for (i = 0; i < images.size(); ++i)
    {
      int h = images[i]->height;
      int w = images[i]->width;
      if (w + nowx > maxW)
      {
        nowy += nowmaxH + 5;
        nowx = 0;
        nowmaxH = 0;
      }

      posis.push_back(cvPoint(nowx, nowy));
      nowx += w + 5;
      nowmaxH = max(nowmaxH, h);
    }

    nowy += nowmaxH + 5;
    IplImage * ret = cvCreateImage(cvSize(maxW, nowy), 8, 3);
    for (i = 0; i < images.size(); ++i)
    {
      //debug2(posis[i].x, posis[i].y);
      CvExt::setSubImage(ret, images[i], posis[i].x, posis[i].y);
    }
    return ret;
  }

  /////////////////////////////////////////////////////////////////////////////////////
  // 转换到 Gray
  /////////////////////////////////////////////////////////////////////////////////////
  //RGB[A] to Gray:Y ← 0.299·R+0.587·G+0.114·B
  IplImage * cvtColorToGray(IplImage * img)
  {
    IplImage * gray = cvCreateImage(cvGetSize(img), 8, 1);
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
      return gray;
  }


  ////////////////////////////////////////////////////////
  // 给定输入灰度图片，获得他的0-1二值化表示
  ////////////////////////////////////////////////////////
  IplImage * getBinaryImage(IplImage * img, int blocksize, int param)
  {
    IplImage * ret = cvCreateImage(cvGetSize(img), 8, 1);
    unsigned char * data = (unsigned char *) img->imageData;
    int step = img->widthStep;
    unsigned char * retdata = (unsigned char *) ret->imageData;
    int stepret = ret->widthStep;

    for (int i = 0; i < ret->height; ++i)
      for (int j = 0; j < ret->width; ++j)
      {
        int average = 0;
        int count = 0;
        for (int x = -5; x <= 5; ++x)
          for (int y = -5; y <= 5; ++y)
          {
            int ni = i + x;
            int nj = j + y;
            if (ni >= 0 && ni < img->height && nj >= 0 && nj < img->width)
            {
              count++;
              average += data[ni * step + nj];
            }
          }
          int now = data[i * step + j];
          average /= count;
          average -= param;
          if (now > average)
            retdata[i * stepret + j] = 255;
          else
            retdata[i * stepret + j] = 0;
      }
      return ret;
  }

#ifndef DSP
  //////////////////////////////////////////////////////////////////////////////////////////
  // 给定输入图片，获得它的0-1二值化表示
  //////////////////////////////////////////////////////////////////////////////////////////
  IplImage * getBinaryImage(IplImage * img)
  {
    IplImage * gray = cvtColorToGray(img);

    IplImage * binaryImage = cvCreateImage(cvGetSize(img), 8, 1);
    cvAdaptiveThreshold(gray, binaryImage, 255,
      CV_ADAPTIVE_THRESH_MEAN_C,
      CV_THRESH_BINARY,
      11, 5 );

    //IplImage * binaryImage = getBinaryImage(gray, 11, 5);

    cvReleaseImage(&gray);
    return binaryImage;
  }
#endif

#ifndef DSP
  ///////////////////////////////////////////////////////////////////////////////////////////
  // 给定输入单通道图片，获得他的0-255二值化表示
  ///////////////////////////////////////////////////////////////////////////////////////////
  IplImage * getBinaryImageFromSingleChannelImage(IplImage * img)
  {
    int i, j;
    double minValue = 1e10;
    double maxValue = 0;

    for (i = 0; i < img->height; ++i)
      for (j = 0; j < img->width; ++j)
      {
        double nowValue = cvGet2D(img, i, j).val[0];
        if (nowValue < minValue) minValue = nowValue;
        if (nowValue > maxValue) maxValue = nowValue;
      }

      double range = maxValue - minValue;
      IplImage * inputImage = cvCreateImage(cvGetSize(img), 8, 1);

      for (i = 0; i < img->height; ++i)
        for (j = 0; j < img->width; ++j)
        {
          double delta = cvGet2D(img, i, j).val[0] - minValue;
          cvSet2D(inputImage, i, j, cvScalar(delta / range * 255));
        }

        IplImage * binaryImage = cvCreateImage(cvGetSize(inputImage), 8, 1);
        cvAdaptiveThreshold(inputImage, binaryImage, 255,
          CV_ADAPTIVE_THRESH_MEAN_C,
          CV_THRESH_BINARY_INV,
          11, 5 );


        return binaryImage;
  }
#endif

  ////////////////////////////////////////////////////////////////////////////////////
  // 给定一个图像的二值化，求他的skeletonization
  // 输入图像为二值化后的图像
  ////////////////////////////////////////////////////////////////////////////////////
  IplImage * getSkeletonImage(IplImage * img)
  {
    int i, j;
    int dir[8][2] = { {-1, 0}, {1, 0}, {0, -1}, {0, 1}, {1, 1}, {1, -1}, {-1, -1}, {-1, 1} };

    IplImage * ret = cvCreateImage(cvGetSize(img), 8, 1);
    int height = img->height;
    int width = img->width;
    int step = img->widthStep;
    unsigned char * data = (unsigned char * ) img->imageData;
    unsigned char * retdata = (unsigned char *) ret->imageData;

    bool * now = skelNow;
    bool * erase = skelErase;
    for (i = 1; i < height - 1; ++i)
      for (j= 1; j < width - 1; ++j)
      {
        if (data[i * step + j] != 0) 
          now[i * width + j] = true;
        else
          now[i * width + j] = false;
        erase[i * width + j] = false;
      }

      while (true)
      {
        bool changed = false;

        for (i = 1; i < height - 1; ++i)
          for (j = 1; j < width - 1; ++j)
            if (now[i * width + j])
            {
              int neighNone = 0;
              int neighIs = 0;
              for (int k = 0; k < 8; ++k)
              {
                int ni = i + dir[k][0];
                int nj = j + dir[k][1];
                if (now[ni * width + nj]) 
                  neighIs++;
                else
                  neighNone++;
              }

              if (neighNone > 0 && neighIs >= 2 && neighIs <= 6)
              {
                bool t = now[(i - 1) * width + j];
                bool r = now[i * width + j + 1];
                bool b = now[(i + 1) * width +j];
                bool l = now[i * width + j - 1];
                bool tl = now[(i - 1) * width + j - 1];
                bool tr = now[(i - 1) * width + j + 1];
                bool bl = now[(i + 1) * width + j - 1];
                bool br = now[(i + 1) * width + j + 1];

                if (t && r && b) continue;
                if (l && r && b) continue;
                int vx = 0;
                if (t && !tr) vx++;
                if (tr && !r) vx++;
                if (r && !br) vx++;
                if (br && !b) vx++;
                if (b && !bl) vx++;
                if (bl && !l) vx++;
                if (l && !tl) vx++;
                if (tl && !t) vx++;
                if (vx == 1) 
                {
                  erase[i * width + j] = true;
                  changed = true;
                }
              }
            }

            if (!changed) break;

            for (i = 0; i < height; ++i)
              for (j = 0; j < width; ++j)
                if (erase[i * width + j])
                {
                  now[i * width + j] = false;
                  erase[i * width + j] = false;
                }
      }

      for (i = 0; i < height; ++i)
        for (j = 0; j < width; ++j)
          if (now[i * width + j])
            retdata[i * step + j] = 255;
          else
            retdata[i * step + j] = 0;
      return ret;
  }

  ////////////////////////////////////////////////////////////////
  // 给定二值化后的骨架，抽取Junction 特征
  ////////////////////////////////////////////////////////////////
  void extractJunctionFeature(IplImage * binary, int & r1, int & r2)
  {
    r1 = 0;
    r2 = 0;
    int height = binary->height;
    int width = binary->width;
    int step = binary->widthStep;
    unsigned char * data = (unsigned char *) binary->imageData;

    int i, j;
    for (i = 1; i < height - 1; ++i)
      for (j = 1; j < width - 1; ++j)
        if (data[i * step + j] != 0)
        {
          int neighs = 0;
          for (int l = -1; l <= 1; ++l)
            for (int m = -1; m <= 1; ++m)
              if (data[(i + l) * step + j + m] != 0)
                neighs++;
          if (neighs == 2) 
            r1++;
          if (neighs == 4) 
            r2++;
        }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////
  // 给定输入的CvMat，获得他的图像表示
  ////////////////////////////////////////////////////////////////////////////////////////////
  IplImage * getMatImage(CvMat * in)
  {
    double max = -1e50;
    double min = 1e50;
    double t;
    int i, j;
    for (i = 0; i < in->height; ++i)
      for (j = 0; j < in->width; ++j)
      {
        t = cvmGet(in, i, j);
        if (t > max) max=t;
        if (t < min) min=t;
      }

      double alpha= 255 / (max-min);

      IplImage * img = cvCreateImage(cvSize(in->width,in->height) ,8 ,3);
      unsigned char * data = (unsigned char *) img->imageData;
      int step = img->widthStep;

      double output;
      for (i = 0; i < in->height; ++i)
      {
        for (j = 0; j < in->width; ++j)
        {
          output = (cvmGet(in,i,j)-min) * alpha;
          data[i * step + j * 3 + 0] = (unsigned char)output;
          data[i * step + j * 3 + 1] = (unsigned char)output;
          data[i * step + j * 3 + 2] = (unsigned char)output;
        }
      }
      return img;
  }

#ifndef DSP
  ////////////////////////////////////////////////////
  // 给定二值化图像，返回HoughLine检测的结果
  ////////////////////////////////////////////////////
  IplImage * getHoughLines(IplImage * img, vector<pair<CvPoint, CvPoint> > & ret)
  {
    IplImage* color_dst = cvCreateImage( cvGetSize(img), 8, 3 );
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* lines = 0;
    int i;
    cvCvtColor(img, color_dst, CV_GRAY2BGR );

    lines = cvHoughLines2(img,
      storage,
      CV_HOUGH_PROBABILISTIC,
      1,
      CV_PI/180,
      80,
      30,
      10 );
    for( i = 0; i < lines->total; i++ )
    {
      CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
      cvLine( color_dst, line[0], line[1], CV_RGB(255,0,0), 3, 8 );
      ret.push_back(make_pair(line[0], line[1]));
    }
    return color_dst;
  }

  ///////////////////////////////////////////////////////////////////
  // 给定两张灰度图片，返回绝对差图片 
  ///////////////////////////////////////////////////////////////////
  IplImage * getDeltaImage(IplImage * img1, IplImage * img2)
  {
    IplImage * grayDiff = cvCreateImage(cvGetSize(img1), 8, 1);
    cvAbsDiff(img1, img2, grayDiff);

    return grayDiff;
  }


  ///////////////////////////////////////////////////////
  // 给定图片，返回Canny 边缘检测的结果
  ///////////////////////////////////////////////////////
  IplImage * getCannyDetection(IplImage * img)
  {
    IplImage * gray = cvCreateImage(cvGetSize(img), 8, 1);
    cvCvtColor(img, gray, CV_BGR2GRAY);
    IplImage * canny = cvCreateImage(cvGetSize(img), 8, 1);
    cvCanny(gray, canny, 50, 150);

    cvReleaseImage(&gray);
    return canny;
  }


#endif

#ifndef DSP
  ///////////////////////////////////////////////////////
  // 给定图片，返回Sobel 边缘检测的结果
  // 对于建筑物特别适用
  ///////////////////////////////////////////////////////
  IplImage * getSobelDetection(IplImage * img)
  {
    IplImage * gray = cvCreateImage(cvGetSize(img), 8, 1);
    cvCvtColor(img, gray, CV_BGR2GRAY);

    int height = img->height;
    int width = img->width;

    IplImage * xSobel = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);
    IplImage * ySobel = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);
    cvSobel(gray, xSobel, 1, 0, 3);
    cvSobel(gray, ySobel, 0, 1, 3);
    IplImage *	 sobel = cvCreateImage(cvGetSize(img), 8, 1);

    float * xSobelFloat= (float *) xSobel->imageData;
    float * ySobelFloat = (float *) ySobel->imageData;
    unsigned char * sobelData = (unsigned char *) sobel->imageData;
    for (int i = 0; i < height; ++i)
      for (int j = 0; j < width; ++j)
      {
        float eng1 = xSobelFloat[i * width + j];
        float eng2 = ySobelFloat[i * width + j];
        float eng = eng1 * eng1 + eng2 * eng2;
        eng = sqrt(eng);
        int value = eng;
        if (value > 255) value = 255;
        sobelData[i * sobel->widthStep + j] = value;
      }

      IplImage * ret = cvCreateImage(cvGetSize(img), 8, 1);
      cvThreshold(sobel, ret, 150, 255, CV_THRESH_BINARY);
      cvReleaseImage(&gray);
      cvReleaseImage(&xSobel);
      cvReleaseImage(&ySobel);
      cvReleaseImage(&sobel);
      return ret;
  }


#endif

  ////////////////////////////////////////////////////
  // 给定图片，将其亮度增强
  ////////////////////////////////////////////////////
  void brightnessEnhance(IplImage* srcImg, int brightness)
  {
    int val;

    unsigned char * data = (unsigned char *) srcImg->imageData;
    int step = srcImg->widthStep;
    int upper = srcImg->height * srcImg->width * 3;
    for (int off = 0; off < upper; ++off)
    {
      val = data[off];
      val += brightness;
      if (val > 255) val = 255;
      if (val < 0) val = 0;
      data[off] = val;
    }
  }

  void brightnessEnhance(IplImage* srcImg, double ratio)
  {
    int val;

    unsigned char * data = (unsigned char *) srcImg->imageData;
    int step = srcImg->widthStep;
    int upper = srcImg->height * srcImg->width * 3;
    for (int off = 0; off < upper; ++off)
    {
      val = data[off];
      val = val * ratio;
      if (val > 255) val = 255;
      if (val < 0) val = 0;
      data[off] = val;
    }
  }

  void grayEnhance(IplImage * gray, int delta)
  {
    for (int i = 0; i < gray->height; ++i)
      for (int j = 0; j < gray->width; ++j)
      {
        int c = ((unsigned char *) gray->imageData)[i * gray->widthStep + j];
        c += delta;
        if (c > 255) c = 255; if (c < 0) c = 0;
        ((unsigned char *) gray->imageData)[i * gray->widthStep + j] = c;
      }
  }

#ifndef DSP
  ////////////////////////////////////////////////////////////
  // 给定图片，将其中所有满足特定值的像素坐标全部取出
  ////////////////////////////////////////////////////////////
  vector<CvPoint> extractPointsGivenColor(IplImage * img, const CvScalar & color)
  {
    vector<CvPoint> ret;

    for (int i = 0; i < img->height; ++i)
      for (int j = 0; j < img->width; ++j)
        if (cvGet2D(img, i, j).val[0] == color.val[0] && 
          cvGet2D(img, i, j).val[1] == color.val[1] && 
          cvGet2D(img, i, j).val[2] == color.val[2])
          ret.push_back(cvPoint(j, i));

    return ret;
  }

  //////////////////////////////////////////////////
  // 将多张图像并成一行显示
  //////////////////////////////////////////////////
  IplImage * combineImageRow(vector<IplImage *> images)
  {
    int height = 0;
    int width = 0;
    int i;
    for (i = 0; i < images.size(); ++i) {
      width += images[i]->width;
      if (images[i]->height > height) height = images[i]->height;
    }
    if (width <= 200) width = 200;
    IplImage * bigImage = cvCreateImage(cvSize(width,height), 8, 3);

    int lastWidth = 0;
    for (i = 0; i < images.size(); ++i) {
      for (int x = 0; x < images[i]->height; ++x)
        for (int y = 0; y < images[i]->width; ++y){
          cvSet2D(bigImage, x, y + lastWidth, cvGet2D(images[i],x,y));
        }
        lastWidth += images[i]->width;
    }
    return bigImage;
  }
#endif

  /////////////////////////////////////////////////////
  // 灰度图直方图归一化
  /////////////////////////////////////////////////////
  int grayContrastEnhance(IplImage * src)
  {
    float p[256];
    int i, j, x, y, k;

    memset(p, 0, sizeof(p));

    int height = src->height;
    int width = src->width;
    float wMulh = height * width;

    unsigned char * data = (unsigned char * ) src->imageData;
    int step = src->widthStep;

    for (i = 0; i < height; ++i)
      for (j = 0; j < width; ++j)
      {
        int v = data[i * step + j];
        p[v]++;
      }

      //求存放图像各个灰度级的出现概率
      for (i = 0; i < 256; i++)
        p[i] = p[i] / wMulh;

      //求存放各个灰度级之前的概率和
      for (i = 1; i < 256; i++)
        p[i] = p[i - 1] + p[i];

      //直方图变换
      for (i = 0; i < height; ++i)
        for (j = 0; j < width; ++j)
        {
          int v = data[i * step + j];
          v = p[v] * 255 + 0.5;
          if (v > 255) v = 255; 
          data[i * step + j] = v;
        }

        return 1;
  }

  ///////////////////////////////////////////////////////////////
  // 增强图像的对比度，直方图均一划
  ///////////////////////////////////////////////////////////////
  int contrastEnhance(IplImage * src)
  {
    float p[256][3];
    int i, j, x, y, k;

    memset(p, 0, sizeof(p));

    int height = src->height;
    int width = src->width;
    float wMulh = height * width;

    unsigned char * data = (unsigned char * ) src->imageData;
    int step = src->widthStep;

    for (i = 0; i < height; ++i)
      for (j = 0; j < width; ++j)
        for (k = 0; k < 3; ++k)
        {
          int v = data[i * step + j * 3 + k];
          p[v][k]++;
        }

        //求存放图像各个灰度级的出现概率
        for (i = 0; i < 256; i++)
          for (k = 0; k < 3; ++k)
            p[i][k] = p[i][k] / wMulh;

        //求存放各个灰度级之前的概率和
        for (i = 1; i < 256; i++)
          for (k = 0; k < 3; ++k)
            p[i][k] = p[i - 1][k] + p[i][k];

        //直方图变换
        for (i = 0; i < height; ++i)
          for (j = 0; j < width; ++j)
            for (k = 0; k < 3; ++k)
            {
              int v = data[i * step + j * 3 + k];
              v = p[v][k] * 255 + 0.5;
              if (v > 255) v = 255; 
              data[i * step + j * 3 + k] = v;
            }

            return 1;
  }

  // 获得矩形的面积
  inline int getRectArea(const CvRect & rect)
  {
    return rect.height * rect.width;
  }

  inline void refineRect(CvRect & rect, IplImage * img)
  {
    int sx = rect.x;
    int ex = rect.x + rect.width - 1;
    int sy = rect.y;
    int ey = rect.y + rect.height - 1;
    if (sx < 0) sx = 0;
    if (ex >= img->width) ex = img->width - 1;
    if (sy < 0) sy = 0;
    if (ey >= img->height) ey = img->height - 1;
    rect.x = sx; rect.y = sy;
    rect.width = ex - sx + 1;
    rect.height = ey - sy + 1;
  }

  ////////////////////////////////////////////////////////
  // 给定输入灰度图片，获得他的0-1二值化表示, 
  // OpenCV自适应二值化的DSP实现
  ////////////////////////////////////////////////////////
  int sum[400][400];
  void getBinaryImage(IplImage * img, int blocksize, int param, IplImage * ret)
  {
    unsigned char * data = (unsigned char *) img->imageData;
    int step = img->widthStep;
    unsigned char * retdata = (unsigned char *) ret->imageData;
    int stepret = ret->widthStep;

    memset(sum, 0, sizeof(sum));
    int i, j;
    for (j = 2; j <= ret->width; ++j)
      sum[1][j] = sum[1][j - 1] + data[(j - 1)];

    for (i = 2; i <= ret->height; ++i)
    {
      int nowsum = 0;
      for (int j = 1; j <= ret->width; ++j)
      {
        nowsum += data[(i - 1) * step + (j - 1)];
        sum[i][j] = sum[i - 1][j] + nowsum;
      }
    }

    //cout << sum[img->height][img->width] << endl;

    int w = blocksize / 2;
    for (i = 1; i <= ret->height; ++i)
      for (j = 1; j <= ret->width; ++j)
      {
        int sx = max(1, i - w);
        int ex = min(ret->height, i + w);
        int sy = max(1, j - w);
        int ey = min(ret->width, j + w);

        int average = sum[ex][ey] - sum[sx - 1][ey] - sum[ex][sy - 1] + sum[sx - 1][sy - 1];
        int count = (ex - sx + 1) * (ey - sy + 1);
        int now = data[(i - 1) * step + (j - 1)];

        average /= count;
        average -= param;

        //cout << average << " " << count << " " << now << endl;
        //system("pause");

        if (now > average)
          retdata[(i - 1) * stepret + j - 1] = 255;
        else
          retdata[(i - 1) * stepret + j - 1] = 0;
      }
  }

#ifndef DSP

  ////////////////////////////////////////////////////////////////////
  // 获得直方图的图像表示
  ////////////////////////////////////////////////////////////////////
  IplImage * getHistogramImage(vector<double> & his)
  {
    int i;
    vector<double> nor = his;
    double sum = 0;
    for (i = 0; i < his.size(); ++i)
      sum += his[i];

    double maxh = 0;
    for (i = 0; i < nor.size(); ++i)
    {
      nor[i] /= sum;
      maxh = max(maxh, nor[i]);
    }

    IplImage * img = cvCreateImage(cvSize(nor.size(), 200), 8, 3);
    cvZero(img);

    for (i = 0; i < nor.size(); ++i)
    {
      int h = nor[i] / (maxh * 2) * 200;
      cvLine(img, cvPoint(i, 200), cvPoint(i, 200 - h), CV_RGB(255, 0, 0), 1);
    }
    cvLine(img, cvPoint(0, 50), cvPoint(his.size() - 1, 50), CV_RGB(0, 255, 0), 1);
    return img;
  }

  ////////////////////////////////////////////////////////////////////
  // 获得图像中的车灯信息
  ////////////////////////////////////////////////////////////////////
  void getLightInImage(IplImage * img, IplImage * light)
  {
    IplImage * org = (IplImage *) cvClone(img);
    brightnessEnhance(org, 0.125);

    cvNamedWindow("org");
    cvShowImage("org", org);

    int night_thres = 22;
    int height = img->height;
    int width = img->width;
    unsigned char * nowLightImageData = (unsigned char *) org->imageData;
    unsigned char * isLightImageData = (unsigned char *) light->imageData;
    int step = org->widthStep;
    int binarystep = light->widthStep;

    // 在晚上
    for (int i = 0; i < height; ++i)
      for (int j = 0; j < width; ++j)
      {
        int b = nowLightImageData[i * step + j * 3];
        int g = nowLightImageData[i * step + j * 3 + 1];
        int r = nowLightImageData[i * step + j * 3 + 2];
        isLightImageData[i * binarystep + j] = 0;
        if (i <= 3 || j <= 3 || i >= height - 3 || j >= width - 3) continue;
        if (r >= 21) isLightImageData[i * binarystep + j] = 255;
      }

      cvReleaseImage(&org);
  }

  void getIntegral(IplImage * gray, vector<vector<int> > & integ)
  {
    int i, j;
    int H = gray->height;
    int W = gray->width;
    unsigned char * data = (unsigned char *) gray->imageData;
    int step = gray->widthStep;

    integ.resize(H);
    for (i = 0; i < H; ++i)
      integ[i].resize(W);

    integ[0][0] = data[0];
    for (j = 1; j < W; ++j)
      integ[0][j] = integ[0][j - 1] + data[j];

    for (i = 1; i < H; ++i)
    {
      int nowsum = 0;
      for (j = 0; j < W; ++j)
      {
        nowsum += data[i * step + j];
        integ[i][j] = integ[i - 1][j] + nowsum;
      }
    }
  }

  inline int getIntegral_Point(vector<vector<int> > & integ, int x, int y)
  {
    if (x < 0 || y < 0) return 0;
    if (x > integ.size() - 1) x = integ.size() - 1;
    if (y > integ[0].size() - 1) y = integ[0].size() - 1; 
    return integ[x][y];
  }

  inline int getIntegral_Region(vector<vector<int> > & integ, int x1, int x2, int y1, int y2)
  {
    int sum1 = getIntegral_Point(integ, x2, y2);
    int sum2 = getIntegral_Point(integ, x2, y1 - 1);
    int sum3 = getIntegral_Point(integ, x1 - 1, y2);
    int sum4 = getIntegral_Point(integ, x1 - 1, y1 - 1);
    return sum1 - sum2 - sum3 + sum4;
  }



#endif

  void compressAVI(const string & str, const string & avifile)
  {
    /**
    FileFinder finder(str);
    CvVideoWriter * writer = NULL;
    bool first = true;
    int cnt = 0;
    while (finder.hasNext())
    {
    string file = finder.next();
    if (file.find("jpg") == string::npos) continue;
    cout << file << endl;
    IplImage * img = cvLoadImage(file.data());
    if (first)
    {
    first = false;
    writer = cvCreateVideoWriter(avifile.data(), -1, 12, cvGetSize(img));
    }
    int ret = cvWriteFrame(writer, img);
    cvReleaseImage(&img);
    //cnt++;
    //if (cnt == 100) break;
    }
    if (writer != NULL) cvReleaseVideoWriter(&writer);
    */
  }

  IplImage * readImageFromBinaryRGB(const string & file, int height, int width)
  {
    IplImage * img = cvCreateImage(cvSize(width, height), 8, 3);
    unsigned char * data = (unsigned char *) img->imageData;
    int step = img->widthStep;

    FILE * fin = fopen(file.data(), "rb");

    for (int i = 0; i < height; ++i)
      for (int j = 0; j < width; ++j)
      {
        int r = getc(fin);
        int g = getc(fin);
        int b = getc(fin);
        //if (r == EOF) 
        //	system("pause");
        data[i * step + j * 3 + 2] = r;
        data[i * step + j * 3 + 1] = g;
        data[i * step + j * 3 + 0] = b;
      }

      fclose(fin);
      return img;
  }

}

ostream & operator << (ostream & out, const CvPoint & p)
{
  out << "(" << p.x << "," << p.y << ")";
  return out;
}

ostream & operator << (ostream & out, const CvRect & rect)
{
  out << "(" << rect.x << "," << rect.y << ") " << rect.width << " " << rect.height;
  return out;
}

///////////////////////////////////////////////
// 扩展OPENCV GUI 模块
///////////////////////////////////////////////
#ifndef DSP
#include "Project_GiavalTracker.h"
#include "Util.h"
//#include "highgui.h"

void cvRectangle(IplImage * img, const CvRect & rect, const CvScalar & color, int w)
{
  CvPoint left = cvPoint(rect.x, rect.y);
  CvPoint right = cvPoint(rect.x + rect.width - 1, rect.y + rect.height - 1);
  cvRectangle(img, left, right, color, w);
}

void cvRectangle(IplImage * img, const GTRect & rect, const CvScalar & color, int w)
{
  CvPoint left = cvPoint(rect.left, rect.top);
  CvPoint right = cvPoint(rect.right, rect.bottom);
  cvRectangle(img, left, right, color, w);
}

IplImage * cvLoadImage(string str) { return cvLoadImage((const char *) str.data()); }

map<string, IplImage*> showdimages;
void cvShowImage(const string & windowname, IplImage * img, int width, int height, int x, int y)
{
  IplImage * cimg = cvCloneImage(img);
  if (showdimages.find(windowname) != showdimages.end())
    cvReleaseImage(&showdimages[windowname]);
  showdimages[windowname] = cimg;

  cvNamedWindow(windowname.data());
  if (width == -1 || height == -1) { width = img->width; height = img->height; }
  cvResizeWindow(windowname.data(), width, height);
  cvMoveWindow(windowname.data(), x + 64, y);
  IplImage * simg = cvCreateImage(cvSize(width, height), 8, img->nChannels);
  cvResize(img, simg);
  cvShowImage(windowname.data(), simg);
  cvReleaseImage(&simg);
}

void cvOutputShowdImage(const string & path)
{
  Util::mkdir(path);
  for (map<string, IplImage *>::iterator itr = showdimages.begin(); itr != showdimages.end(); ++itr)
  {
    string file = path + string(1, DirectoryChar) + itr->first + ".jpg";
    cvSaveImage(file.data(), itr->second);
    //cvReleaseImage(&itr->second);
  }
}

void cvText(IplImage* img, const char* text, int x, int y)
{
  CvFont font;

  double hscale = 1.0;
  double vscale = 1.0;
  int linewidth = 2;
  cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC,hscale,vscale,0,linewidth);

  CvScalar textColor =cvScalar(0,255,255);
  CvPoint textPos =cvPoint(x, y);

  cvPutText(img, text, textPos, &font,textColor);
}

void cvShowImage(const string & windowname, IplImage * img, bool autoArrange, bool clear)
{
  static const int SCREENWIDTH = 1500;
  static int nowx = 0;
  static int nowy = 0;
  static int nowmaxH = 0;
  if (clear) { nowx = 0; nowy = 0; nowmaxH = 0; }
  if (img == NULL) return;

  int height = img->height;
  int width = img->width;
  if (width + nowx > SCREENWIDTH && nowmaxH > 0)
  {
    nowy += nowmaxH + 32;
    nowx = 0; 
    nowmaxH = 0;
  }

  cvShowImage(windowname, img, width, height, nowx, nowy);
  nowx += width + 32;
  nowmaxH = max(nowmaxH, height);
}

void loadAndShowImages(const string & path, const string & prefix)
{
  FileFinder finder(path);
  while (finder.hasNext())
  {
    string file = finder.next();
    string windowname = prefix + Util::getFileTrueName(file);
    IplImage * img = cvLoadImage(file);
    cvShowImage(windowname, img, true, false);
    cvReleaseImage(&img);
  }
  cvWaitKey(0);
}


void loadAndMergeShowImages(const string & path, const string & windowname)
{
  const int SCREEN_WIDTH = 1300;
  const int SCREEN_HEIGHT = 800;

  vector<IplImage *> srcs;
  FileFinder finder(path);
  while (finder.hasNext())
  {
    string file = finder.next();
    IplImage * img = cvLoadImage(file);
    srcs.push_back(img);
  }

  IplImage * merge = CvExt::combineImageRowCol(srcs, SCREEN_WIDTH);
  cvShowImage(windowname, merge, SCREEN_WIDTH, min(SCREEN_HEIGHT, merge->height), 0, 0);
  cvWaitKey(0);

  for (int i = 0; i < srcs.size(); ++i)
    cvReleaseImage(&srcs[i]);
  cvReleaseImage(&merge);
}

void cvDestroyAllWindows(bool destroy)
{
  for (map<string, IplImage *>::iterator itr = showdimages.begin(); itr != showdimages.end(); ++itr)
    if (itr->second != NULL)
      cvReleaseImage(&itr->second);
  cvDestroyAllWindows();

  cvShowImage("", NULL, true, true);
}

map<string, vector<IplImage *> > blockImages;
void addDebugBlock(const string & name, IplImage * img)
{
  IplImage * block = cvCreateImage(cvSize(64, 64), 8, 3);
  cvResize(img, block);

  blockImages[name].push_back(block);
  //cvReleaseImage(&img);
}

void showDebugBlocks()
{
  for (map<string, vector<IplImage *> >::iterator itr = blockImages.begin(); itr != blockImages.end(); ++itr)
  {
    IplImage * combineImage = CvExt::combineImageRow(itr->second);
    cvShowImage(itr->first, combineImage, true, false);

    cvReleaseImage(&combineImage);
    for (int i = 0; i < itr->second.size(); ++i)
      cvReleaseImage(&itr->second[i]);
  }
  blockImages.clear();

  cvShowImage("NULL", NULL, true, true);
}

////////////////////////////////////////////////////////////////////
// 图像缩略图拼接算法，YYU50
////////////////////////////////////////////////////////////////////

const int BB = 5;
const int BLOCK_IMAGE_H = 75;
const int BLOCK_IMAGE_W = 75;
const int RET_IMAGE_H = 900;
const int RET_IMAGE_W = 600;
const int RET_IMAGE_BLOCK_H = 900 / 75;
const int RET_IMAGE_BLOCK_W = 600 / 75;

IplImage * combineImageFromSmall(vector<IplImage *> images, IplImage * srcImage, double alpha = 0.65)
{
  random_shuffle(images.begin(), images.end());
  vector<IplImage *> newimages = images;
  while (newimages.size() < RET_IMAGE_BLOCK_H * RET_IMAGE_BLOCK_W)
  {
    random_shuffle(images.begin(), images.end());
    newimages.insert(newimages.end(), images.begin(), images.end());
  }
  images = newimages;

  int height = RET_IMAGE_H;
  int width = RET_IMAGE_W;
  IplImage * img = cvCreateImage(cvSize(width, height), 8, 3);
  cvResize(srcImage, img);
    
  unsigned char * data = (unsigned char *) img->imageData;
  int step = img->widthStep;
  int averr[BB][BB], averg[BB][BB], averb[BB][BB];

  for (int i = 0; i < RET_IMAGE_BLOCK_H; ++i)
  {
    for (int j = 0; j < RET_IMAGE_BLOCK_W; ++j)
    {
      int off = (i * RET_IMAGE_BLOCK_W + j) % images.size();
      IplImage * smallimg = cvCreateImage(cvSize(BLOCK_IMAGE_H, BLOCK_IMAGE_W), 8, 3);
      cvResize(images[off], smallimg);

      unsigned char * smalldata = (unsigned char *) smallimg->imageData;
      int smallstep = smallimg->widthStep;
      int sx = i * BLOCK_IMAGE_H;
      int sy = j * BLOCK_IMAGE_W;
      int ex = sx + BLOCK_IMAGE_H - 1;
      int ey = sy + BLOCK_IMAGE_W - 1;

      
      int xsize = BLOCK_IMAGE_H / BB;
      int ysize = BLOCK_IMAGE_W / BB;

      memset(averr, 0, sizeof(averr));
      memset(averg, 0, sizeof(averg));
      memset(averb, 0, sizeof(averb));
      int blockcnt = BLOCK_IMAGE_H * BLOCK_IMAGE_W / BB / BB;

      for (int x = sx; x <= ex; ++x)
      {
        for (int y = sy; y <= ey; ++y)
        {
          int r = data[(x) * step + (y) * 3 + 2];
          int g = data[(x) * step + (y) * 3 + 1];
          int b = data[(x) * step + (y) * 3 + 0];
          int xb = (x - sx) / xsize;
          int yb = (y - sy) / ysize;
          averr[xb][yb] += r;
          averg[xb][yb] += g;
          averb[xb][yb] += b;
        }
      }

      for (int i = 0; i < BB; ++i)
      {
        for (int j = 0; j < BB; ++j)
        {
          averr[i][j] /= blockcnt;
          averg[i][j] /= blockcnt;
          averb[i][j] /= blockcnt;
        }
      }

      for (int x = 0; x < BLOCK_IMAGE_H; ++x)
      {
        for (int y = 0; y < BLOCK_IMAGE_W; ++y)
        {
          double nowalpha = alpha;
          if (x == 0 || y == 0 || x == BLOCK_IMAGE_H - 1 || y == BLOCK_IMAGE_W - 1) nowalpha = 1.0;

          int r1 = smalldata[x * smallstep + y * 3 + 2];
          int g1 = smalldata[x * smallstep + y * 3 + 1];
          int b1 = smalldata[x * smallstep + y * 3 + 0];

          int r2 = data[(sx + x) * step + (sy + y) * 3 + 2];
          int g2 = data[(sx + x) * step + (sy + y) * 3 + 1];
          int b2 = data[(sx + x) * step + (sy + y) * 3 + 0];

          int xb = (x) / xsize;
          int yb = (y) / ysize;

          int r = r1 * (1 - nowalpha) + averr[xb][yb] * nowalpha;
          int g = g1 * (1 - nowalpha) + averg[xb][yb] * nowalpha;
          int b = b1 * (1 - nowalpha) + averb[xb][yb] * nowalpha;

          data[(sx + x) * step + (sy + y) * 3 + 2] = r;
          data[(sx + x) * step + (sy + y) * 3 + 1] = g;
          data[(sx + x) * step + (sy + y) * 3 + 0] = b;
        }
      }
    }
  }
  return img;
}

/**
// NOTE you SHOULD cvReleaseImage() for the return value when end of the code.
- (IplImage *)CreateIplImageFromUIImage:(UIImage *)image {
  // Getting CGImage from UIImage
  CGImageRef imageRef = image.CGImage;

  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
  // Creating temporal IplImage for drawing
  IplImage *iplimage = cvCreateImage(
    cvSize(image.size.width,image.size.height), IPL_DEPTH_8U, 4
  );
  // Creating CGContext for temporal IplImage
  CGContextRef contextRef = CGBitmapContextCreate(
    iplimage->imageData, iplimage->width, iplimage->height,
    iplimage->depth, iplimage->widthStep,
    colorSpace, kCGImageAlphaPremultipliedLast|kCGBitmapByteOrderDefault
  );
  // Drawing CGImage to CGContext
  CGContextDrawImage(
    contextRef,
    CGRectMake(0, 0, image.size.width, image.size.height),
    imageRef
  );
  CGContextRelease(contextRef);
  CGColorSpaceRelease(colorSpace);

  // Creating result IplImage
  IplImage *ret = cvCreateImage(cvGetSize(iplimage), IPL_DEPTH_8U, 3);
  cvCvtColor(iplimage, ret, CV_RGBA2BGR);
  cvReleaseImage(&iplimage);

  return ret;
}

// NOTE You should convert color mode as RGB before passing to this function
- (UIImage *)UIImageFromIplImage:(IplImage *)image {
  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
  // Allocating the buffer for CGImage
  NSData *data =
    [NSData dataWithBytes:image->imageData length:image->imageSize];
  CGDataProviderRef provider =
    CGDataProviderCreateWithCFData((CFDataRef)data);
  // Creating CGImage from chunk of IplImage
  CGImageRef imageRef = CGImageCreate(
    image->width, image->height,
    image->depth, image->depth * image->nChannels, image->widthStep,
    colorSpace, kCGImageAlphaNone|kCGBitmapByteOrderDefault,
    provider, NULL, false, kCGRenderingIntentDefault
  );
  // Getting UIImage from CGImage
  UIImage *ret = [UIImage imageWithCGImage:imageRef];
  CGImageRelease(imageRef);
  CGDataProviderRelease(provider);
  CGColorSpaceRelease(colorSpace);
  return ret;
}


UIImage combineImageFromSmall(UIImage * target, NSArray * images)
{
  vector<IplImage *> myimages;
  for (int i = 0; i < images.count; ++i)
    myimages.push_back(CreateIplImageFromUIImage(images.indexOfObject(i)));

  IplImage * mytarget = CreateIplImageFromUIImage(target);

  IplImage * ret = combineImageFromSmall(myimages, mytarget);

  UIImage retjy = UIImageFromIplImage(ret);

  cvReleaseIamge(&ret);
  cvReleaseImage(&mytarget);
  for (int i = 0; i < myimages.size(); ++i)
    cvReleaseImage(&myimages[i]);

  return retjy;
}
*/

void SetErrorLabe(IplImage *pImp, CvPoint p, int Labe)
{
  CvFont font;
  char sztext[10]={0};

  font = cvFont(1);
  sprintf(sztext,"%d",Labe);
  cvPutText(pImp,sztext,p,&font,cvScalar(0,0,255));
}

#endif

#endif