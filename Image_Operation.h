#ifndef IMAGE_OPERATION
#define IMAGE_OPERATION

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <cmath>
#include <vector>
#include <iostream>
using namespace std;

////////////////////////////////////////////////////////////////////////////
//函数名：getPix
//作用  ：返回给定图像的基于channel信道的，x,y坐标下的像素值
//参数  ：
//int     x                       距离图像起始点的高度
//int     y                       距离图像起始点的宽度
//int     channel                 信道——BGR默认
//                                单信道也有可能
//返回值：
//像素值
////////////////////////////////////////////////////////////////////////////
unsigned char getPix(int i, int j, int channel, IplImage * in)
{ 
  int step      = in->widthStep;  
  int channels  = in->nChannels;  
  uchar* data      = (uchar *)in->imageData;  
  //printf("Processing a %dx%d image with %d channels\n",height,width,channels); 

  return data[i*step+j*channels+channel];
}


////////////////////////////////////////////////////////////////////////////
//函数名：setPix
//作用  ：设置给定图像的基于channel信道的，x,y坐标下的像素值
//参数  ：
//int     x                       距离图像起始点的高度
//int     y                       距离图像起始点的宽度
//int     channel                 信道——BGR默认
//                                单信道也有可能
//像素值
////////////////////////////////////////////////////////////////////////////
unsigned char & setPix(int i, int j, int channel, IplImage * in)
{ 
  int step      = in->widthStep;  
  int channels  = in->nChannels;  
  uchar* data      = (uchar *)in->imageData;  
  //printf("Processing a %dx%d image with %d channels\n",height,width,channels); 

  return data[i*step+j*channels+channel];
}


inline double rgb2lab_F(double x)
{
  if (x>0.008856) return x/3;
  else return 7.787*x+16.0/116.0;
}

/** 将一个HSV空间下的颜色根据H分量转变成RGB下的颜色，用于直方图的显示 */
CvScalar hsv2rgb( float hue )
{
  int rgb[3], p, sector;
  static const int sector_data[][3]=
  {{0,2,1}, {1,2,0}, {1,0,2}, {2,0,1}, {2,1,0}, {0,1,2}};
  hue *= 0.033333333333333333333333333333333f;
  sector = cvFloor(hue);
  p = cvRound(255*(hue - sector));
  p ^= sector & 1 ? 255 : 0;

  rgb[sector_data[sector][0]] = 255;
  rgb[sector_data[sector][1]] = 0;
  rgb[sector_data[sector][2]] = p;

  return cvScalar(rgb[2], rgb[1], rgb[0],0);
}

////////////////////////////////////////////////////////////////////////////
//函数名：getHistogramImage
//作用  ：返回给定图像的HSV h-s 直方图分布图像
//				如果需要直方图的统计信息，请使用Region类提供的函数
//参数  ：IplImage * src 原始图像
////////////////////////////////////////////////////////////////////////////
IplImage * getHistogramImage(IplImage * src)
{
  IplImage* hsv = cvCreateImage( cvGetSize(src), 8, 3 );
  IplImage* h_plane = cvCreateImage( cvGetSize(src), 8, 1 );
  IplImage* s_plane = cvCreateImage( cvGetSize(src), 8, 1 );
  IplImage* v_plane = cvCreateImage( cvGetSize(src), 8, 1 );
  IplImage* planes[] = { h_plane, s_plane };

  /** H 分量划分为16个等级，S分量划分为8个等级 */
  int h_bins = 16, s_bins = 8;
  int hist_size[] = {h_bins, s_bins};

  /** H 分量的变化范围 */
  float h_ranges[] = { 0, 180 }; 

  /** S 分量的变化范围*/
  float s_ranges[] = { 0, 255 };
  float* ranges[] = { h_ranges, s_ranges };

  /** 输入图像转换到HSV颜色空间 */
  cvCvtColor( src, hsv, CV_BGR2HSV );
  cvCvtPixToPlane( hsv, h_plane, s_plane, v_plane, 0 );

  /** 创建直方图，二维, 每个维度上均分 */
  CvHistogram * hist = cvCreateHist( 2, hist_size, CV_HIST_ARRAY, ranges, 1 );
  /** 根据H,S两个平面数据统计直方图 */
  cvCalcHist( planes, hist, 0, 0 );

  /** 获取直方图统计的最大值，用于动态显示直方图 */
  float max_value;
  cvGetMinMaxHistValue( hist, 0, &max_value, 0, 0 );


  /** 设置直方图显示图像 */
  int height = 240;
  int width = (h_bins*s_bins*6);
  IplImage* hist_img = cvCreateImage( cvSize(width,height), 8, 3 );
  cvZero( hist_img );

  /** 用来进行HSV到RGB颜色转换的临时单位图像 */
  IplImage * hsv_color = cvCreateImage(cvSize(1,1),8,3);
  IplImage * rgb_color = cvCreateImage(cvSize(1,1),8,3);
  int bin_w = width / (h_bins * s_bins);
  for(int h = 0; h < h_bins; h++)
  {
    for(int s = 0; s < s_bins; s++)
    {
      int i = h*s_bins + s;
      /** 获得直方图中的统计次数，计算显示在图像中的高度 */
      float bin_val = cvQueryHistValue_2D( hist, h, s );
      int intensity = cvRound(bin_val*height/max_value);

      /** 获得当前直方图代表的颜色，转换成RGB用于绘制 */
      cvSet2D(hsv_color,0,0,cvScalar(h*180.f / h_bins,s*255.f/s_bins,255,0));
      cvCvtColor(hsv_color,rgb_color,CV_HSV2BGR);
      CvScalar color = cvGet2D(rgb_color,0,0);

      cvRectangle( hist_img, cvPoint(i*bin_w,height),
          cvPoint((i+1)*bin_w,height - intensity),
          color, -1, 8, 0 );
    }
  }
  return hist_img;
}


///////////////////////////////////////////////////////////////////
// 获得给定图像的一个super-pixel，
// srcImage 为输入图像
// rowblocks 为图像每行划分成多少个superpixel
// rowpixelperblock 为图像每个superpixel，每行包含多少个像素
// x,y为super-pixel的坐标，x为行，y为列
///////////////////////////////////////////////////////////////////
IplImage * getSuperPixel(IplImage * srcImage,int rowblocks,int rowpixelperblock,int x,int y)
{
  int size = rowblocks * rowpixelperblock;
  IplImage * newImage = cvCreateImage(cvSize(size,size),8,3);
  if (srcImage->height != newImage->height || 
      srcImage->width != newImage->width) 
    cvResize(srcImage,newImage);
  else
    newImage = cvCloneImage(srcImage);

  IplImage * ret = cvCreateImage(cvSize(rowpixelperblock,rowpixelperblock),8,3);
  int startx = x * rowpixelperblock;
  int starty = y * rowpixelperblock;
  for (int i=0;i<ret->height;++i)
    for (int j=0;j<ret->width;++j)
    {
      setPix(i,j,2,ret) = getPix(startx+i,starty+j,2,newImage);
      setPix(i,j,1,ret) = getPix(startx+i,starty+j,1,newImage);
      setPix(i,j,0,ret) = getPix(startx+i,starty+j,0,newImage);
    }

  cvReleaseImage(&newImage);
  return ret;
}


////////////////////////////////////////////////////////////////////////////
// Get part of given image
////////////////////////////////////////////////////////////////////////////
IplImage * getPartialImage(IplImage * srcImage, CvRect rect)
{
  if (rect.x == 0 && rect.y == 0 && 
      rect.height == srcImage->height && rect.width == srcImage->width)
    return cvCloneImage(srcImage);
  IplImage * ret = cvCreateImage(cvSize(rect.width, rect.height), 8, 3);
  for (int i = 0; i < ret->height; ++i)
    for (int j = 0; j < ret->width; ++j)
      cvSet2D(ret, i, j, cvGet2D(srcImage, i + rect.x, j + rect.y));
  return ret;
}

///////////////////////////////////////////////////////////////////////////
// Set part of given image
///////////////////////////////////////////////////////////////////////////
void setPartialImage(IplImage * srcImage, IplImage * paintImage, int x, int y)
{
  int stopX = min(srcImage->height, paintImage->height + x);
  int stopY = min(srcImage->width, paintImage->width + y);
  for (int i = x; i < stopX; ++i)
    for (int j = y; j < stopY; ++j)
      cvSet2D(srcImage, i, j, cvGet2D(paintImage, i - x, j - y));
}

double getImageDistance(IplImage * a, IplImage * b)
{
  if (a->height != b->height || a->width != b->width) return 1e20;

  IplImage* hsv1 = cvCreateImage(cvGetSize(a), 8, 3);
  cvCvtColor(a, hsv1, CV_BGR2HSV);
  IplImage* h_plane1 = cvCreateImage(cvGetSize(hsv1), 8, 1);
  cvCvtPixToPlane(hsv1, h_plane1, NULL, NULL, 0);

  IplImage* hsv2 = cvCreateImage( cvGetSize(a), 8, 3 );
  cvCvtColor( b, hsv2, CV_BGR2HSV );
  IplImage* h_plane2 = cvCreateImage( cvGetSize(hsv2), 8, 1 );
  cvCvtPixToPlane( hsv2, h_plane2, NULL, NULL, 0 );

  double ret = 0;
  for (int i=0;i<h_plane1->height;++i)
    for (int j=0;j<h_plane1->width;++j)
      ret+= abs(cvGet2D(h_plane1,i,j).val[0] - cvGet2D(h_plane2,i,j).val[0]);

  cvReleaseImage(&hsv1);
  cvReleaseImage(&hsv2);
  cvReleaseImage(&h_plane1);
  cvReleaseImage(&h_plane2);
  return ret;
}

/////////////////////////////////////////////////////////////
// Combine multiple images , line per line 
/////////////////////////////////////////////////////////////
IplImage * combineImage(vector<IplImage *> images)
{
  int height = 0;
  int width = 0;
  for (int i = 0; i < images.size(); ++i) {
    height += images[i]->height;
    if (images[i]->width > width) width = images[i]->width;
  }
  IplImage * bigImage = cvCreateImage(cvSize(width,height),8,3);

  int lastHeight = 0;
  for (int i = 0; i < images.size(); ++i) {
    for (int x = 0; x < images[i]->height; ++x)
      for (int y = 0; y < images[i]->width; ++y){
        cvSet2D(bigImage,lastHeight+x,y,cvGet2D(images[i],x,y));
      }
    lastHeight += images[i]->height;
  }
  return bigImage;
}

//////////////////////////////////////////////////
// Combine multiple images , row per row
//////////////////////////////////////////////////
IplImage * combineImageRow(vector<IplImage *> images)
{
  int height = 0;
  int width = 0;
  for (int i = 0; i < images.size(); ++i) {
    width += images[i]->width;
    if (images[i]->height > height) height = images[i]->height;
  }
  IplImage * bigImage = cvCreateImage(cvSize(width,height),8,3);

  int lastWidth = 0;
  for (int i = 0; i < images.size(); ++i) {
    for (int x = 0; x < images[i]->height; ++x)
      for (int y = 0; y < images[i]->width; ++y){
        cvSet2D(bigImage, x, y + lastWidth, cvGet2D(images[i],x,y));
      }
    lastWidth += images[i]->width;
  }
  return bigImage;
}

////////////////////////////////////////////////////
// Get Image SuperPixel Visual Image(For paper)
////////////////////////////////////////////////////
IplImage * getSplitImage(IplImage * img) 
{
  IplImage * splitImage = cvCreateImage(cvSize(img->width * 3 / 2, img->height * 3 / 2), 8, 3);
  for (int i = 0; i < splitImage->height; ++i) 
    for (int j = 0; j < splitImage->width; ++j) 
      cvSet2D(splitImage, i, j, cvScalar(255, 255, 255, 0));

  int blocks = img->height / 8;
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) {
      IplImage * smallImage = getPartialImage(img, cvRect(i * blocks, j * blocks, blocks, blocks));
      setPartialImage(splitImage, smallImage, i * blocks * 3 / 2, j * blocks * 3 / 2);
      cvReleaseImage(&smallImage);
    }
  return splitImage;
}


#endif
