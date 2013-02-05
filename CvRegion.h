#ifndef CV_REGION_H
#define CV_REGION_H

#include "stdafx.h"
#include "geometry.h"
#include "CvImageOperationExt.h"

// make sure that h1 and h2 are normalized
double getBhattacharyyaDistance(vector<double> & h1, vector<double> & h2)
{
  if (h1.size() == 0) return 0;
  double ans = 0;
  for (int i = 0; i < h1.size(); ++i)
    ans += sqrt(h1[i] * h2[i]);
  return sqrt(1 - ans);
}

int cvRegion_convexHullBuffer[200000];


const int h_hist_bins = 30;

/** HOG 直方图 */
const int hog_hist_bins = 9;

class CvRegion
{
public:
  /** 区域内的点 */
  /** 区域内的点 .x 横坐标(宽度) .y 纵坐标(高度) */
  vector<CvPoint> points;

  /** 多边形凸包 */
  vector<CvPoint> poly;
  PigPolygon poly2;

  /** 最小矩形覆盖 */
  /** 最小矩形覆盖 .width 是高 .height是宽 */
#ifndef DSP
  CvBox2D box;
  CvPoint2D32f boxpt[4];
  CvPoint boxpti[4];
#endif

  /** 最小平行坐标轴矩形覆盖 */
  int minH; 
  int maxH;
  int minW;
  int maxW;
  int rectW;
  int rectH;

  /** 矩形区域 */
  CvRect rect;

  /** 面积 */
  double area;

  /** 周长 */
  double length;

  /** 像素点矩形面积比 */
  double ratio;
  double rectRatio;

  double diffRatio;

  /** 区域的邻居以及他们之间的边界点 */
  map<CvRegion *, vector<CvPoint> > boundarys;

  /** 区域的邻居以及他们边界上的能量 */
  map<CvRegion *, double> boundaryEnergyHeng;
  map<CvRegion *, double> boundaryEnergyShu;

  /** 平均RGB */
  double averRR;
  double averGG;
  double averBB;

  double averGray;
  double torGray;

  /** 平均色相，饱和度，色调 */
  double averH;
  double averS;
  double averV;

  /** 平均Lab 
  L =  L * 255 / 100; a =  a + 128; b =  b + 128
  */
  double averL;
  double averA;
  double averB;

  double score;

  CvPoint center;

  /** H 色调的直方图 */
  vector<double> h_hist;

  /** HOG 直方图 */
  /** from degree 0 to 180, 9 bins */
  vector<double> hog_hist;

  /** 原始图像 */
  IplImage * sourceImage;
  int srcHeight;
  int srcWidth;

  /** 区域内的边缘能量 */
  double totalEnergy;

  /** 区域内的平均边缘能量，能量方差 */
  double averEnergy;
  double torEnergy;

  /** 区域的额外信息，可设置成自己想要的数据 */
  int label;
  string sLabel;
  int tag;
  int filterTag;

  // 区域中横向有效线段个数
  int validHeng;
  double validHengRatio;

  // 区域中横向有效线段平均位置
  int validHengAverH;
  double validHengPosi;

  // 区域中竖向有效线段
  int validShu;
  int validShuAverW;
  double validShuPosi;
  double validShuRatio;

  // 区域中不在同一个连通分量内的其他像素点的比例
  double otherpointsRatio;
  // 区域中前景点的比例
  double forepointsRatio;

  // 横竖坐标的均值
  double yratio;
  double yratio2;
  double xratio, xratio2;

  // 区间是否经过合并
  bool hasmerge;

  // 是否是前景点还是车尾
  bool isFore;

  // 周围的邻居个数
  int neighbors;

  // 区域所在前景的连同分量的大小 
  int inForePoints;
  int inForeH;
  int inForeW;

  /** 构造函数 */
  CvRegion(int srcH, int srcW)
  {
    srcHeight = srcH;
    srcWidth = srcW;
    totalEnergy = 0;
    averGray = 0;
    torGray = 0;
    center.x = 0;
    center.y = 0;
    tag = 0;
    hasmerge = false;
    isFore = false;
    neighbors = 0;
    inForePoints = 0;
    inForeH = 0; 
    inForeW = 0;
  }

  CvRegion()
  {
    totalEnergy = 0;
    averGray = 0;
    torGray = 0;
    center.x = 0;
    center.y = 0;
    tag = 0;
    hasmerge = false;
    isFore = false;
    neighbors = 0;
    inForePoints = 0;
    inForeH = 0; 
    inForeW = 0;
  }


  /** 输出 */
  friend ostream & operator << (ostream & out, CvRegion & region)
  {
    out << "Ps = " << region.points.size() << " ";
    out << "" << region.center << " ";
    out << "" << "H = " << region.rectH << "[" << region.minH << "-" << region.maxH << "]" <<  " W = " << region.rectW << "[" << region.minW << "-" << region.maxW << "]";// << " MinRect = " << 
    return out;
  }

  double getDistance(CvRegion & other)
  {
    double dist = sqr(center.x - other.center.x) + sqr(maxH - other.maxH);
    dist = sqrt(dist);
    return dist;
  }

  /** 计算区域内点的特征 */
  //void calRegionFeature(IplImage * srcImage, 
  //  IplImage * gray)
  //{
  //  int k, i, j;
    
    // 计算平均RGB
    /**
    averRR = 0;
    averGG = 0;
    averBB = 0;
    if (srcImage != NULL)
    {
      unsigned char * imageData = (unsigned char *) srcImage->imageData;
      int step = srcImage->widthStep;

      for (k = 0; k < points.size(); ++k)
      {
        i = points[k].y;
        j = points[k].x;

        averRR += imageData[i * step + j * 3 + 2];
        averGG += imageData[i * step + j * 3 + 1];
        averBB += imageData[i * step + j * 3 + 0];
      }
      averRR /= points.size();
      averGG /= points.size();
      averBB /= points.size();
    }
    */

    // 计算能量等特征，中心坐标
    /**
    unsigned char * grayData = gray != NULL ? (unsigned char *) gray->imageData : NULL;
    int grayStep = 0;
    if (gray != NULL) grayStep = gray->widthStep;
    for (k = 0; k < points.size(); ++k)
    {
      i = points[k].y;
      j = points[k].x;
      if (gray != NULL) averGray += grayData[i * grayStep + j];
    }
    */

    // 计算平均灰度
    //averGray /= points.size();

    /** 区域内的平均能量 */
    /**
    averEnergy = totalEnergy / points.size();
    torEnergy = 0;
    torEnergy /= points.size();
    torEnergy = sqrt(torEnergy);

    // 计算灰度方差
    if (gray != NULL)
    {
      torGray = 0;
      for (k = 0; k < points.size(); ++k)
      {
        i = points[k].y;
        j = points[k].x;
        if (gray != NULL) 
          torGray += (grayData[i * grayStep + j] - averGray) * (grayData[i * grayStep + j] - averGray);
      }
      torGray /= points.size();
      torGray = sqrt(torGray);
    }
    */

  //  calRegionBasicFeature();
  //}

  void calRegionBasicFeature()
  {
    int i, j, k;

    /** 计算区域内的平行坐标轴矩形覆盖 */
    minH = 9999;
    maxH = -1;
    minW = 9999;
    maxW = -1;
    center.x = center.y = 0;
    for (k = 0; k < points.size(); ++k)
    {
      int nowW = points[k].x;
      int nowH = points[k].y;
      minH = min(minH, nowH);
      maxH = max(maxH, nowH);
      minW = min(minW, nowW);
      maxW = max(maxW, nowW);
      center.x += nowW;
      center.y += nowH;
    }

    rectW = maxW - minW + 1;
    rectH = maxH - minH + 1;
    area = rectW * rectH;
    center.x /= (int) points.size();
    center.y /= (int) points.size();

    ratio = (double) points.size() / rectH / rectW;

    /** 计算车辆矩形区域 */
    rect = cvRect(minW, maxH - rectW, rectW, rectW);

    // 如果车辆较大，则适当提高其高度
    if (rectW >= 40) rect.y -= 10, rect.height += 10;
    if (rectW >= 50) rect.y -= 5, rect.height += 5;

    rectRatio = (double) points.size() / rect.height / rect.width;

    validHeng = 0;
    validHengAverH = 0;
    validHengPosi = 0;
    vector<int> cnt(maxH + 1, 0);

    for (k = 0; k < points.size(); ++k)
    {
      i = points[k].y;
      j = points[k].x;
      cnt[i]++;
    }

    for (i = minH; i <= maxH; ++i)
    {
      if (cnt[i] >= rectW * 7 / 10) 
      {
        validHeng++;
        validHengAverH += i;
      }
    }

    validHengRatio = 0;
    if (rectH > 0) validHengRatio = (double) validHeng / rectH;
    if (validHeng > 0) validHengAverH /= validHeng;
    if (validHeng > 0) validHengPosi = (double) (validHengAverH - minH) / rectH; 

    validShu = 0;
    validShuAverW = 0;
    validShuPosi = 0;
    vector<int> cntW(maxW + 1, 0);

    for (k = 0; k < points.size(); ++k)
    {
      i = points[k].y;
      j = points[k].x;
      cntW[j]++;
    }

    for (i = minW; i <= maxW; ++i)
    {
      if (cntW[i] >= rectH * 7 / 10)
      {
        validShu++;
        validShuAverW += i;
      }
    }

    validShuRatio = 0;
    if (rectW > 0) validShuRatio = (double) validShu / rectW;
    if (validShu > 0) validShuAverW /= validShu;
    if (validShu > 0) validShuPosi = (double) (validShuAverW - minW) / rectW;

    yratio = ((double) (center.y - minH)) / (rectH);
    yratio2 = ((double) (center.y - rect.y)) / rect.height;
    xratio = ((double) (center.x - minW)) / (rectW);
    xratio2 = ((double) (center.x - rect.x)) / rect.width;
  }

  /** 获得区域内的点集 */
  vector<CvPoint> getPointsInConvexHull()
  {
    vector<CvPoint> ret;
    for (int i = 0; i < sourceImage->height; ++i)
      for (int j = 0; j < sourceImage->width; ++j)
      {
        Point now(j, i);
        if (poly2.inside(now) >= 0) 
          ret.push_back(cvPoint(j, i));
      }
      return ret;
  }

  //////////////////////////////////////////////////////////////
  // 绘图函数
  //////////////////////////////////////////////////////////////
#ifndef DSP

  void addToDebug(const string & reason, IplImage * src)
  {
    //IplImage * sub = CvExt::getSubImage(src, rect);
    //addDebugBlock(reason, sub);
    //cvReleaseImage(&sub);
    this->drawRegionText(src, CV_RGB(255, 0, 0), reason);
  }

  /** 绘制凸包 */
  void drawRegionConvexHull(IplImage * img)
  {
    for (unsigned int i = 0; i < poly.size(); ++i)
    {
      CvPoint & nowPoint = poly[i];
      CvPoint & nextPoint = poly[(i + 1) % poly.size()];
      cvLine(img, nowPoint, nextPoint, cvScalar(0, 0, 255), 1);
    }
  }

  /** 绘制凸包内的点 */
  void drawPointInConvexHull(IplImage * img)
  {
    for (int i = 0; i < sourceImage->height; ++i)
      for (int j = 0; j < sourceImage->width; ++j)
      {
        Point now(j, i);
        if (poly2.inside(now) >= 0) 
          cvSet2D(img, i, j, cvGet2D(sourceImage, i, j));
      }
  }

  /** 绘制最小矩形覆盖 */
  void drawRegionMinRect(IplImage * img)
  {
    for (int i = 0; i < 4; ++i)
    {
      CvPoint & nowPoint = boxpti[i];
      CvPoint & nextPoint = boxpti[(i + 1) % 4];
      cvLine(img, nowPoint, nextPoint, cvScalar(0, 0, 255), 1);
    }
  }

  /** 绘制区域的文字信息 */
  void drawRegionText(IplImage * img, CvScalar color, const string & text)
  {
    CvFont font = cvFont(1, 1);
    cvPutText(img, text.data(), cvPoint((minW + maxW) / 2, maxH - 5), &font, color);
  }

  /** 绘制区域特征到图像 */
  void drawRegionFeature(IplImage * img, int x, int y, CvScalar color = CV_RGB(0, 0, 0))
  {
    CvFont font = cvFont(1, 1);
    char temp[100];
    sprintf(temp, "w=%d,h=%d,cx=%d,cy=%d,hasmerge=%d", rectW, rectH, center.x, center.y, hasmerge);
    cvPutText(img, temp, cvPoint(x, y), &font, color);
    y += 20;

    sprintf(temp, "ps=%d,area=%.2lf", this->points.size(), this->area);
    cvPutText(img, temp, cvPoint(x, y), &font, color);
    y += 20;

    sprintf(temp, "vH=%d,vHP=%.2lf", this->validHeng, this->validHengPosi);
    cvPutText(img, temp, cvPoint(x, y), &font, color);
    y += 20;

    sprintf(temp, "vS=%d,vSP=%.2lf", this->validShu, this->validShuPosi);
    cvPutText(img, temp, cvPoint(x, y), &font, color);
    y += 20;

    sprintf(temp, "ratio=%.2lf, rectratio=%.2lf, diffratio=%.2lf", this->ratio, this->rectRatio, this->diffRatio);
    cvPutText(img, temp, cvPoint(x, y), &font, color);
    y += 20;

    sprintf(temp, "otherpointratio=%.2lf, fpr=%.2lf", this->otherpointsRatio, this->forepointsRatio);
    cvPutText(img, temp, cvPoint(x, y), &font, color);
    y += 20;

    sprintf(temp, "yratio=%.2lf, yratio2=%.2lf, xratio=%.2lf, xratio2=%.2lf", this->yratio, this->yratio2
      , this->xratio, this->xratio2);
    cvPutText(img, temp, cvPoint(x, y), &font, color);
    y += 20;

    sprintf(temp, "isFore=%d, neighbors=%d, inforeps=%d, fh=%d,fw=%d", this->isFore, this->neighbors, 
      this->inForePoints, this->inForeH, this->inForeW);
    cvPutText(img, temp, cvPoint(x, y), &font, color);
    y += 20;

    sprintf(temp, "score=%.2lf", this->score);
    cvPutText(img, temp, cvPoint(x, y), &font, color);
    y += 20;
  }

  vector<double> getAllFeature(int env)
  {
    vector<double> ret;
    double srcArea = srcWidth * srcHeight;
    double srcW = srcWidth;
    double srcH = srcHeight;

    //ret.push_back(tick);
    ret.push_back(rectW / srcW); ret.push_back(rectH / srcH); 
    ret.push_back(center.x / srcW); ret.push_back(center.y / srcH);
    ret.push_back(hasmerge);
    ret.push_back(points.size() / srcArea); ret.push_back(area / srcArea);
    ret.push_back(validHeng / (double) rectH); ret.push_back(validHengPosi);
    ret.push_back(validShu / (double) rectW); ret.push_back(validShuPosi);
    ret.push_back(ratio); ret.push_back(rectRatio);
    ret.push_back(otherpointsRatio); ret.push_back(forepointsRatio);
    ret.push_back(yratio); ret.push_back(yratio2);
    ret.push_back(xratio); ret.push_back(xratio2);
    ret.push_back(env);
    ret.push_back(this->isFore);
    ret.push_back(this->neighbors);
    ret.push_back(this->inForeH / srcH);
    ret.push_back(this->inForeW / srcW);
    ret.push_back(this->inForePoints / srcArea);
    return ret;
  }


  /** 绘制区域边界 */
  void drawRegionBoundary(IplImage * img)
  {
    for (map<CvRegion *, vector<CvPoint> >::iterator itr = boundarys.begin(); 
      itr != boundarys.end(); ++itr)
    {
      CvRegion * pRegion = itr->first;
      int ps = itr->second.size();
      double engH = boundaryEnergyHeng[pRegion];
      engH /= ps;

      double engS = boundaryEnergyShu[pRegion];
      engS /= ps;

      if (engH >= 250 || engS >= 250)
      {
        for (unsigned int i = 0; i < itr->second.size(); ++i)
          cvSet2D(img, itr->second[i].y, itr->second[i].x, CV_RGB(0, 255, 255));
      }
    }
  }

  /** 在图片上绘制该区域的原始像素 */
  void drawRegion(IplImage * img)
  {
    unsigned char * imgdata = (unsigned char *) img->imageData;
    int step = img->widthStep;
    unsigned char * srcdata = (unsigned char *) sourceImage->imageData;

    int height = img->height;
    int width = img->width;

    for (unsigned int i = 0; i < points.size(); ++i)
    {
      int ni = points[i].y;
      int nj = points[i].x;
      imgdata[ni * step + nj * 3 + 0] = srcdata[ni * step + nj * 3 + 0];
      imgdata[ni * step + nj * 3 + 1] = srcdata[ni * step + nj * 3 + 1];
      imgdata[ni * step + nj * 3 + 2] = srcdata[ni * step + nj * 3 + 2];
    }
  }
#endif

  // 简单合并两个区域
  void mergeRegion(CvRegion & r)
  {
    minH = min(minH, r.minH);
    maxH = max(maxH, r.maxH);
    minW = min(minW, r.minW);
    maxW = max(maxW, r.maxW);

    rectW = maxW - minW + 1;
    rectH = maxH - minH + 1;
    area = rectW * rectH;

    this->center.x = center.x * this->points.size() + r.center.x * r.points.size();
    this->center.x /= (this->points.size() + r.points.size());

    this->center.y = center.y * this->points.size() + r.center.y * r.points.size();
    this->center.y /= (this->points.size() + r.points.size());

    points.insert(points.end(), r.points.begin(), r.points.end());
    r.points.clear();
    hasmerge = true;

    calRegionBasicFeature();
  }

  /////////////////////////////////////////////////////////////////////////////////
  // 根据输入的原始图像，及其Segmentation的结果
  // 返回每个Region的信息，
  // 其中还包含了对区域边界的计算 
  // segmentation 的结果，comID从1开始到coms
  //////////////////////////////////////////////////////////////////////////////////
  static vector<CvRegion> getRegionFromSegment(IplImage * img, vector<vector<int> > comID, int coms, const set<string> & features)
  {
    int i, j;
    CvMat * gradEnergy = NULL;
    CvMat * theda = NULL;
    IplImage * gray = NULL;

    int height = img->height;
    int width = img->width;

    /** 特征图像准备 */

    /** 图像灰度 */
    if (features.find("gray") != features.end())
    {
      gray = cvCreateImage(cvGetSize(img), 8, 1);
      cvCvtColor(img, gray, CV_BGR2GRAY);
    }

    if (features.find("rgb") == features.end())
      img = NULL;

    /** 开始区域生成 */
    vector<CvRegion> regions = vector<CvRegion>(coms, CvRegion(height, width));

    for (i = 0; i < height; ++i)
    {
      for (j = 0; j < width; ++j)
      {
        int id = comID[i][j] - 1;
        regions[id].points.push_back(cvPoint(j, i));
      }
    }

    // 构造每个区域
    for (i = 0; i < coms; ++i)
      if (regions[i].points.size() < 15000 && regions[i].points.size() > 0) // 避免对大物体的无意义抽取，谨慎优化
        regions[i].calRegionBasicFeature();
      else
        regions[i].points.clear();

    if (gradEnergy != NULL) cvReleaseMat(&gradEnergy);
    if (theda != NULL) cvReleaseMat(&theda);
    if (gray != NULL) cvReleaseImage(&gray);

    return regions;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////
// FLOOD FILL 
// 根据输入的单通道二值化图像，获得类似Segmentation的 comID二维数组，
// 即每个像素点属于哪个连通分量 
// 对于每个非零元素进行FLOOD_FILL，comID从0开始编号，获得CvRegion表示
////////////////////////////////////////////////////////////////////////////////////////////
int dir[8][2] = { {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {-1, -1}, {1, 1}, {-1, 1}, {1, -1} };
void floodFill(IplImage * img, vector<vector<int> > & comID, vector<CvRegion> & regions, int dirs, int minSize, int maxSize)
{
  int i, j;
  int height = img->height;
  int width = img->width;

  for (i = 0; i < height; ++i)
    for (j = 0; j < width; ++j)
      comID[i][j] = -1;

  regions.clear();
  int coms = 0;
  unsigned char * data = (unsigned char *)img->imageData;
  int step = img->widthStep;
  for (i = height - 1; i >= 0; --i)
    for (j = 0; j < width; ++j)
      if (data[i * step + j] != 0 && comID[i][j] == -1)
      {
        comID[i][j] = coms;
        vector<CvPoint> points;
        queue<pair<int, int> > q;
        q.push(make_pair(i, j));
        regions.push_back(CvRegion(height, width));

        while (q.size() > 0)
        {
          pair<int, int> top = q.front();
          q.pop();
          int x = top.first;
          int y = top.second;
          points.push_back(cvPoint(y, x));

          for (int k = 0; k < dirs; ++k)
          {
            int nx = x + dir[k][0];
            int ny = y + dir[k][1];
            if (nx >= 0 && nx < height && ny >= 0 && ny < width && data[nx * step + ny] != 0 && comID[nx][ny] == -1)
            {
              comID[nx][ny] = coms;
              q.push(make_pair(nx, ny));
              regions.back().points.push_back(cvPoint(ny, nx));
            }
          }
        }

        if (points.size() <= maxSize && points.size() >= minSize) // 只抽取适度大小的物体
          regions.back().calRegionBasicFeature();
        else
        {
          regions.pop_back();
          coms--;
        }
        coms++;
      }

}

////////////////////////////////////////////////////////////////////////////////////////////
// FLOOD FILL 
// 根据输入的单通道二值化图像，获得类似Segmentation的 comID二维数组，
// 即每个像素点属于哪个连通分量 
// 对于每个非零元素进行FLOOD_FILL，comID从2开始编号
// 对于零元素，统一编号为1
////////////////////////////////////////////////////////////////////////////////////////////
int floodFill(IplImage * img, vector<vector<int> > & comID, int dirs, vector<int> & comSize, 
  vector<int> & minH, vector<int> & maxH, vector<int> & minW, vector<int> & maxW)
{
  int i, j;
  int height = img->height;
  int width = img->width;
  comID.resize(height);
  for (i = 0; i < height; ++i)
    comID[i].resize(width);

  for (i = 0; i < height; ++i)
    for (j = 0; j < width; ++j)
      comID[i][j] = -1;

  comSize.clear();
  minH.clear(); maxH.clear(); minW.clear(); maxW.clear();
  comSize.push_back(0); comSize.push_back(0);
  minH.push_back(9999); minH.push_back(9999); minW.push_back(9999); minW.push_back(9999);
  maxH.push_back(0); maxH.push_back(0); maxW.push_back(0); maxW.push_back(0);

  int coms = 1;
  unsigned char * data = (unsigned char *)img->imageData;
  int step = img->widthStep;
  for (i = height - 1; i >= 0; --i)
    for (j = 0; j < width; ++j)
      if (comID[i][j] == -1)
      {
        if (data[i * step + j] != 0)
        {
          coms++;
          comID[i][j] = coms;
          queue<pair<int, int> > q;
          q.push(make_pair(i, j));
          comSize.push_back(0);
          minH.push_back(9999); minW.push_back(9999);
          maxH.push_back(0); maxW.push_back(0);

          while (q.size() > 0)
          {
            pair<int, int> top = q.front();
            q.pop();
            comSize[coms]++;
            int x = top.first;
            int y = top.second;

            minH[coms] = min(minH[coms], x);
            maxH[coms] = max(maxH[coms], x);
            minW[coms] = min(minW[coms], y);
            maxW[coms] = max(maxW[coms], y);

            for (int k = 0; k < dirs; ++k)
            {
              int nx = x + dir[k][0];
              int ny = y + dir[k][1];
              if (nx >= 0 && nx < height && ny >= 0 && ny < width && data[nx * step + ny] != 0 && comID[nx][ny] == -1)
              {
                comID[nx][ny] = coms;
                q.push(make_pair(nx, ny));
              }
            }
          }
        }
        else
        {
          comID[i][j] = 1;
          comSize[1]++;
          minH[1] = min(minH[1], i);
          maxH[1] = max(maxH[1], i);
          minW[1] = min(minW[1], j);
          maxW[1] = max(maxW[1], j);
        }
      }
      return coms;
}

//////////////////////////////////////////////////////////////////////
// 对于给定图像进行种子填充，不同的灰度值是做不同的区域。
// id 从0开始编号
//////////////////////////////////////////////////////////////////////
int floodFill(IplImage * img, vector<vector<int> > & comID, vector<int> & comSize, int dirs)
{
  int i, j;
  int height = img->height;
  int width = img->width;
  comID = vector<vector<int> >(height, vector<int>(width, -1));
  comSize.clear();

  int coms = 0;
  unsigned char * data = (unsigned char *)img->imageData;
  int step = img->widthStep;
  for (i = height - 1; i >= 0; --i)
    for (j = 0; j < width; ++j)
      if (comID[i][j] == -1)
      {
        int value = data[i * step + j];
        int comsize = 0;
        comID[i][j] = coms;
        queue<pair<int, int> > q;
        q.push(make_pair(i, j));

        while (q.size() > 0)
        {
          pair<int, int> top = q.front();
          q.pop();
          int x = top.first;
          int y = top.second;
          comsize++;

          for (int k = 0; k < dirs; ++k)
          {
            int nx = x + dir[k][0];
            int ny = y + dir[k][1];
            if (nx >= 0 && nx < height && ny >= 0 && ny < width && data[nx * step + ny] == value && comID[nx][ny] == -1)
            {
              comID[nx][ny] = coms;
              q.push(make_pair(nx, ny));
            }
          }
        }
        coms++;
        comSize.push_back(comsize);
      }
      return coms;
}

#endif
