#include <cstdio>
#include <cstdlib>
#include <vector>
#include <map>
#include <fstream>
#include <ctime>
#include <bitset>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>

#include "imageSegmentation.h"
#include "Util.h"
#include "CvRegion.h"
#include "CvImageOperationExt.h"
#include "geometry.h"
#include "SalientRegion.h"
#include "ImageAnalyzer.h"
#include "ShapeContext.h"
#include "ImageEditing.h"
#include "Research_EditingModel.h"
#include "Image_Feature.h"
#include "Algo_Algorithm.h"

#include "GIST/createGabor.h"
#include "GIST/prefilt.h"
#include "GIST/gistGabor.h"
#include "GIST/base64encoder.h"

#include <boost/serialization/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
 
#define debug1(x) cout << #x" = " << x << endl;
#define debug2(x, y) cout << #x" = " << x << " " << #y" = " << y << endl;
#define debug3(x, y, z) cout << #x" = " << x << " " << #y" = " << y << " " << #z" = " << z << endl;
#define debug4(x, y, z, w) cout << #x" = " << x << " " << #y" = " << y << " " << #z" = " << z << " " << #w" = " << w << endl;
using namespace std;

int getAverGray(IplImage * img)
{
  IplImage * nowGrayImage = cvCreateImage(cvGetSize(img), 8, 1);
  cvCvtColor(img, nowGrayImage, CV_BGR2GRAY);

  int averGray = 0;
  unsigned char * data = (unsigned char *) nowGrayImage->imageData;
  int step = nowGrayImage->widthStep;
  int height = img->height;
  int width = img->width;

  for (int i = 0; i < height; ++i)
    for (int j = 0; j < width; ++j)
      averGray += data[i * step + j];
  averGray /= height;
  averGray /= width;

  cvReleaseImage(&nowGrayImage);
  return averGray;
}

vector<string> split(const string & src, const string & delimit, 
	const string &  null_subst = "")
{    
	if(src.empty() || delimit.empty()) 
		return vector<string>();

	vector<string> v; 
	size_t deli_len = delimit.size();    
	long index = string::npos, last_search_position = 0;    
	while((index = src.find(delimit, last_search_position)) != string::npos)    
	{    
		if(index == last_search_position)    
			v.push_back(null_subst);    
		else   
			v.push_back(src.substr(last_search_position , 
			index - last_search_position));    
		last_search_position = index + deli_len;    
	}    
	string last_one = src.substr(last_search_position);    
	v.push_back(last_one.empty() ? null_subst : last_one);    
	return v;    
}

//// 对建筑图进行上下裁剪
IplImage * buildingRefine(IplImage * img, int & upper, int & lower)
{
  IplImage * subSobel = CvExt::getSobelDetection(img);
  vector<int> hist(img->height, 0);
  unsigned char * data = (unsigned char *) subSobel->imageData;
  int step = subSobel->widthStep;
  for (int i = 0; i < subSobel->height; ++i)
    for (int j = 0; j < subSobel->width; ++j)
      if (data[i * step + j] > 0) hist[i]++;

  upper = 0;
  while (hist[upper] == 0) upper++;

  lower = img->height - 1;
  while (hist[lower] < 3) lower--;
  if (upper > lower) upper = lower;

  cvReleaseImage(&subSobel);
  return CvExt::getSubImage(img, cvRect(0, upper, img->width, lower - upper + 1));
}

// 针对晚上图片的主题建筑分割
// 主要找敏感区域X方向直方图的顶点
#include "SalientRegion.h"
vector<CvRect> buildingSegmentationNight(IplImage * img, string outputpath, int avergrayupper = 90)
{
  //cout << file << endl;
  int avergray = getAverGray(img);
  vector<CvRect> ret;

	//string filename = Util::getFileTrueName(file);
  IplImage * resize = cvCreateImage(cvSize(160, 120), 8, 3);
  cvResize(img, resize);

	vector<vector<int> > mask;
  TimeUtil tu;
  tu.startCount("sal");
	CvMat * mat = SalientRegion::getImageSalientMap(resize);
  //tu.stopCount("sal");

	//string srcfile = outputpath + filename + "_src.jpg";
	//string salientmapfile = outputpath + filename + "_map.jpg";
	//string histogramfile = outputpath + filename + "_his.jpg";
  //string segfile = outputpath + filename + "_seg.jpg";

	vector<double> histogram(mat->width, 0);
	for (int i = 0; i < mat->height; ++i)
		for (int j = 0; j < mat->width; ++j)
			histogram[j] += cvmGet(mat, i, j);
	for (int j = 0; j < 10; ++j)
		histogram[j] = histogram[histogram.size() - 1 - j] = 0;

	IplImage * hisimg = CvExt::getHistogramImage(histogram);

  int maxi = 0;
  for (int i = 1; i < histogram.size(); ++i)
    if (histogram[i] > histogram[maxi]) maxi = i;

  int l = maxi;  
  while (l > 0 && (histogram[maxi] - histogram[l]) / histogram[maxi] < 0.2) l--;
  int r = maxi;
  while (r < histogram.size() - 1 && (histogram[maxi] - histogram[r]) / histogram[maxi] < 0.2) r++;
  cvLine(hisimg, cvPoint(l, 0), cvPoint(l, hisimg->height), CV_RGB(0, 255, 0));
  cvLine(hisimg, cvPoint(r, 0), cvPoint(r, hisimg->height), CV_RGB(0, 255, 0));

  int ll = 0;
  for (ll = 0; ll < histogram.size(); ++ll)
    if ((histogram[maxi] - histogram[ll]) / histogram[ll] < 0.2) break;
  int rr = 0;
  for (rr = histogram.size() - 1; rr >= 0; --rr)
    if ((histogram[maxi] - histogram[rr]) / histogram[rr] < 0.2) break;
  cvLine(hisimg, cvPoint(ll, 0), cvPoint(ll, hisimg->height), CV_RGB(0, 0, 255));
  cvLine(hisimg, cvPoint(rr, 0), cvPoint(rr, hisimg->height), CV_RGB(0, 0, 255));

  vector<CvRect> builds;
  int srcl = l * img->width / resize->width;
  int srcr = r * img->width / resize->width;
  int srcw = srcr - srcl + 1;
  builds.push_back(cvRect(srcl - srcw, 0, srcw * 3, img->height));
  builds.push_back(cvRect(srcl - srcw * 2, 0, srcw * 5, img->height));

  int srcll = ll * img->width / resize->width;
  int srcrr = rr * img->width / resize->width;
  int srcww = srcrr - srcll + 1;
  builds.push_back(cvRect(srcll, 0, srcww, img->height));

  for (int i = 0; i < builds.size(); ++i)
  {
    CvRect rect = builds[i]; int upper; int lower;
    CvExt::refineRect(rect, img);

    vector<double> vhis(rect.height, 0);
    for (int y = 0; y < rect.height; ++y)
      for (int x = 0; x < rect.width; ++x)
      {
        int ny = y + rect.y;
        ny = ny * resize->height / img->height;
        int nx = x + rect.x;
        nx = nx * resize->width / img->width;
        if (ny < 0 || nx < 0 || ny >= 240 || nx >= 320) cout << ny << " " << nx << " ";
		  	vhis[y] += cvmGet(mat, ny, nx);
      }

    for (int y = 0; y < 10; ++y)
      vhis[y] = vhis[vhis.size() - 1 - y] = 0;
    int maxy = 0;
    for (int y = 0; y < vhis.size(); ++y)
      if (vhis[y] > vhis[maxy]) maxy = y;

    CvRect vrect = rect;
    vrect.height = maxy + 1;
    IplImage * subImage = CvExt::getSubImage(img, rect);
    IplImage * vbuilding = CvExt::getSubImage(img, vrect);
    IplImage * v_hisimg = CvExt::getHistogramImage(vhis);
    IplImage * vRefine = buildingRefine(vbuilding, upper, lower);
    vrect.y += upper;
    vrect.height = lower - upper + 1;
    ret.push_back(vrect);

    IplImage * subSobel = CvExt::getSobelDetection(subImage);
    IplImage * subRefine = buildingRefine(subImage, upper, lower);
    CvRect sobelrect = rect;
    sobelrect.y += upper;
    sobelrect.height = (lower - upper + 1);
    IplImage * sobelbuilding = CvExt::getSubImage(img, sobelrect);
    ret.push_back(sobelrect);

    //string vbuildfile = outputpath + filename + "_buildv" + StringOperation::toString(i) + ".jpg";
    //string sobelfile = outputpath + filename + "_sobel" + StringOperation::toString(i) + ".jpg";
    //string srcfile = outputpath + filename + "_src" + StringOperation::toString(i) + ".jpg";
    ////string sobelbuildfile = outputpath + filename + "_builds" + StringOperation::toString(i) + ".jpg";
    //string vhisimgfile = outputpath + filename + "_hisv" + StringOperation::toString(i) + ".jpg";

    cvReleaseImage(&subImage);
    cvReleaseImage(&subSobel);
    cvReleaseImage(&subRefine);
    cvReleaseImage(&sobelbuilding);
    cvReleaseImage(&v_hisimg);
    cvReleaseImage(&vbuilding);
    cvReleaseImage(&vRefine);
  }

	cvReleaseImage(&hisimg);
	//cvReleaseImage(&img);
  cvReleaseImage(&resize);
  cvReleaseMat(&mat);

  return ret;
}

void calGIST(string file)
{
  string path = Util::getFilePath(file);
  string gistfile = path + "\\" + Util::getFileTrueName(file) + "_gist.txt";
  if (Util::existFile(gistfile.data())) return;

  IplImage * img = cvLoadImage(file.data());
  vector<CvPoint> ps;
  vector<double> fea = getGIST(img);

  ofstream fout(gistfile.data());
  fout << file << endl;
  for (int i = 0; i < fea.size(); ++i)
    fout << fea[i] << " ";
  fout << endl;
  fout.close();

  cvReleaseImage(&img);
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

// 计算GIST特征的EXE
// 参数： 要计算GIST的文件
void calGISTFeatureEXE(int argc, char ** argv)
{
  string file = "";
  if (argc == 1) return;
  file = string(argv[1]);
  calGIST(file);
}

// 计算Building Segmentation的EXE
// 参数：要进行建筑分割的文件
void buildingSegmentEXE(int argc, char ** argv)
{
  if (argc == 1) return;
  string file = string(argv[1]);
  string path = Util::getFilePath(file);
  IplImage * src = cvLoadImage(file);
  vector<CvRect> rects = buildingSegmentationNight(src, path);
  
  cout << "###" << endl;
  IplImage * img = cvLoadImage(file);
  for (int i = 0; i < rects.size(); ++i)
  {
    IplImage * sub = CvExt::getSubImage(img, rects[i]);
    string filesave = Util::getFileWithOutExt(file) + "_" + StringOperation::toString(i) + "_" + 
      StringOperation::toString(rects[i].x) + "_" + StringOperation::toString(rects[i].y) + "_" + 
      StringOperation::toString(rects[i].width) + "_" + StringOperation::toString(rects[i].height) + ".jpg";
    cvSaveImage(filesave.data(), sub);
    cout << filesave << endl;
    cvReleaseImage(&sub);
  }
}

// 搜索EXE
vector<vector<short int> > features;
vector<string> files;

inline int dist(vector<short int> & f1, vector<short int> & f2)
{
  int ret = 0;
  for (int i = 0; i < f1.size(); ++i)
    ret += abs(f1[i] - f2[i]);
  return ret;
}

map<string, int> clustermap;
void loadCluster(string filepath)
{
  ifstream fin(filepath.data());
  string file;
  int id;
  while (fin >> file >> id)
  {
    file = Util::getFileTrueName(file);
    clustermap[file] = id;
  }
  fin.close();
}

// 根据搜索图片，返回搜索结果和对应的类别
vector<pair<string, string> > search_nearest(string file, string cateid, int results)
{
  IplImage * img = cvLoadImage(file);

  const string dir = "D:\\data\\Landmark_Inception\\buildings2\\";
  static bool first = true;
  if (first)
  {
    first = false;
    loadCluster("D:\\data\\Landmark_Inception\\buildings2\\64\\cluster.txt");
    loadCluster("D:\\data\\Landmark_Inception\\buildings2\\110\\cluster.txt");
    loadCluster("D:\\data\\Landmark_Inception\\buildings2\\182\\cluster.txt");
    loadCluster("D:\\data\\Landmark_Inception\\buildings2\\184\\cluster.txt");
    loadCluster("D:\\data\\Landmark_Inception\\buildings2\\186\\cluster.txt");
    loadCluster("D:\\data\\Landmark_Inception\\buildings2\\219\\cluster.txt");
    loadCluster("D:\\data\\Landmark_Inception\\buildings2\\255\\cluster.txt");
    loadCluster("D:\\data\\Landmark_Inception\\buildings2\\290\\cluster.txt");
    loadCluster("D:\\data\\Landmark_Inception\\buildings2\\402\\cluster.txt");
    loadCluster("D:\\data\\Landmark_Inception\\buildings2\\406\\cluster.txt");
    loadCluster("D:\\data\\Landmark_Inception\\buildings2\\446\\cluster.txt");
    loadCluster("D:\\data\\Landmark_Inception\\buildings2\\520\\cluster.txt");
    loadCluster("D:\\data\\Landmark_Inception\\buildings2\\583\\cluster.txt");
    loadCluster("D:\\data\\Landmark_Inception\\buildings2\\590\\cluster.txt");
    loadCluster("D:\\data\\Landmark_Inception\\buildings2\\620\\cluster.txt");
    loadCluster("D:\\data\\Landmark_Inception\\buildings2\\633\\cluster.txt");
  }

  vector<double> fea = getGIST(img);
  vector<short int> sfea(fea.size());
  for (int i = 0; i < fea.size(); ++i) sfea[i] = (fea[i] * 10000);

  vector<pair<int, int> > diss;
  for (int j = 0; j < files.size(); ++j)
    diss.push_back(make_pair(dist(sfea, features[j]), j));
  sort(diss.begin(), diss.end());

  string belong = Util::getFatherPathName(files[diss[0].second]);
  if (cateid != "") belong = cateid;

  map<int, vector<string> > clusterres;
  vector<int> cluids;

  for (int j = 0; j < 1000; ++j)
  {
    double distance = diss[j].first;
    string filename = files[diss[j].second];
    string nowbelong = Util::getFatherPathName(filename);
    string fileid = Util::getFileTrueName(files[diss[j].second]);
    string cateid = Util::getFatherPathName(files[diss[j].second]);
    if (cateid != belong) continue;

    int cluid = clustermap[fileid];
    clusterres[cluid].push_back(dir + filename);
    cluids.push_back(cluid);
  }

  vector<pair<string, string> > ret;

  sort(cluids.begin(), cluids.end());
  cluids.resize(unique(cluids.begin(), cluids.end()) - cluids.begin());
  
  for (int i = 0; ; i = (i + 1) % cluids.size())
  {
    int total = 0;
    for (int j = 0; j < cluids.size(); ++j)
      total += clusterres[cluids[j]].size();
    if (total == 0) break;

    vector<string> & v = clusterres[cluids[i]];
    if (v.size() > 0)
    {
      ret.push_back(make_pair(v.front(), StringOperation::toString(i)));
      v.erase(v.begin());
    }

    if (ret.size() >= results) break;
  }

  return ret;
}

// 搜索相近建筑EXE
void searchEXE(int argc, char ** argv)
{
  if (argc == 1) return;
  string file = string(argv[1]);

  IplImage * srcImage = cvLoadImage(file);
  double srcRatio = (double) srcImage->height / (double) srcImage->width;

  std::ifstream ifs("D:\\data\\Landmark_Inception\\gist_file_full3.txt", ios::binary);
  boost::archive::binary_iarchive ia(ifs);
  ia >> files;
  ia >> features;
  ifs.close();
  cout << "Loaded Index Buildings： " << files.size() << endl;

  cout << "###" << endl;
  vector<pair<string, string> > nears = search_nearest(file, "", 50);
  string dir = Util::getFileWithOutExt(file) + "\\";
  Util::mkdir(dir);
  for (int i = 0; i < nears.size(); ++i)
  {
    string save = dir + StringOperation::toString(100 + i) + "!" + Util::getFileTrueName(nears[i].first) + ".jpg";
    IplImage * now = cvLoadImage(nears[i].first);
    double nowRatio = (double) now->height / (double) now->width;
    double ratio = srcRatio / nowRatio;
    double avergray = getAverGray(now);
    if (avergray > 100) continue;
    //debug1(avergray);
    if (ratio > 1.6 || ratio < 0.6) continue;
    cvSaveImage(save.data(), now);
    //cout << save << endl;
  }

  cvReleaseImage(&srcImage);
}

IplImage * editing(IplImage * src, IplImage * maskimg, IplImage * dst, CvRect rect)
{
	IplImage * realsrc = cvCreateImage(cvSize(rect.width, rect.height), 8, 3);
	cvResize(src, realsrc);
	IplImage * realmask = cvCreateImage(cvSize(rect.width, rect.height), 8, 1);
	cvResize(maskimg, realmask);

	IplImage * ret = cvCreateImage(cvGetSize(dst), 8, 3);

	ImageEditing::imageEditing3(realsrc, dst, realmask, rect.x, rect.y, ret);

	cvReleaseImage(&realmask);
	cvReleaseImage(&realsrc);
	return ret;
}

// 图像编辑EXE
// 参数：建筑 要嵌入的原图 x, y, width, height, 结果文件
void editingEXE(int argc, char ** argv)
{
  string srcFile = string(argv[1]);
  string dstFile = string(argv[2]);
  int x = StringOperation::toInt(argv[3]);
  int y = StringOperation::toInt(argv[4]);
  int width = StringOperation::toInt(argv[5]);
  int height = StringOperation::toInt(argv[6]);
  string saveFile = Util::getFileWithOutExt(srcFile) + "_edit.jpg";

  IplImage * src = cvLoadImage(srcFile);
  IplImage * dst = cvLoadImage(dstFile);
  //cout << src << endl;
  //cout << dst << endl;

  int dstH = dst->height;
  int dstW = dst->width;
  IplImage * resizeDst = cvCreateImage(cvSize(320, 240), 8, 3);
  cvResize(dst, resizeDst);

  x = x * 320 / dstW;
  y = y * 240 / dstH;
  width = width * 320 / dstW;
  height = height * 240 / dstH;

  CvRect rect = cvRect(x, y, width, height);
 
  TimeUtil tu;
  tu.startCount("seg");
  //vector<CvRect> rects = buildingSegmentationNight(srcFile, "D:\\");
  //tu.stopCount("seg");

  ImageEditing::removeRegion(resizeDst, rect);
  tu.startCount("mask");
  IplImage * mask = ImageEditing::genMask(src);
  //tu.stopCount("mask");

  tu.startCount("edit");
  IplImage * save = editing(src, mask, resizeDst, rect);
  //tu.stopCount("edit");

  cvSaveImage(saveFile.data(), save);
}

void genvideo()
{
  IplImage * srcImage = cvLoadImage("D:\\data\\Landmark_Inception\\realtimevideo\\1.jpg");
  IplImage * dstImage = cvCreateImage(cvSize(srcImage->width * 3 / 2, srcImage->height), 8, 3);

  for (int i = 0; i < srcImage->width / 2; ++i)
  {
    cvZero(dstImage);
    CvExt::setSubImage(dstImage, srcImage, i, 0);
    string save = string("D:\\data\\Landmark_Inception\\realtimevideo\\") + "movie_" + StringOperation::toString(1000 + i) + ".jpg";
    cvSaveImage(save.data(), dstImage);
  }
}

void runvideo()
{
  IplImage * embedd = cvLoadImage("D:\\data\\Landmark_Inception\\realtimevideo\\embedd.jpg");
  IplImage * mask = ImageEditing::genMask(embedd);
  IplImage * resizeDst = cvCreateImage(cvSize(160, 120), 8, 3);
  IplImage * result = cvCreateImage(cvSize(160, 120), 8, 3);


  cvNamedWindow("debug");
  cvNamedWindow("src");
  cvMoveWindow("src", 100, 100);
  cvMoveWindow("debug", 100, 500);

  TimeUtil tu;
  FileFinder finder("D:\\data\\Landmark_Inception\\realtimevideo\\origin\\");
  while (finder.hasNext())
  {
    for (int i = 0; i < 10; ++i) if (finder.hasNext()) finder.next();
    if (!finder.hasNext()) break;

    tu.startCount("loop");

    string file = finder.next();
    //cout << file << endl;
    IplImage * dst = cvLoadImage(file);
    cvResize(dst, resizeDst);

    tu.startCount("seg");
    vector<CvRect> rects = buildingSegmentationNight(resizeDst, "D:\\");
    tu.stopCount("seg");

    tu.startCount("edit");
    IplImage * save = editing(embedd, mask, resizeDst, rects[3]);
    tu.stopCount("edit");

    cvResize(save, result);

    cvShowImage("debug", result);
    cvShowImage("src", resizeDst);
    cvWaitKey(1);

    cvReleaseImage(&save);
    cvReleaseImage(&dst);

    tu.stopCount("loop");

    //cvWaitKey(0);
  }

  system("pause");
}

void yyu()
{
  FileFinder finder("D:\\yyu50\\");
  vector<IplImage *> images;
  int H = 20, W = 16;

  map<IplImage *, string> names;
  while (finder.hasNext())
  {
    string file = finder.next();
    IplImage * img = cvLoadImage(file);
    images.push_back(img);
    names[img] = Util::getFileTrueName(file);
    cout << file << endl;
  }

  //random_shuffle(images.begin(), images.end());

  IplImage * src = cvLoadImage("D:\\yyu.jpg");
  IplImage * ret = combineImageFromSmall(images, src);
  cvSaveImage("D:\\yyu_mix.bmp", ret);
}

int main(int argc, char ** argv)
{
  //runvideo();

  //yyu();
  //return 0;
  
  /**
  char ** arg = new char *[7];
  arg[1] = "D:\\website\\media\\datas\\5fd4c90bd3a7dd51939a806c2a19ca70\\src_1_163_43_78_289\\123!2e65e39f9a5e8fd3d16899974061d1ea_src1.jpg";
  arg[2] = "D:\\website\\media\\datas\\5fd4c90bd3a7dd51939a806c2a19ca70\\src.jpg";
  arg[3] = "163";
  arg[4] = "43";
  arg[5] = "78";
  arg[6] = "289";
  */
  
  //calGISTFeatureEXE(argc, argv);
  //buildingSegmentEXE(argc, argv);
  //searchEXE(argc, argv);
  editingEXE(argc, argv);

  return 0;
}
