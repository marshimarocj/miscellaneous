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
vector<CvRect> buildingSegmentationNight(string file, string outputpath, int avergrayupper = 90)
{
  IplImage * img = cvLoadImage(file.data());
  int avergray = getAverGray(img);
  vector<CvRect> ret;
  
  if (avergray > avergrayupper) 
  {
    cvReleaseImage(&img);
    return ret;
  }

  //debug1(avergray);
  //cvNamedWindow("debug");
  //cvShowImage("debug", img);
  //cvWaitKey(0);

  string recordfile = outputpath + "record.txt";
  //ofstream fout(recordfile.data(), ios_base::app);
  //fout << file << endl;
  

	string filename = Util::getFileTrueName(file);
  
  IplImage * resize = cvCreateImage(cvSize(320, 240), 8, 3);
  cvResize(img, resize);

	IplImage * seg, * salMap, * salImage;
	vector<vector<int> > mask;
	CvMat * mat = SalientRegion::getImageSalientRegion(resize, seg, salMap, salImage, mask);

  //vector<vector<int> > comID;
  //int numC;
  //IplImage * segsmall = GraphBasedImageSegmentation::GraphBasedImageSeg(resize, comID, numC, 200, 100);

	string srcfile = outputpath + filename + "_src.jpg";
	string salientmapfile = outputpath + filename + "_map.jpg";
	string histogramfile = outputpath + filename + "_his.jpg";
  string segfile = outputpath + filename + "_seg.jpg";

	vector<double> histogram(mat->width, 0);
	for (int i = 0; i < mat->height; ++i)
		for (int j = 0; j < mat->width; ++j)
			histogram[j] += cvmGet(mat, i, j);
	for (int j = 0; j < 20; ++j)
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
  //builds.push_back(cvRect(srcl, 0, srcr - srcl + 1, img->height));
  builds.push_back(cvRect(srcl - srcw, 0, srcw * 3, img->height));
  builds.push_back(cvRect(srcl - srcw * 2, 0, srcw * 5, img->height));

  int srcll = ll * img->width / resize->width;
  int srcrr = rr * img->width / resize->width;
  int srcww = srcrr - srcll + 1;
  builds.push_back(cvRect(srcll, 0, srcww, img->height));

  //builds.push_back(cvRect(0, 0, img->width, img->height));

  //fout << builds.size() << endl;
  debug1(builds);
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

    string vbuildfile = outputpath + filename + "_buildv" + StringOperation::toString(i) + ".jpg";
    string sobelfile = outputpath + filename + "_sobel" + StringOperation::toString(i) + ".jpg";

    string srcfile = outputpath + filename + "_src" + StringOperation::toString(i) + ".jpg";
    string sobelbuildfile = outputpath + filename + "_builds" + StringOperation::toString(i) + ".jpg";
    string vhisimgfile = outputpath + filename + "_hisv" + StringOperation::toString(i) + ".jpg";

    //fout << rect << endl;
    cvSaveImage(vbuildfile.data(), vRefine);
    //cvSaveImage(sobelfile.data(), subSobel);
    //cvSaveImage(refinefile.data(), subRefine);
    cvSaveImage(sobelbuildfile.data(), sobelbuilding);
    cvSaveImage(srcfile.data(), subImage);

    cvReleaseImage(&subImage);
    cvReleaseImage(&subSobel);
    cvReleaseImage(&subRefine);
    cvReleaseImage(&sobelbuilding);
    cvReleaseImage(&v_hisimg);
    cvReleaseImage(&vbuilding);
    cvReleaseImage(&vRefine);
  }

	//cvSaveImage(srcfile.data(), img);
	//cvSaveImage(salientmapfile.data(), salMap);
	//cvSaveImage(histogramfile.data(), hisimg);
  //cvSaveImage(segfile.data(), segsmall);

	cvReleaseImage(&hisimg);
	cvReleaseImage(&seg);
	cvReleaseImage(&salMap);
	cvReleaseImage(&salImage);
	cvReleaseImage(&img);
  cvReleaseImage(&resize);
  //cvReleaseImage(&segsmall);
  cvReleaseMat(&mat);

 // fout.close();
  return ret;
}

void testShapeMatch()
{
	IplImage * img1 = cvLoadImage("D:\\data\\Building\\5.jpg");
	IplImage * img2 = cvLoadImage("D:\\data\\Building\\5.jpg");

	vector<CvPoint> ps1 = CvExt::extractPointsGivenColor(img1, CV_RGB(255, 255, 255));
	vector<CvPoint> ps2 = CvExt::extractPointsGivenColor(img2, CV_RGB(255, 255, 255));
	ShapeContext sc1(ps1);
	ShapeContext sc2(ps2);

	vector<pair<CvPoint, CvPoint> > match = ShapeContext::shapeContextMatch(sc1, sc2);
	debug3(ps1.size(), ps2.size(), match.size());

	CvMat * H = ShapeContext::findHomography(match);

	vector<IplImage *> images; images.push_back(img1); images.push_back(img2);
	IplImage * com = CvExt::combineImageRow(images);

	for (int k = 0; k < match.size(); ++k)
	{
		CvPoint left = match[k].first;
		CvPoint right = match[k].second;
		right.x += img1->width;

		if (k % 1 == 0)
		{
			cvLine(com, left, right, CV_RGB(255, 0, 0), 1);
			CvPoint tou = ShapeContext::getHomographyTrans(left, H);
			tou.x += img1->width;
			cvLine(com, left, tou, CV_RGB(0, 255, 0), 1);
			//debug3(left, right, tou);
		}
	}

	debug1(ShapeContext::getMatchError(match, H));

	cvNamedWindow("debug");
	cvShowImage("debug", com);

	cvWaitKey(0);
}

IplImage * editing(IplImage * src, IplImage * maskimg, IplImage * dst, CvRect rect)
{
	IplImage * realsrc = cvCreateImage(cvSize(rect.width, rect.height), 8, 3);
	cvResize(src, realsrc);
	IplImage * realmask = cvCreateImage(cvSize(rect.width, rect.height), 8, 1);
	cvResize(maskimg, realmask);

	//cout << "realmask" << endl;
	//cvShowImage("mask", realmask);
	//cvWaitKey(0);

	IplImage * ret = cvCreateImage(cvGetSize(dst), 8, 3);

	ImageEditing::imageEditing3(realsrc, dst, realmask, rect.x, rect.y, ret);

	cvReleaseImage(&realmask);
	cvReleaseImage(&realsrc);
	return ret;
}

IplImage * srcImage;
vector<CvPoint> ps;

void on_mouse( int _event, int x, int y, int flags, void* zhang)
{
	if (_event == CV_EVENT_LBUTTONDOWN)
	{
		static int cc = 0;
		printf("x = %d; y = %d;\n", x , y);
		ps.push_back(cvPoint(x, y));
	}
}

#include "SIFT/sift.h"
void calSIFTFeature(string path)
{
  string output = path + "\\feature.txt";
  ofstream fout(output.data());

  FileFinder finder(path);
  while (finder.hasNext())
  {
    string file = finder.next();
    cout << file << endl;
    if (file.find("jpg") == string::npos) continue;
    if (file.find("build3") != string::npos) continue;

    IplImage * img = cvLoadImage(file.data());

    vector<vector<double> > feas;
    vector<CvPoint> ps;
    getSIFT(img, feas, ps);

    fout << file << endl;
    fout << feas.size() << endl;
    for (int i = 0; i < feas.size(); ++i)
    {
      fout << ps[i].x << " " << ps[i].y << endl;
      for (int j = 0; j < feas[i].size(); ++j)
        fout << feas[i][j] << " ";
      fout << endl;
    }

    cvReleaseImage(&img);
  }

  fout.close();
}

void calGIST(string file)
{
  cout << file << endl;
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
 
void calGISTDir(string path)
{
  //string output = path + "\\feature_gist_new.txt";
  //ofstream fout(output.data());

  FileFinder finder(path);
  static int cnt = 0;
  while (finder.hasNext())
  {
    string file = finder.next();
    if (file.find("jpg") == string::npos) continue;
    if (file.find("build3") != string::npos) continue;
    //cout << file << endl;
    Util::run("gist.exe ", file);
  }
}

vector<vector<short int> > features;
vector<string> files;

inline int dist(vector<short int> & f1, vector<short int> & f2)
{
  int ret = 0;
  for (int i = 0; i < f1.size(); ++i)
    ret += abs(f1[i] - f2[i]);
  return ret;
}

void loadGIST(string dir)
{
  int cnt = 0;
  FileFinder finder(dir);
  while (finder.hasNext())
  {
    string file = finder.next();
    if (file.find("gist.txt") == string::npos) continue;

    ifstream fin(file.data());
    string filename;
    fin >> filename;
    filename = Util::getFatherPathName(dir) + "\\" + Util::getFileName(filename);
    //filename = Util::getFileName(filename);
    files.push_back(filename);
    vector<short int> fea(960);
    for (int i = 0; i < 960; ++i)
    {
      double d;
      fin >> d;
      fea[i] = (d * 10000);
    }
    features.push_back(fea);
    fin.close();
    cnt++;
    if (cnt % 100 == 0) cout << cnt << endl;
  }
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

vector<IplImage *> search_nearest(IplImage * img, int results)
{
  static bool first = true;
  if (first)
  {
    first = false;
    loadCluster("F:\\data\\Building\\buildings2\\64\\cluster.txt");
    loadCluster("F:\\data\\Building\\buildings2\\110\\cluster.txt");
    loadCluster("F:\\data\\Building\\buildings2\\182\\cluster.txt");
    loadCluster("F:\\data\\Building\\buildings2\\184\\cluster.txt");
    loadCluster("F:\\data\\Building\\buildings2\\186\\cluster.txt");
    loadCluster("F:\\data\\Building\\buildings2\\219\\cluster.txt");
    loadCluster("F:\\data\\Building\\buildings2\\255\\cluster.txt");
    loadCluster("F:\\data\\Building\\buildings2\\290\\cluster.txt");
    loadCluster("F:\\data\\Building\\buildings2\\402\\cluster.txt");
    loadCluster("F:\\data\\Building\\buildings2\\406\\cluster.txt");
    loadCluster("F:\\data\\Building\\buildings2\\446\\cluster.txt");
    loadCluster("F:\\data\\Building\\buildings2\\520\\cluster.txt");
    loadCluster("F:\\data\\Building\\buildings2\\583\\cluster.txt");
    loadCluster("F:\\data\\Building\\buildings2\\590\\cluster.txt");
    loadCluster("F:\\data\\Building\\buildings2\\620\\cluster.txt");
    loadCluster("F:\\data\\Building\\buildings2\\633\\cluster.txt");
  }

  vector<double> fea = getGIST(img);
  vector<short int> sfea(fea.size());
  for (int i = 0; i < fea.size(); ++i) sfea[i] = (fea[i] * 10000);

  vector<pair<int, int> > diss;
  for (int j = 0; j < files.size(); ++j)
    diss.push_back(make_pair(dist(sfea, features[j]), j));
  sort(diss.begin(), diss.end());

  vector<IplImage*> ret1;
  vector<IplImage*> ret2;
  string belong = Util::getFatherPathName(files[diss[0].second]);

  map<int, vector<IplImage *> > clusterres;
  vector<int> cluids;

  string dir = "F:\\data\\Building\\buildings2\\";
  for (int j = 0; j < 1000; ++j)
  {
    double distance = diss[j].first;
    string filename = files[diss[j].second];
    string nowbelong = Util::getFatherPathName(filename);
    string fileid = Util::getFileTrueName(files[diss[j].second]);
    string cateid = Util::getFatherPathName(files[diss[j].second]);
    if (cateid != belong) continue;

    IplImage * tarImage = cvLoadImage(dir + filename.data());

    int cluid = clustermap[fileid];
    debug4(filename, nowbelong, fileid, cluid);
    clusterres[cluid].push_back(tarImage);
    cluids.push_back(cluid);
  }

  vector<IplImage *> ret;

  sort(cluids.begin(), cluids.end());
  cluids.resize(unique(cluids.begin(), cluids.end()) - cluids.begin());

  
  for (int i = 0; ; i = (i + 1) % cluids.size())
  {
    vector<IplImage *> & v = clusterres[cluids[i]];
    if (v.size() > 0)
    {
      ret.push_back(v.front());
      v.erase(v.begin());
    }

    if (ret.size() >= results) break;
  }

  return ret;
}

void searchAndReplace(IplImage * srcImage, CvRect & rect)
{
  IplImage * region = CvExt::getSubImage(srcImage, rect);
  vector<IplImage *> similars = search_nearest(region, 40);

  IplImage * dealImage = cvCloneImage(srcImage);
  CvRect removerect = rect;
  ImageEditing::removeRegion(dealImage, removerect);
  cvShowImage("remove", dealImage, 128, 128, 512, 0);

  for (int i = 0; i < similars.size(); ++i)
    cvShowImage("near" + StringOperation::toString(i), similars[i], 64, 64, 128 * (i % 10), 512 - 128 + 128 * (i/ 10));
  cvWaitKey(200);

  cout << "Input Similar Landmark to Inception:" << endl;
  int cho;
  cin >> cho;
  
  IplImage * mask = ImageEditing::genMask(similars[cho]);
  int start = clock();
  IplImage * ret = editing(similars[cho], mask, dealImage, rect);
  int stop = clock();
  debug2("seamless cloning", stop - start);
  cvShowImage("ret", ret, 256, 256, 256, 0);
  cvReleaseImage(&ret);
  cvReleaseImage(&mask);
  cvWaitKey(100);

  cvReleaseImage(&region);
  cvReleaseImage(&dealImage);
  for (int i = 0; i < similars.size(); ++i)
    cvReleaseImage(&similars[i]);
}



void testBuildingExtraction()
{
  DirFinder dir("G:\\data\\Building\\images\\");
  while (dir.hasNext())
  {
    string dirpath = dir.next();
    //dirpath = "G:\\data\\Building\\images\\test";
    FileFinder finder(dirpath);

    string name = Util::getFileName(dirpath);
	  string output = "G:\\data\\Building\\buildings2\\" + name + "\\";
    Util::mkdir(output);

    while (finder.hasNext())
    {
      string file = finder.next();
      cout << file << endl;
      buildingSegmentationNight(file, output);
    }
    //break;
  }
}


void testClusterByColor(const string & path, const string & off)
{
  ifstream fin1(path + "\\cpsForZyh\\" + off + ".cps");
  set<int> ok;
  int id;
  while (fin1 >> id)
    ok.insert(id);
  fin1.close();

  ifstream fin2(path + "\\cpsForZyh\\" + off + ".cfg");
  set<string> cons;
  string line;
  int cnt = 0;
  while (fin2 >> line)
  {
    if (ok.find(cnt) != ok.end()) cons.insert(line);
    cnt++;
  }
  fin2.close();
  

  FileFinder finder(path + "\\" + off);
  vector<string> files; 
  vector<vector<double> > features;

  cnt = 0;
  while (finder.hasNext())
  {
    string file = finder.next();
    if (file.find("jpg") == string::npos) continue;

    //cout << file << endl;
    string name = Util::getFileName(file);
    name = name.substr(0, name.find_first_of("_"));
    //if (cons.find(name) == cons.end()) continue;
    //cout << name << endl;

    cout << file << endl;

    files.push_back(file);
    IplImage * img = cvLoadImage(file);

    IplImage * mask = ImageEditing::genMask(img);

    vector<double> fea = getHueHistogram(img, mask);
    features.push_back(fea);

    cvReleaseImage(&img);
    cvReleaseImage(&mask);
  }

  vector<int> clusters = K_MEANS(features);
  int C = 0;
  for (int i = 0; i < clusters.size(); ++i)
    C = max(C, clusters[i] + 1);

  for (int i = 0; i < C; ++i)
    Util::mkdir(path + "\\" + off + "\\" + StringOperation::toString(i));

  for (int i = 0; i < files.size(); ++i)
    Util::copyFile(files[i], path + "\\" + off + "\\" + StringOperation::toString(clusters[i]) + "\\" + Util::getFileName(files[i]));

  ofstream fout(path + "\\" + off + "\\" + "cluster.txt");
  for (int i = 0; i < files.size(); ++i)
    fout << files[i] << "\t" << clusters[i] << endl;
  fout.close();

  debug1(clusters);
}





int __main(int argc, char ** argv)
{
  /**
  testClusterByColor("F:\\data\\Building\\buildings2\\", "64");
  testClusterByColor("F:\\data\\Building\\buildings2\\", "110");
  testClusterByColor("F:\\data\\Building\\buildings2\\", "182");
  testClusterByColor("F:\\data\\Building\\buildings2\\", "184");
  testClusterByColor("F:\\data\\Building\\buildings2\\", "186");
  testClusterByColor("F:\\data\\Building\\buildings2\\", "219");
  testClusterByColor("F:\\data\\Building\\buildings2\\", "255");
  testClusterByColor("F:\\data\\Building\\buildings2\\", "290");
  testClusterByColor("F:\\data\\Building\\buildings2\\", "402");
  testClusterByColor("F:\\data\\Building\\buildings2\\", "406");
  testClusterByColor("F:\\data\\Building\\buildings2\\", "446");
  testClusterByColor("F:\\data\\Building\\buildings2\\", "520");
  testClusterByColor("F:\\data\\Building\\buildings2\\", "583");
  testClusterByColor("F:\\data\\Building\\buildings2\\", "590");
  testClusterByColor("F:\\data\\Building\\buildings2\\", "620");
  testClusterByColor("F:\\data\\Building\\buildings2\\", "633");
  */

  //return 0;

  //testBuildingExtraction();
  //return 0;

  //loadAndShowImages("E:\\analysis\\wordcut\\1000000013000006-20110912110824-2-0.jpg\\", "wordcut", true);
  //loadAndMergeShowImages("E:\\analysis\\plate\\1000000013000006-20110912110824-2-0.jpg\\", "plate");
  //cvDestroyAllWindows(true);
  //loadAndMergeShowImages("E:\\analysis\\horrizon_plate\\1000000013000006-20110912110824-2-0.jpg\\", "horrizon_plate");
  //cvDestroyAllWindows(true);

  //return 0;

  //yyu();
  //return 0;

  //string file = argv[1];
  //calGIST(file);
  //return 0;

  /**
  IplImage * img = cvLoadImage("D:\\test.jpg");
  vector<double> fea = getGIST(img);

  for (int i = 0; i < fea.size(); ++i) 
    cout << fea[i] << " ";
	return 0;
	*/
  
  //calGISTFeature("G:\\data\\Building\\buildings\\110\\");

 /**
  DirFinder finder("G:\\data\\Building\\buildings2\\");
  while (finder.hasNext())
  {
    string dir = finder.next();
    calGISTDir(dir);
  }
  return 0;
  */

  /**
  DirFinder finder("G:\\data\\Building\\buildings2\\");
  while (finder.hasNext())
  {
    string dir = finder.next();
    loadGIST(dir);
  }
  */
  
  /**
  loadGIST("F:\\data\\Building\\buildings2\\64\\");
  loadGIST("F:\\data\\Building\\buildings2\\110\\");
  loadGIST("F:\\data\\Building\\buildings2\\148\\");
  loadGIST("F:\\data\\Building\\buildings2\\182\\");
  loadGIST("F:\\data\\Building\\buildings2\\184\\");
  loadGIST("F:\\data\\Building\\buildings2\\186\\");
  loadGIST("F:\\data\\Building\\buildings2\\219\\");
  loadGIST("F:\\data\\Building\\buildings2\\406\\");
  loadGIST("F:\\data\\Building\\buildings2\\520\\");
  loadGIST("F:\\data\\Building\\buildings2\\620\\");
  loadGIST("F:\\data\\Building\\buildings2\\637\\");
  */

  //search_nearest("G:\\data\\Building\\buildings\\110\\00c31d884e6a5999f5fe8efb6e3714c4_build1.jpg");
  
  // 读取索引
  
  //loadGIST("G:\\data\\Building\\buildings2\\64\\");

  
  std::ifstream ifs("F:\\data\\Building\\buildings2\\gist_file_full3.txt", ios::binary);
  boost::archive::binary_iarchive ia(ifs);
  cout << "Loading file..." << endl;
  ia >> files;
  cout << "Loading feature..." << endl;
  ia >> features;
  ifs.close();
  

  cout << "Loaded Index Buildings： " << files.size() << endl;
  
  string query;
  while (true)
  {
    cout << "Input Query Image ID:" << endl;
    cin >> query;
    if (query == "-1") break;

    string file = "F:\\data\\Building\\query\\" + query + ".jpg";
    cout << file << endl;

    IplImage * img = cvLoadImage(file);
    cvShowImage("src", img, 256, 256, 0, 0);
  
    cvWaitKey(200);

    // 建筑分割并绘制
    vector<CvRect> rects = buildingSegmentationNight(file, "F:\\data\\Building\\debug\\", 255);
    debug1(rects);
    for (int i = 0; i < rects.size(); ++i)
    {
      //cvRectangle(img, rects[i], CV_RGB(50 * i, 50 * i, 50 * i), 1);
      IplImage * subimage = CvExt::getSubImage(img, rects[i]);
      cvShowImage("src" + StringOperation::toString(i), subimage, 64, 64, 128 * i, 300);
      cvReleaseImage(&subimage);
    }
    cvWaitKey(500);

    cout << "Input Landmark Area ID:" << endl;
    int id;
    cin >> id;
    searchAndReplace(img, rects[id]);
    cvReleaseImage(&img);

    cvOutputShowdImage("F:\\data\\Building\\debug\\" + query + "\\");
  }
  

  /**
  ofstream ofs("E:\\gist_file_full3.txt", ios::binary);
  boost::archive::binary_oarchive oa(ofs);
  oa << files;
  oa << features;
  ofs.close();
  */

  return 0;
}
