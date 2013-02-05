#ifndef IMAGE_ANNOTATION_H
#define IMAGE_ANNOTATION_H

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <iostream>
#include <set>
#include <vector>
#include <map>
#include <fstream>
#include <string>
#include <sstream>
#include <boost/foreach.hpp>

#include "Image_Operation.h"
#include "ImageIndex.h"
#include "Util.h"
using namespace std;

#include <iostream>
using namespace std;

const int inf=0x3fffffff;

// Minimum Cost Maximum Flow in Matrix
// Node start from 1 suggested
const int MaxN=2000;

// 流量上限
//int c[MaxN][MaxN],f[MaxN][MaxN];
//int w[MaxN][MaxN];
//int pnt[MaxN],value[MaxN],d[MaxN],mk[MaxN],open[MaxN],oldque[MaxN];


struct CodeBookStruct
{
	IplImage * img;
	string word;
	int order;
};


//////////////////////////////////////////////////////////////////
// 给定一个hashcode，与其相关的所有codebook图片
//////////////////////////////////////////////////////////////////
vector<Feature *> codeFeatures;
map<string,vector<Feature *> > codebookFeatures;

void GenCodeFeature(string codepath,string savefile,string standfile)
{
	FileFinder f(codepath);
	while (f.hasNext())
	{
		string currentdir(f.next());
		if (currentdir.find(".")!=currentdir.npos) continue;

		FileFinder images(currentdir);
		while (images.hasNext())
		{
			int order;
			long long code;

			string imageName(images.next());
			IplImage * bookimage = cvLoadImage(imageName.data());
			if (bookimage==NULL) continue;

			int rows = bookimage->height / blocksize;
			int cols = bookimage->width / blocksize;
			for (int m=0;m<rows;++m)
				for (int n=0;n<cols;++n)
				{
					IplImage * smallImage = getPartialImage(
						bookimage,cvRect(m*blocksize,n*blocksize,blocksize,blocksize));
					Mix mix;
					mix.init(smallImage);
					Hsv hsv;
					hsv.init(smallImage);
					Feature * feature = Feature::factory("mix");
					feature->init(mix.featureList,hsv.featureList);
					feature->srcImage = imageName;
					char temp[100];
					sprintf(temp,"%d_%d",m,n);
					string posi(temp);
					feature->featureInfo = posi;
					codeFeatures.push_back(feature);
					cvReleaseImage(&smallImage);

					if (codebookFeatures.find(imageName) == codebookFeatures.end())
						codebookFeatures[imageName] = vector<Feature * >();
					if (m==0)
						codebookFeatures[imageName].push_back(feature);
				}
			cvReleaseImage(&bookimage);
		}
		cout<<codeFeatures.size()<<endl;
	}
	cout<<"Total code book = "<<codeFeatures.size()<<endl;

	ofstream fout(savefile.data());
	BOOST_FOREACH(Feature * feature,codeFeatures)
	{
		for (int i=0;i<feature->featureList.size();++i)
			fout<<feature->featureList[i]<<" ";
		fout<<endl;
		fout<<feature->featureInfo<<endl;
		fout<<feature->srcImage<<endl;
	}
	fout.close();

	ofstream fout2(standfile.data());
		for (map<string,vector<Feature *> >::iterator itr = codebookFeatures.begin();
			itr!=codebookFeatures.end();++itr)
		{
			fout2<<itr->first<<endl;
			for (int i=0;i<itr->second.size();++i)
			{
				Feature * feature = itr->second[i];
				for (int i=0;i<feature->featureList.size();++i)
					fout2<<feature->featureList[i]<<" ";
				fout2<<endl;
			}
		}
	fout2.close();
}


bool com(const pair<double,string> & a, const pair<double,string> & b)
{
	return a.first < b.first;
}


void CalVisualWordDistribution()
{
	freopen("D:\\CLEF\\CLEF_8_8_new4\\distribution.txt","w",stdout);

	int size = codebookFeatures.size();
	vector<pair<double,string> > f(codeFeatures.size());

	map<string,vector<Feature *> > ::iterator itr ;
	for (itr = codebookFeatures.begin();itr!=codebookFeatures.end();++itr)
	{
		cout<<itr->first<<endl;
		map<string,int> count;
		for (int i=0;i<itr->second.size();++i)
		{
			Feature * srcFeature = itr->second[i];
			for (int j=0;j<codeFeatures.size();++j)
			{
				pair<double,string> a;
				a.first = srcFeature->calDistantce(codeFeatures[j]);
				a.second = codeFeatures[j]->srcImage;
				f[j] = a;
			}
			sort(f.begin(),f.end(),com);
			for (int j=0;j<20;++j) 
				count[Util::getFatherPathName(f[j].second)]++;
		}
		for (map<string,int>::iterator itr = count.begin();itr!=count.end();++itr)
		{
			cout<<itr->first<<" "<<itr->second<<endl;
		}
		//system("pause");
	}
}

void ReadCodeFeature(string codefeature,string standfile)
{
	codeFeatures.clear();
	ifstream fin(codefeature.data());
	char temp[1000];
	while (fin.getline(temp,1000,'\n'))
	{
		Feature * feature = Feature::factory("mix");
		feature->featureList = vector<int>(Feature::getFeatureLen("hsv")+
			Feature::getFeatureLen("mix"));
		istringstream sin(temp);
		for (int i=0;i<feature->featureList.size();++i)
			sin>>feature->featureList[i];

		fin.getline(temp,1000,'\n');
		feature->featureInfo = temp;
		fin.getline(temp,1000,'\n');
		feature->srcImage = temp;

		codeFeatures.push_back(feature);
		//system("pause");
		if (codeFeatures.size() % 1000 ==0) cout<<codeFeatures.size()<<endl;
	}
	cout<<codeFeatures.size()<<endl;

	ifstream fin2(standfile.data());
	while (fin2.getline(temp,1000))
	{
		string imageName(temp);
		for (int i=0;i<20;++i)
		{
			fin2.getline(temp,1000);
			Feature * feature = Feature::factory("mix");
			feature->featureList = vector<int>(Feature::getFeatureLen("hsv")+
				Feature::getFeatureLen("mix"));
			istringstream sin(temp);
			for (int i=0;i<feature->featureList.size();++i)
				sin>>feature->featureList[i];
			
			
					if (codebookFeatures.find(imageName) == codebookFeatures.end())
						codebookFeatures[imageName] = vector<Feature * >();
						codebookFeatures[imageName].push_back(feature);
		}
	}

	fin2.close();
}




void BruteAnnotation(IplImage * img,string path)
{
	Util::mkdir(path.data());

	string anno = path+"\\anno.txt";
	ofstream fout(anno.data());
	if (img==NULL) return;

	IplImage * closeImage = cvCreateImage(cvSize(rowblocks*blocksize,rowblocks*blocksize),8,3);
	string closeSmallImage;
	string closeImagePosi;

	for (int i=0;i<rowblocks;++i)
	{
		for (int j=0;j<rowblocks;++j)
		{
			IplImage * querySmallImage = getSuperPixel(img,rowblocks,blocksize,i,j);
		
				Mix mix;
					mix.init(querySmallImage);
					Hsv hsv;
					hsv.init(querySmallImage);
					Feature * feature = Feature::factory("mix");
					feature->init(mix.featureList,hsv.featureList);


			// 找到与输入superpixel最相近的superpixel
			double minDis = 1e20;
			string closestWord("NULL");			
			for (int k=0;k<codeFeatures.size();++k)
			{
				double dis = feature->calDistantce(codeFeatures[k]);
				if (dis<minDis)
				{
					minDis = dis;
					closeSmallImage = codeFeatures[k]->srcImage;
					closeImagePosi = codeFeatures[k]->featureInfo;
					closestWord = Util::getFatherPathName(codeFeatures[k]->srcImage);
				}
			}

			cout<<closestWord<<"\t";
			fout<<closestWord<<"\t";
			char queryfilename[200];
			char minfilename[200];
			IplImage * img = cvLoadImage(closeSmallImage.data());
			int startX,startY;
			sscanf(closeImagePosi.data(),"%d_%d",&startX,&startY);
			IplImage * minSmallImage = Feature::getPartialImage(img,cvRect(startX*blocksize,
				startY*blocksize,blocksize,blocksize));

			// 拼凑相近图片
			for (int x=0;x<blocksize;++x)
				for (int y=0;y<blocksize;++y)
					cvSet2D(closeImage,i*blocksize+x,j*blocksize+y,
						cvGet2D(minSmallImage,x,y));
		}
		fout<<endl;
		cout<<endl;
	}
	string savepath = path+"\\closeImage.jpg";
	cvSaveImage(savepath.data(),closeImage);

	IplImage * img2 = cvCloneImage(closeImage);
	cvResize(img,img2);
	savepath = path+"\\oriImage.jpg";
	cvSaveImage(savepath.data(),img2);
	fout.close();
}

#endif