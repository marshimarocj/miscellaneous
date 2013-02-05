#ifndef IMAGE_FEATURE
#define IMAGE_FEATURE

#include "stdafx.h"

using namespace std;

inline int sqr(int x)
{return x*x;}


char* strrev(char* szT)
{
  string s(szT);
  reverse(s.begin(), s.end());
  strncpy(szT, s.c_str(), s.size());
  szT[s.size()+1] = '\0';
  return szT;
}

char* itoa(int value, char*  str, int radix)
{
  int rem = 0;
  int pos = 0;
  char ch  = '!';
  do
  {
    rem = value % radix;
    value /= radix;
    if ( 16 == radix )
    {
      if( rem >= 10 && rem <= 15 )
      {
        switch(rem)
        {
          case 10:
            ch = 'a';
            break;
          case 11:
            ch ='b';
            break;
          case 12:
            ch = 'c';
            break;
          case 13:
            ch ='d';
            break;
          case 14:
            ch = 'e';
            break;
          case 15:
            ch ='f';
            break;
        }
      }
    }
    if('!' == ch)
    {
      str[pos++] = (char) ( rem + 0x30 );
    }
    else
    {
      str[pos++] = ch;
    }
  } while( value != 0 );
  str[pos] = '\0';
  return strrev(str);
}

///////////////////////////////////////////////////////////////////////////////
// Image Features 
///////////////////////////////////////////////////////////////////////////////
class Feature{
  public:
    vector<int> featureList;

    int getIntHashCode();
    long long getLongHashCode();

    double distance;

    string featureInfo;
    string srcImage;

  public:
    Feature(){}
    virtual ~Feature(){}

    virtual int getDefaultFeatureDim() = 0;

    virtual void init(IplImage * img , CvRect rect , CvMat * mask , int
        featureDim) = 0;

    void init(IplImage * img , CvRect rect , CvMat * mask)
    { init(img , rect , mask , getDefaultFeatureDim()); }  

    void init(IplImage * img , CvRect rect)
    { init(img , rect , NULL); } 

    void init(IplImage * img)
    { init(img , cvRect(0 , 0 , img->width , img->height)); }

    void init(vector<int> & a,vector<int> &b)
    {
      this->featureList = vector<int>(a.size()+b.size());
      for (int i = 0; i < a.size(); ++i)
        featureList[i] = a[i];
      for (int j = 0; j < b.size(); ++j)
        featureList[a.size() + j] = b[j];
    }

    vector<double> getNormalized()
    {
      vector<double> ret(featureList.size());
      double sum = 0;
      for (int i = 0; i < featureList.size(); ++i)
        sum += featureList[i];
      for (int i = 0; i < featureList.size(); ++i)
        ret[i] = (double)featureList[i] / sum;
      return ret;
    }

    double calDistantce(Feature * other)
    {
      double ret = 0.0;
      for (int i=0;i<this->featureList.size();++i)
        ret+= abs(this->featureList[i] - other->featureList[i]);
      return ret;
    }

    static int getFeatureLen(string name);
    static Feature* factory(string name);
    static Feature* factory(Feature* feature);
    static IplImage * getPartialImage(IplImage * srcImage,CvRect rect);

    /** Get image color tone */
    static string getImageColorTone(IplImage * srcImage, double threshold =
        3.0);

    /** Get image main color */
    static string getImageMainColor(IplImage * srcImage);

    static vector<string> getImageColorTones()
    {
      vector<string> ret;
      ret.push_back("black");
      ret.push_back("gray");
      ret.push_back("blue");
      ret.push_back("violet");
      ret.push_back("red");
      ret.push_back("green");
      ret.push_back("neutral");
      return ret;
    }

    static bool comFeature(Feature * a,Feature * b)
    {	
      return a->distance<b->distance;
    }

    friend ostream& operator << (ostream& out, const Feature& feature);
};

ostream& operator << (ostream & out , const Feature & feature)
{
  //out << "FEAUTRE" << endl;
  for (int i = 0; i < feature.featureList.size(); ++i) {
    out << feature.featureList[i] << "\t";
    //if (i % 10 == 9) out << endl;
  }
  return out;
}

class Lab: public Feature{
  public:
    Lab(): Feature(){}
    Lab(const Lab& lab){
      this->featureList = lab.featureList;
    }
    ~Lab(){}

    int getDefaultFeatureDim()
    {
      return Feature::getFeatureLen("lab");
    }

    void init(IplImage * img , CvRect rect , CvMat * mask , int
        featureDim);
    void init(IplImage * img , CvRect rect , CvMat * mask)
    { init(img , rect , mask , getDefaultFeatureDim()); }  
    void init(IplImage * img , CvRect rect)
    { init(img , rect , NULL); } 

    void init(IplImage * img)
    { init(img , cvRect(0 , 0 , img->width , img->height)); }
};


class AverageSD: public Feature{
  public:
    AverageSD(): Feature(){}
    AverageSD(const AverageSD & aver){
      this->featureList = aver.featureList;
    }
    ~AverageSD(){}

    int getDefaultFeatureDim()
    {
      return Feature::getFeatureLen("averageSD");
    }

    void init(IplImage * img , CvRect rect , CvMat * mask , int
        featureDim);
    void init(IplImage * img , CvRect rect , CvMat * mask)
    { init(img , rect , mask , getDefaultFeatureDim()); }  
    void init(IplImage * img , CvRect rect)
    { init(img , rect , NULL); } 

    void init(IplImage * img)
    { init(img , cvRect(0 , 0 , img->width , img->height)); }
};

class Law: public Feature{
  public:
    Law():Feature(){}
    Law(const Law& law){
      this->featureList = law.featureList;
    }
    ~Law(){}

    int getDefaultFeatureDim()
    {
      return Feature::getFeatureLen("law");
    }

    void init(IplImage * img , CvRect rect , CvMat * mask , int
        featureDim);
    void init(IplImage * img , CvRect rect , CvMat * mask)
    { init(img , rect , mask , getDefaultFeatureDim()); }  
    void init(IplImage * img , CvRect rect)
    { init(img , rect , NULL); } 

    void init(IplImage * img)
    { init(img , cvRect(0 , 0 , img->width , img->height)); }
};

class Hsv:public Feature
{
  public:
  public:
    Hsv():Feature(){}
    Hsv(const Hsv& hsv){
      this->featureList = hsv.featureList;
    }
    ~Hsv(){}

    int getDefaultFeatureDim()
    {
      return Feature::getFeatureLen("hsv");
    }

    void init(IplImage * img , CvRect rect , CvMat * mask , int
        featureDim);
    void init(IplImage * img , CvRect rect , CvMat * mask)
    { init(img , rect , mask , getDefaultFeatureDim()); }  
    void init(IplImage * img , CvRect rect)
    { init(img , rect , NULL); } 

    void init(IplImage * img)
    { init(img , cvRect(0 , 0 , img->width , img->height)); }
};

class LBP:public Feature
{
  public:
  public:
    LBP():Feature(){}
    LBP(const LBP& lbp){
      this->featureList = lbp.featureList;
    }
    ~LBP(){}


    int getIntHashCode();

    int getDefaultFeatureDim()
    {
      return Feature::getFeatureLen("lbp");
    }

    void init(IplImage * img , CvRect rect , CvMat * mask , int
        featureDim);
    void init(IplImage * img , CvRect rect , CvMat * mask)
    { init(img , rect , mask , getDefaultFeatureDim()); }  
    void init(IplImage * img , CvRect rect)
    { init(img , rect , NULL); } 

    void init(IplImage * img)
    { init(img , cvRect(0 , 0 , img->width , img->height)); }
};

class Mix:public Feature
{
  public:
  public:
    Mix():Feature(){}
    Mix(const Mix& mix){
      this->featureList = mix.featureList;
    }
    ~Mix(){}

    int getDefaultFeatureDim()
    {
      return Feature::getFeatureLen("mix");
    }

    void init(IplImage * img , CvRect rect , CvMat * mask , int
        featureDim);
    void init(IplImage * img , CvRect rect , CvMat * mask)
    { init(img , rect , mask , getDefaultFeatureDim()); }  
    void init(IplImage * img , CvRect rect)
    { init(img , rect , NULL); } 

    void init(IplImage * img)
    { init(img , cvRect(0 , 0 , img->width , img->height)); }
};


class Code:public Feature
{
  public:
    Code():Feature(){}
    Code(const Code& code){
      this->featureList = code.featureList;
    }
    ~Code(){}

    long long code2;

    int getDefaultFeatureDim()
    {
      return Feature::getFeatureLen("code");
    }

    void init(IplImage * img , CvRect rect , CvMat * mask , int
        featureDim);
    void init(IplImage * img , CvRect rect , CvMat * mask)
    { init(img , rect , mask , getDefaultFeatureDim()); }  
    void init(IplImage * img , CvRect rect)
    { init(img , rect , NULL); } 

    void init(IplImage * img)
    { init(img , cvRect(0 , 0 , img->width , img->height)); }

    long long getLongHashCode();
};

class GIST:public Feature
{
  public:
    GIST():Feature(){}
    GIST(const GIST& gist){
      this->featureList = gist.featureList;
    }
    ~GIST(){}

    int getDefaultFeatureDim()
    {
      return Feature::getFeatureLen("gist");
    }

    void init(IplImage * img , CvRect rect , CvMat * mask , int
        featureDim);
    void init(IplImage * img , CvRect rect , CvMat * mask)
    { init(img , rect , mask , getDefaultFeatureDim()); }  
    void init(IplImage * img , CvRect rect)
    { init(img , rect , NULL); } 

    void init(IplImage * img)
    { init(img , cvRect(0 , 0 , img->width , img->height)); }
};

//////////////////////////////////////////////////////////
//Feature Implementation
//////////////////////////////////////////////////////////
class LAW{
  public:
    double filters[9][5][5];

    LAW();
    //filter image and output 9 channels to each law mask
    void filter(IplImage* img, IplImage* feature[]);
};

class ImageFeature{
  public:
    static void extLab(IplImage* img, vector<vector<int> >& features);
    static void extLaw(IplImage* img, vector<vector<int> >& features);
    static void extTotal(IplImage* img, vector<vector<int> >& features);
    static void extHsv(IplImage* img, vector<vector<int> >& features);
};

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
const double lawMask[9][5][5]= {
  {{-1, -2, 0, 2, 1},
    {-4, -8, 0, 8, 4}, 
    {-6, -12, 0, 12, 6},
    {-4, -8, 0, 8, 4}, 
    {-1, -2, 0, 2, 1}
  }, 
  {{1, -4, 6, -4, 1}, 
    {4, -16, 24, -16, 4},
    {6, -24, 36, -24, 6},
    {4, -16, 24, -16, 4},
    {1, -4, 6, -4, 1}
  },
  {{-1, 0, 2, 0, -1},
    {-4, 0, 8, 0, -4},
    {-6, 0, 12, 0, -6},
    {-4, 0, 8, 0, -4},
    {-1, 0, 2, 0, -1}
  },
  {{1, -4, 6, -4, 1},
    {-4, 16, -24, 16, -4},
    {6, -24, 36, -24, 6},
    {-4, 16, -24, 16, -4},
    {1, -4, 6, -4, 1}
  },
  {{-1, 4, -6, 4, -1},
    {0, 0, 0, 0, 0},
    {2, -8, 12, -8, 2},
    {0, 0, 0, 0, 0},
    {-1, 4, -6, 4, -1}
  },
  {{1, 0, -2, 0, 1},
    {0, 0, 0, 0, 0},
    {-2, 0, 4, 0, -2},
    {0, 0, 0, 0, 0},
    {1, 0, -2, 0, 1}
  },
  {{1, 2, 0, -2, -1},
    {2, 4, 0, -4, -2},
    {0, 0, 0, 0, 0},
    {-2, -4, 0, 4, 2},
    {-1, -2, 0, 2, 1}
  },
  {{-1, 4, -6, 4, -1},
    {-2, 8, -12, 8, -2},
    {0, 0, 0, 0, 0},
    {2, -8, 12, -8, 2},
    {1, -4, 6, -4, 1}
  },
  {{1, 0, -2, 0, 1},
    {2, 0, -4, 0, 2},
    {0, 0, 0, 0, 0},
    {-2, 0, 4, 0, -2},
    {-1, 0, 2, 0, -1}
  }
};

Feature* Feature::factory(string name){
  if(name == "lab")
    return new Lab();
  if(name == "law")
    return new Law();
  if (name == "hsv")
    return new Hsv();
  if (name == "lbp")
    return new LBP();
  if (name=="mix")
    return new Mix();
  if (name =="code")
    return new Code();
  if (name =="gist")
    return new GIST();
  if (name =="averageSD")
    return new AverageSD();
  return NULL;
}

int Feature::getFeatureLen(string name)
{
  if (name =="law")
    return 18;
  if (name =="lab")
    return 32;
  if (name =="hsv")
    return 16;
  if (name == "lbp")
    return 40;
  if (name =="mix") 
    return 42;
  if (name =="code")
    return 16;
  if (name =="gist")
    return 960;
  if (name =="averageSD")
    return 12;
  return 0;
};



Feature* Feature::factory(Feature* feature){
  Lab* lab = dynamic_cast<Lab*>(feature);
  if(lab){
    return new Lab(*lab);
  }
  Law* law = dynamic_cast<Law*>(feature);
  if(law){
    return new Law(*law);
  }
  Hsv* hsv = dynamic_cast<Hsv*>(feature);
  if (hsv)
    return new Hsv(*hsv);
  LBP* lbp = dynamic_cast<LBP*>(feature);
  if (lbp) 
    return new LBP(*lbp);
  Mix* mix = dynamic_cast<Mix*>(feature);
  if (mix) 
    return new Mix(*mix);
  Code * code = dynamic_cast<Code*>(feature);
  if (code)
    return new Code(*code);
  AverageSD * averSD = dynamic_cast<AverageSD*>(feature);
  if (averSD)
    return new AverageSD(*averSD);
  return NULL;
}

IplImage * Feature::getPartialImage(IplImage * srcImage,CvRect rect)
{
  IplImage * ret = cvCreateImage(cvSize(rect.width,rect.height),8,3);
  for (int i=0;i<ret->height;++i)
    for (int j=0;j<ret->width;++j)
      cvSet2D(ret,i,j,cvGet2D(srcImage,i+rect.x,j+rect.y));
  return ret;
}


inline int getABBucketID(int data)
{
  static int range[12][2] = 
  {
    {-128 + 127 , -70 + 127},
    {-70 + 127 , -50 + 127},
    {-50 + 127 , -30 + 127},
    {-30 + 127 , -20 + 127},
    {-20 + 127 , -10 + 127},
    {-10 + 127 , 0 + 127},
    {0 + 127 , 10 + 127},
    {10 + 127 , 20 + 127},
    {20 + 127 , 30 + 127},
    {30 + 127 , 50 + 127},
    {50 + 127 , 70 + 127},
    {70 + 127 , 128 + 127}
  };

  for (int i = 0; i < 12; ++i)
    if (data > range[i][0] && data <= range[i][1]) 
      return i;
}

string Feature::getImageMainColor(IplImage * img)
{
  static const int TONES = 6;
  static string name[TONES] = 
  {
    "black",
    "gray",
    "blue",
    "violet",
    "red",
    "green",
    //"yellow"
  };

  static int LAB[TONES][3] = 
  {
    // black
    {0  ,   127,  127},
    // gray
    {255,   127,  127},
    // blue
    {224,   0,    0},
    // violet 
    {224,   255,  0},
    // red
    {224,   255,  255},
    // green
    {224,   0,    255},
    // yellow
    //{244,   127,  255}
  };

  int count[TONES];
  memset(count , 0 , sizeof(count));

  IplImage * lab = cvCreateImage(cvGetSize(img) , 8 , 3);
  cvCvtColor(img , lab , CV_BGR2Lab);

  /** Cal the color ditribution */
  for (int i = 0; i < lab->height; ++i)
    for (int j = 0; j < lab->width; ++j) {
      int L = (int)cvGet2D(lab , i , j).val[0];
      int a = (int)cvGet2D(lab , i , j).val[1];
      int b = (int)cvGet2D(lab , i , j).val[2];
      /** First consider black , if L is lower than 64 , we thought its
       * black
       */
      if (L < 64) {
        count[0]++;
        continue;
      }

      /** Second we find the most nearest color to this pixel except
       * gray 
       */
      int minDis = 10000;
      int minTar = -1;
      for (int k = 2; k < TONES; ++k) {
        if (abs(a - LAB[k][1]) + abs(b - LAB[k][2]) < minDis) {
          minDis = abs(a - LAB[k][1]) + abs(b - LAB[k][2]);
          minTar = k;
        }
      }

      /** gray is consider specially */
      if (abs(a - LAB[1][1]) + abs(b - LAB[1][2]) < 4) {
        minTar = 1;
        minDis = abs(a - LAB[1][1]) + abs(b - LAB[1][2]);
      }

      count[minTar]++;
    }

  cvReleaseImage(&lab);

  int colorfulCount = 0;
  int averCount = 0;
  int maxCount = -1;
  int maxTone = -1;
  for (int i = 0; i < TONES; ++i) {
    averCount += count[i];
    if (count[i] > maxCount) {
      maxCount = count[i];
      maxTone = i;
    }
  }
  for (int i = 2; i < TONES; ++i) 
    colorfulCount += count[i];

  for (int i = 0; i < TONES; ++i) 
    cout << name[i] << "\t";
  cout << endl;
  for (int i = 0; i < TONES; ++i)
    cout << count[i] << "\t";
  cout << endl;

  averCount /= TONES;
  return name[maxTone];
}


string Feature::getImageColorTone(IplImage * img, double threshold)
{
  static const int TONES = 6;
  static string name[TONES] = 
  {
    "black",
    "gray",
    "blue",
    "violet",
    "red",
    "green",
    //"yellow"
  };

  static int LAB[TONES][3] = 
  {
    // black
    {0  ,   127,  127},
    // gray
    {255,   127,  127},
    // blue
    {224,   0,    0},
    // violet 
    {224,   255,  0},
    // red
    {224,   255,  255},
    // green
    {224,   0,    255},
    // yellow
    //{244,   127,  255}
  };

  int count[TONES];
  memset(count , 0 , sizeof(count));

  IplImage * lab = cvCreateImage(cvGetSize(img) , 8 , 3);
  cvCvtColor(img , lab , CV_BGR2Lab);

  /** Cal the color ditribution */
  for (int i = 0; i < lab->height; ++i)
    for (int j = 0; j < lab->width; ++j) {
      int L = (int)cvGet2D(lab , i , j).val[0];
      int a = (int)cvGet2D(lab , i , j).val[1];
      int b = (int)cvGet2D(lab , i , j).val[2];
      /** First consider black , if L is lower than 64 , we thought its
       * black
       */
      if (L < 64) {
        count[0]++;
        continue;
      }

      /** Second we find the most nearest color to this pixel except
       * gray 
       */
      int minDis = 10000;
      int minTar = -1;
      for (int k = 2; k < TONES; ++k) {
        if (abs(a - LAB[k][1]) + abs(b - LAB[k][2]) < minDis) {
          minDis = abs(a - LAB[k][1]) + abs(b - LAB[k][2]);
          minTar = k;
        }
      }

      /** gray is consider specially */
      if (abs(a - LAB[1][1]) + abs(b - LAB[1][2]) < 8) {
        minTar = 1;
        minDis = abs(a - LAB[1][1]) + abs(b - LAB[1][2]);
      }

      count[minTar]++;
    }

  cvReleaseImage(&lab);

  int colorfulCount = 0;
  int averCount = 0;
  int maxCount = -1;
  int maxTone = -1;
  for (int i = 0; i < TONES; ++i) {
    averCount += count[i];
    if (count[i] > maxCount) {
      maxCount = count[i];
      maxTone = i;
    }
  }
  for (int i = 2; i < TONES; ++i) 
    colorfulCount += count[i];

  averCount /= TONES;

  /** if max tone is colorful color */
  if (maxTone >= 2) {
    if (maxCount >= threshold * averCount  && colorfulCount - maxCount <=
        averCount)
      return name[maxTone];
  }
  if (maxTone <= 1) {
    if (maxCount >= threshold * averCount && colorfulCount <= averCount)
      return name[maxTone];
  }
  return "neutral";
}

void Lab::init(IplImage* img, CvRect rect , CvMat * mask , int featureDim){
  IplImage * tmp = getPartialImage(img,rect);

  /** Cvt image to lab color space */
  IplImage* lab = cvCreateImage(cvGetSize(tmp), tmp->depth, 3);
  cvCvtColor(tmp, lab, CV_BGR2Lab);

  this->featureList = vector<int>(12 + 12 + 8);
  for (int i = 0; i < tmp->height; ++i)
    for (int j = 0; j < tmp->width; ++j)
    {
      int L = (int)cvGet2D(lab , i , j).val[0];
      int a = (int)cvGet2D(lab , i , j).val[1];
      int b = (int)cvGet2D(lab , i , j).val[2];
      int Ldim = L / 32;
      int adim = getABBucketID(a);
      int bdim = getABBucketID(b);

      featureList[Ldim]++;
      featureList[8 + adim]++;
      featureList[20 + bdim]++;
    }
  cvReleaseImage(&lab);
}

void Law::init(IplImage* img, CvRect rect, CvMat * mask , int featureDim){
  IplImage * tmp = getPartialImage(img,rect);

  vector<vector<int> >features;
  ImageFeature::extLaw(tmp, features);

  double lawData[18];
  for(int i = 0; i < 18; i++)
    lawData[i] = 0;
  vector<int>::const_iterator tIt[9];
  for(int i = 0; i < 9; i++){
    tIt[i]  = features[i].begin();
    int cnt = 0;
    for(; tIt[i] != features[i].end(); tIt[i]++){
      lawData[2*i] += *(tIt[i]);
      cnt++;
    }
    lawData[2*i] /= cnt;

    tIt[i] = features[i].begin(); 
    cnt = 0;
    for(; tIt[i] != features[i].end(); tIt[i]++){
      lawData[2*i+1] += (*(tIt[i])-lawData[2*i]) * (*(tIt[i])-lawData[2*i]); 
      cnt++;
    }
    lawData[2*i+1] /=cnt;
    lawData[2*i+1] = (int)sqrt((double)lawData[2*i+1]);
  }
  featureList = vector<int>(18);
  for (int i = 0; i < 18; ++i) featureList[i] = (int)lawData[i];

  cvReleaseImage(&tmp);
}

/////////////////////////////////////////////////////////////////////
// HSV Feature
/////////////////////////////////////////////////////////////////////
void Hsv::init(IplImage* img, CvRect rect , CvMat * mask , int featureDim) {
  IplImage * tmp = getPartialImage(img,rect);

  IplImage* hsv = cvCreateImage( cvGetSize(tmp), 8, 3 );
  IplImage* h_plane = cvCreateImage( cvGetSize(tmp), 8, 1 );
  IplImage* s_plane = cvCreateImage( cvGetSize(tmp),8,1);
  IplImage* v_plane = cvCreateImage( cvGetSize(tmp),8,1);
  cvCvtColor( tmp, hsv, CV_BGR2HSV );
  cvCvtPixToPlane( hsv, h_plane, s_plane, v_plane, 0 );

  int hDim = featureDim - 3;
  this->featureList = vector<int>(hDim + 3);
  for (int i=0;i<featureList.size();++i) featureList[i] = 0;

  /**
    (1) 黑色区域:所有v<15%的颜色均归入黑色,令h=0,s=0,v=0;
    (2) 白色区域:所有s<10%且v>80%的颜色归入白色,令h=0,s=0,v=1;
    (3) 彩色区域:位于黑色区域和白色区域以外的颜色,其h,s,v值保持不变.
    */
  int VLow = (int)(255 * 0.15);
  int VHigh =(int)(255 * 0.8);
  int SLow = (int)(255 * 0.1);
  for (int i=0;i<tmp->height;++i)
    for (int j=0;j<tmp->width;++j)
    {
      if (mask != NULL && cvmGet(mask,i + rect.x,j + rect.y)==0.0) continue;
      /** Black Color , low brightness */
      if (cvGet2D(v_plane,i,j).val[0]<VLow) 
      {

        featureList[0]++;
        continue;
      }
      /** White Color , high brightness , low saturation */
      if (cvGet2D(v_plane,i,j).val[0]>VHigh && cvGet2D(s_plane,i,j).val[0]<SLow)
      {
        featureList[1]++;
        continue;
      }
      /** Gray */
      if (cvGet2D(s_plane,i,j).val[0]<SLow) 
      {
        featureList[2]++;
        continue;
      }
      int bucket = (int)cvGet2D(h_plane,i,j).val[0];
      bucket = bucket * hDim / 181;
      featureList[bucket+3]++;
    }

  cvReleaseImage(&tmp);
  cvReleaseImage(&h_plane);
  cvReleaseImage(&s_plane);
  cvReleaseImage(&v_plane);
  cvReleaseImage(&hsv);
}


void AverageSD::init(IplImage * img , CvRect rect , CvMat * mask , int featureDim)
{
  IplImage * tmp = getPartialImage(img,rect);

  /** 输入图像转换到HSV颜色空间 */
  IplImage* hsv = cvCreateImage( cvGetSize(tmp), 8, 3 );
  IplImage* h_plane = cvCreateImage( cvGetSize(tmp), 8, 1 );
  IplImage* s_plane = cvCreateImage( cvGetSize(tmp),8,1);
  IplImage* v_plane = cvCreateImage( cvGetSize(tmp),8,1);
  cvCvtColor( tmp, hsv, CV_BGR2HSV );
  cvCvtPixToPlane( hsv, h_plane, s_plane, v_plane, 0 );

  this->featureList = vector<int>(featureDim);
  for (int i = 0; i < featureDim; ++i) featureList[i] = 0;
  double averR=0,averG=0,averB=0;
  double averH=0,averS=0,averV=0;
  double squH=0,squS=0,squV=0;
  double squR=0,squG=0,squB=0;

  int count = 0;
  for (int i=0;i<tmp->height;++i)
    for (int j=0;j<tmp->width;++j)
    {
      if (mask != NULL && cvmGet(mask,i + rect.x,j + rect.y)==0.0) continue;
      count ++;
      averH += cvGet2D(h_plane,i,j).val[0];
      averS += cvGet2D(s_plane,i,j).val[0];
      averV += cvGet2D(v_plane,i,j).val[0];
      averR += cvGet2D(tmp,i,j).val[2];
      averG += cvGet2D(tmp,i,j).val[1];
      averB += cvGet2D(tmp,i,j).val[0];
    }
  averH /= (count);
  averS /= (count);
  averV /= (count);
  averR /= (count);
  averG /= (count);
  averB /= (count);

  for (int i=0;i<tmp->height;++i)
    for (int j=0;j<tmp->width;++j)
    {
      if (mask != NULL && cvmGet(mask,i + rect.x,j + rect.y)==0.0) continue;
      squH += sqr(cvGet2D(h_plane,i,j).val[0]-averH);
      squS += sqr(cvGet2D(s_plane,i,j).val[0]-averS);
      squV += sqr(cvGet2D(v_plane,i,j).val[0]-averV);
      squR += sqr(cvGet2D(tmp,i,j).val[2]-averR);
      squG += sqr(cvGet2D(tmp,i,j).val[1]-averG);
      squB += sqr(cvGet2D(tmp,i,j).val[0]-averB);
    }

  squH /= (count);
  squS /= (count);
  squV /= (count);
  squR /= (count);
  squG /= (count);
  squB /= (count);

  squH = sqrt(squH);
  squS = sqrt(squS);
  squV = sqrt(squV);
  squR = sqrt(squR);
  squG = sqrt(squG);
  squB = sqrt(squB);

  this->featureList[0] = (int)(averH * 256 / 181);
  this->featureList[1] = (int)averS;
  this->featureList[2] = (int)averV;
  this->featureList[3] = (int)averR;
  this->featureList[4] = (int)averG;
  this->featureList[5] = (int)averB;

  this->featureList[6] = (int)(squH * 256 / 181);
  this->featureList[7] = (int)squS;
  this->featureList[8] = (int)squV;
  this->featureList[9] = (int)squR;
  this->featureList[10] = (int)squG;
  this->featureList[11] = (int)squB;

  cvReleaseImage(&tmp);
  cvReleaseImage(&hsv);
  cvReleaseImage(&h_plane);
  cvReleaseImage(&s_plane);
  cvReleaseImage(&v_plane);
}


/////////////////////////////////////////////////////////////////////
// HSV RGB LAW 混合Feature 生成
// R,G,B H,S,V 均值方差各两遍，LAW 一遍
/////////////////////////////////////////////////////////////////////
void Mix::init(IplImage* img, CvRect rect , CvMat * mask , int featureDim){

  Feature * hsvrgb = Feature::factory("averageSD");
  hsvrgb->init(img , rect , mask);

  this->featureList = vector<int>(24+18);
  for (int i = 0; i < 12; ++i)
    featureList[i] = featureList[i+12] = hsvrgb->featureList[i];

  Feature * feature = Feature::factory("law");
  feature->init(img , rect , mask);
  for (int i=24;i<this->featureList.size();++i)
    featureList[i] = feature->featureList[i-24];

  delete feature;
  delete hsvrgb;
}

//////////////////////////////////////////////////////////////
// 这是一个用来生成图像颜色hashcode的feature
//////////////////////////////////////////////////////////////
void Code::init(IplImage* img, CvRect rect , CvMat * mask , int featureDim){
  IplImage * tmp = getPartialImage(img,rect);

  int hDim = 12;
  Feature * featureHsv = Feature::factory("hsv");
  featureHsv->init(img , rect , mask , hDim);

  this->featureList = vector<int>(hDim+8);
  for (int i=0;i<featureList.size();++i) featureList[i] = 0;
  for (int i = 0; i < hDim; ++i) featureList[i] = featureHsv->featureList[i];

  Feature * featureAver = Feature::factory("averageSD");
  featureAver->init(img , rect , mask);

  this->featureList[hDim] = 0;//averR;
  this->featureList[hDim+1] = 0;//averG;
  this->featureList[hDim+2] = 0;//averB;
  this->featureList[hDim+3] = 0;//(squR+squG+squB) / 3;
  this->featureList[hDim+4] = featureAver->featureList[0];
  this->featureList[hDim+5] = featureAver->featureList[1];
  this->featureList[hDim+6] = featureAver->featureList[2];
  this->featureList[hDim+7] = (featureAver->featureList[6]+featureAver->featureList[7]+
      featureAver->featureList[8]) / 3;

  delete featureHsv;
  delete featureAver;

  LBP lbp;
  lbp.init(tmp);
  this->code2 = lbp.getIntHashCode();
  cvReleaseImage(&tmp);
}

void GIST::init(IplImage * img ,CvRect rect, CvMat * mask , int featureDim)
{
  IplImage * tmp = getPartialImage(img,rect);

  /** 指定计算gistFeature的exe */
  string gist = "D:\\User\\yhzhu\\Coding\\Matlab\\ImageFeature\\GIST\\gistFeature.exe";	
  cvSaveImage("C:\\gistTemp.jpg",tmp);
  system(gist.data());

  ifstream fin("C:\\gist.txt");
  char s[1000];
  fin.getline(s,500);
  fin.getline(s,500);
  int N;
  sscanf(s,"%d",&N);
  this->featureList = vector<int>(N);

  double feature;
  for (int i=0;i<N;++i)
  {
    fin.getline(s,500);
    sscanf(s,"%lf",&feature);
    featureList[i] = (int)(feature * 255);
  }

  fin.close();
  cvReleaseImage(&tmp);
}

long long Code::getLongHashCode()
{

  int hDim = 12;
  int sum = 0;
  for (int i=0;i<hDim;++i)
    sum += this->featureList[i];
  int average = sum / (hDim);

  long long code = 0;
  for (int k=0;k<hDim;++k)
  {
    int mark = 0;
    if (this->featureList[k] >= average /2)
      mark = 1;
    if (this->featureList[k] >= average)
      mark = 2;
    if (this->featureList[k] >= average * 2)
      mark = 3;
    code = (code <<2) + mark;
  }


  /** RGB average */

  for (int k=hDim;k<hDim+3;++k)
  {
    int mark = 0;
    if (featureList[k]>=64) mark = 1;
    if (featureList[k]>=128) mark = 2;
    if (featureList[k]>=192) mark = 3;
    code = (code <<2)+mark;
  }

  int mark = 0;
  if (featureList[hDim+3]>=32) mark = 1;
  if (featureList[hDim+3]>=64) mark = 2;
  if (featureList[hDim+3]>=128) mark = 3;
  code = (code <<2)+mark;

  mark = 0;
  if (featureList[hDim+4]>=45) mark = 1;
  if (featureList[hDim+4]>=90) mark = 2;
  if (featureList[hDim+4]>=135) mark = 3;
  code = (code <<2)+mark;


  /** SV average */

  for (int k=hDim+5;k<hDim+8;++k)
  {
    int mark = 0;
    if (featureList[k]>=64) mark = 1;
    if (featureList[k]>=128) mark = 2;
    if (featureList[k]>=192) mark = 3;
    code = (code <<2)+mark;
  }


  return (code<<50)|code2;
}

/////////////////////////////////////////////////////////////////////
// HSV Feature 生成
// 统计H分量的直方图 
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
// 返回给定整数的二进制表示下，循环移位能够产生的最小整数
/////////////////////////////////////////////////////////////////////
int getMinCirInt(int input,int dim)
{
  int answer = input;
  for (int i=0;i<dim;++i)
  {
    if (input & (1<<(dim-1))) 
    {
      input -= (1<<(dim-1));
      input <<= 1;
      input ++;
    }
    else
      input <<= 1;
    if (input<answer) answer = input;
  }
  return answer;
}

/////////////////////////////////////////////////////////////////////
// 返回给定整数的二进制表示下，0-1交换的个数
/////////////////////////////////////////////////////////////////////
int getChangeBitCount(int input)
{
  char s[30];
  itoa(input,s,2);
  int len = strlen(s);
  int ret = 0;
  for (int i=0;i<len-1;++i)
    if (s[i] != s[i+1]) ret++;
  if (s[0] != s[len-1]) ret++;
  return ret;
}

////////////////////////////////////////////////////////////////////
// 返回给定整数的二进制表示下，1出现的个数
////////////////////////////////////////////////////////////////////
int get1Count(int input)
{
  char s[30];
  itoa(input,s,2);
  int len = strlen(s);
  int ret = 0;
  for (int i=0;i<len;++i)
    if (s[i] == '1') ret++;
  return ret;
}

struct MappingTable
{
  map<int,int> table;

  MappingTable(int dim)
  {
    int start = 0;
    int stop = (1<<dim) -1;
    for (int i=start;i<=stop;++i)
    {
      int pattern = getMinCirInt(i,dim);
      int change = getChangeBitCount(i);
      if (change<=2)
      {
        int ones = get1Count(i);
        table[i] = ones;
      }
      else
        table[i] = dim+1;
    }
  }
}table8(8);


void LBP::init(IplImage* img,CvRect rect,CvMat * mask , int featureDim){
  IplImage * tmp = getPartialImage(img,rect);

  IplImage * gray;
  gray = cvCreateImage(cvSize(tmp->width,tmp->height),8,1);
  cvCvtColor(tmp,gray,CV_BGR2GRAY);

  int neighbours = 8;
  int grayLevel = 4;
  int patterns = neighbours+2;
  vector<int> ret(patterns*grayLevel);
  for (int i=0;i<patterns*grayLevel;++i) ret[i] = 0;

  int dir[8][2]  = {{0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{1,-1},{1,0},{1,1}};
  for (int i=1;i<=gray->height-2;++i)
    for (int j=1;j<=gray->width-2;++j)
    {
      int gc = (int)cvGet2D(gray,i,j).val[0];
      int code = 0;
      int average = 0;
      for (int now=0;now<8;++now)
      {
        average += (int)cvGet2D(gray,i+dir[now][0],j+dir[now][1]).val[0];
        int delta = (int)cvGet2D(gray,i+dir[now][0],j+dir[now][1]).val[0] - gc;
        if (delta>=0)
          code = (code<<1) + 1;
        else
          code = (code<<1);
      }
      /** 计算均方差 */
      average /= 8;
      int var = 0;
      for (int now=0;now<8;++now)
        var += (int)((cvGet2D(gray,i+dir[now][0],j+dir[now][1]).val[0] - average) *
            (cvGet2D(gray,i+dir[now][0],j+dir[now][1]).val[0] - average));
      var /= 8;
      var = (int)sqrt((double)var);

      int nowenergy = 0;
      if (var<=32) nowenergy = 0;
      else if (var<=64) nowenergy = 1;
      else if (var<=128) nowenergy =2;
      else nowenergy = 3;

      int realcode = table8.table[code];
      ret[nowenergy * patterns + realcode]++;
    }

  this->featureList = ret;

  cvReleaseImage(&gray);
  cvReleaseImage(&tmp);
}

LAW::LAW(){
  for(int k = 0; k < 9; k++){
    double mean = 0.0;
    for(int i = 0; i < 5; i++) for(int j = 0; j < 5; j++){
      mean += lawMask[k][i][j];
    }
    mean = mean/25.0;
    for(int i = 0; i < 5; i++) for(int j = 0; j < 5; j++){
      filters[k][i][j] = lawMask[k][i][j]-mean;
    }
    double L1 = 0.0;
    for(int i = 0; i < 5; i++) for(int j = 0; j < 5; j++){
      L1 += fabs(filters[k][i][j]);
    }
    for(int i = 0; i < 5; i++) for(int j = 0; j < 5; j++){
      filters[k][i][j] /= L1;
    }
  }
}

void LAW::filter(IplImage* img, IplImage* feature[]){
  int height = img->height; int width = img->width;
  int step = img->widthStep;
  uchar* data = (uchar*)img->imageData;
  uchar* ftData[9];
  for(int i = 0; i < 9; i++){
    ftData[i] = (uchar*)feature[i]->imageData;
  }
  for(int i = 2; i < height-2; i++) for(int j = 2; j < width-2; j++){
    double ele[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for(int ti = i-2; ti <=i+2; ti++) for(int tj = j-2; tj <=j+2; tj++){
      for(int k = 0; k < 9; k++)
        ele[k] += filters[k][ti-i+2][tj-j+2]*data[ti*step + tj];
    }
    for(int k = 0; k < 9; k++)
      ftData[k][i*step + j] = (uchar)ele[k];
  }

}

//////////////////////////////////////////////////////////////////////////////////////////
// Ext Image Lab Feature 
//////////////////////////////////////////////////////////////////////////////////////////
void ImageFeature::extLab(IplImage* img, vector<vector<int> >& features){
  features.clear();

  int height = img->height; int width = img->width;
  IplImage* lab = cvCreateImage(cvGetSize(img), img->depth, 3);
  uchar* data = (uchar*)lab->imageData;
  int step = lab->widthStep;
  int channels = lab->nChannels;

  cvCvtColor(img, lab, CV_BGR2Lab);
  vector<int> l;
  vector<int> a; 
  vector<int> b;
  for(int i = 0; i < height; i++) for(int j= 0; j < width; j++){
    l.push_back(data[i*step + j*channels + 0]);
    a.push_back(data[i*step + j*channels + 1]);
    b.push_back(data[i*step + j*channels + 2]);
  }
  features.push_back(l);
  features.push_back(a);
  features.push_back(b);

  cvReleaseImage(&lab);
}

void ImageFeature::extLaw(IplImage* img, vector<vector<int> >& features){
  features.clear();

  int height = img->height; int width = img->width;
  IplImage* gray = cvCreateImage(cvGetSize(img), img->depth, 1);
  int step = gray->widthStep;
  cvCvtColor(img, gray, CV_BGR2GRAY);

  LAW law;
  IplImage* filters[9];
  for(int i = 0; i < 9; i++)
    filters[i] = cvCreateImage(cvGetSize(gray), gray->depth, 1);

  law.filter(gray, filters);
  for(int k = 0; k < 9; k++){
    uchar* data = (uchar*)(filters[k]->imageData);
    vector<int> tmp;
    for(int i = 0; i < height; i++) for(int j = 0; j < width; j++){
      tmp.push_back(data[i*step + j]);
    }
    features.push_back(tmp);
  }

  for(int k = 0; k < 9; k++){
    cvReleaseImage(&filters[k]);
  }

  cvReleaseImage(&gray);
}

void ImageFeature::extTotal(IplImage* img, vector<vector<int> >& features){
  features.clear();

  int height = img->height;
  int width = img->width;
  uchar* data = (uchar*)img->imageData;
  int step = img->widthStep; int channels = img->nChannels;

  vector<int> tmp;
  for(int i = 0; i < height; i++) for(int j = 0; j < width; j++)
    for(int k = 0; k < 3; k++){
      tmp.push_back(data[i*step + j*channels + k]);
    }
  features.push_back(tmp);
}

///////////////////////////////////////////////////////////////////////////////////
// Ext Image HSV Feature 
///////////////////////////////////////////////////////////////////////////////////
void ImageFeature::extHsv(IplImage* srcImage, vector<vector<int> >& features){
  features.clear();

  /** 输入图像转换到HSV颜色空间 */
  IplImage* hsv = cvCreateImage( cvGetSize(srcImage), 8, 3 );
  IplImage* h_plane = cvCreateImage( cvGetSize(srcImage), 8, 1 );
  IplImage* s_plane = cvCreateImage( cvGetSize(srcImage),8,1);
  IplImage* v_plane = cvCreateImage( cvGetSize(srcImage),8,1);
  cvCvtColor( srcImage, hsv, CV_BGR2HSV );
  cvCvtPixToPlane( hsv, h_plane, s_plane, v_plane, 0 );

  vector<int> H;
  vector<int> S;
  vector<int> V;
  for (int i=0;i<srcImage->height;++i)
  {
    for(int j=0;j<srcImage->width;++j)
    {
      H.push_back((int)cvGet2D(h_plane,i,j).val[0]);
      S.push_back((int)cvGet2D(s_plane,i,j).val[0]);
      V.push_back((int)cvGet2D(v_plane,i,j).val[0]);
    }
  }
  features.push_back(H);
  features.push_back(S);
  features.push_back(V);
  cvReleaseImage(&hsv);
  cvReleaseImage(&h_plane);
  cvReleaseImage(&s_plane);
  cvReleaseImage(&v_plane);
}


///////////////////////////////////////////////////////////////////////////////
// 给定一张输入图片，返回他的 Hashcode
///////////////////////////////////////////////////////////////////////////////
long long getImageHashCode(IplImage * img)
{
  Code code;
  code.init(img);
  return code.getLongHashCode();
}

/**
vector<long long> getImageHashCodes(IplImage * srcImage,int rowblocks,
    int rowblockpixels,int way)
{
  int histos = rowblocks * rowblocks;
  vector<long long> ret(histos);

  for (int i=0;i<rowblocks;++i)
    for (int j=0;j<rowblocks;++j)
    {
      IplImage * smallImage = getSuperPixel(srcImage,rowblocks,rowblockpixels,i,j);
      if (way ==1)
      {
        Code code;
        code.init(smallImage);
        ret[i*rowblocks +j] = code.getIntHashCode();
        //ret[i*rowblocks +j] = feature->getLongHashCode();
      }
      if (way==2)
      {
        LBP lbp;;
        lbp.init(smallImage);
        ret[i*rowblocks +j] = lbp.getIntHashCode();
        //ret[i*rowblocks+j] = 0;
      }
      cvReleaseImage(&smallImage);
    }
  return ret;
}
*/

int LBP::getIntHashCode()
{
  int patterns = 10;
  vector<int> count(patterns);
  for (int i=0;i<patterns;++i) count[i] = 0;
  for (int i=0;i<this->featureList.size();++i)
    count[i % patterns] += featureList[i];

  int sum = 0;
  for (int i=0;i<patterns;++i)
    sum += count[i];
  int average = sum / patterns;

  int code = 0;
  for (int k=0;k<patterns;++k)
  {
    int mark = 0;
    if (count[k]>= average * 2.5) mark = 1;
    int realmark = mark << (k);
    code = code | realmark;
  }

  /** 计算平均Gray Level */
  int level = 4;
  vector<int> grayCount(level);
  for (int i=0;i<level;++i) grayCount[i] = 0;
  for (int i=0;i<this->featureList.size();++i)
    grayCount[i / patterns] += featureList[i];

  int maxlevel = -1;
  int maxlevelvalue = -1;
  for (int i=0;i<level;++i) if (grayCount[i]>maxlevelvalue)
  {
    maxlevelvalue = grayCount[i];
    maxlevel = i;
  }
  code = code << 2;
  code = code | maxlevel;

  return code;
}

int Feature::getIntHashCode()
{
  int sum = 0;
  for (int i=0;i<this->featureList.size();++i)
    sum += this->featureList[i];
  int average = sum / this->featureList.size();

  int code = 0;
  if (this->featureList.size()<=16)
  {
    for (int k=0;k<featureList.size();++k)
    {
      int mark = 0;
      if (featureList[k]>= average/2) mark = 1;
      if (featureList[k]>= average) mark = 2;
      if (featureList[k]>= average*2) mark = 3;
      int realmark = mark << (k*2);
      code = code | realmark;
    }
    return code;
  }

  if (this->featureList.size()<=32)
  {
    for (int k=0;k<featureList.size();++k)
    {
      int mark = 0;
      if (featureList[k]>= average * 2) mark = 1;
      int realmark = mark << (k);
      code = code | realmark;
    }
    return code;
  }

  double bucketlen = (double)featureList.size() / 32.0;
  average = (int)((double)average * bucketlen);

  int k = 0;
  for (double start = 0;start<featureList.size();start+=bucketlen)
  {
    int s = (int)start;
    int t = (int)(start+bucketlen);
    int nowsum = 0;
    for (int i=s;i<t;++i) nowsum += featureList[i];
    int mark = 0;
    if (nowsum>= average * 2) mark = 1;
    int realmark = mark << (k);
    k++;
    code = code | realmark;
  }
  return code;
}

long long Feature::getLongHashCode()
{
  int sum = 0;
  for (int i=0;i<this->featureList.size();++i)
    sum += this->featureList[i];
  int average = sum / this->featureList.size();

  long long code = 0;
  if (this->featureList.size()<=32)
  {
    for (int k=0;k<featureList.size();++k)
    {
      long mark = 0;
      if (featureList[k]>= average/2) mark = 1;
      if (featureList[k]>= average) mark = 2;
      if (featureList[k]>= average*2) mark = 3;
      long realmark = mark << (k*2);
      code = code | realmark;
    }
    return code;
  }

  if (this->featureList.size()<=64)
  {
    for (int k=0;k<featureList.size();++k)
    {
      long mark = 0;
      if (featureList[k]>= average * 2) mark = 1;
      long realmark = mark << k;
      code = code | realmark;
    }
  }
  return code;
}


vector<double> getHueHistogram(IplImage * img, IplImage * mask)
{
  IplImage* hsv = cvCreateImage( cvGetSize(img), 8, 3 );
  IplImage* hue = cvCreateImage( cvGetSize(img), 8, 1 );

  //Convert img to the HSV color space
  cvCvtColor(img, hsv, CV_BGR2HSV);

  //Split out hue component and store in hue
  cvSplit(hsv, hue, 0, 0, 0);

  //Represents how many hues to cover in histogram, here 180 degrees
  int num_bins = 18;
  vector<double> ret(num_bins, 0);
  int total = 0;

  unsigned char * hData = (unsigned char *) hue->imageData;
  int hstep = hue->widthStep;
  unsigned char * maskData = (unsigned char *) mask->imageData;
  int mstep = mask->widthStep;

  for (int i = 0; i < hue->height; ++i)
    for (int j = 0; j < hue->width; ++j)
      if (maskData[i * mstep + j] != 0)
      {
        total++;
        int h = hData[i * hstep + j];
        int b = h / 10;
        if (b >= num_bins) b = num_bins - 1;
        ret[b]++;
      }

  cvReleaseImage(&hsv);
  cvReleaseImage(&hue);

  if (total > 0)
    for (int i = 0; i < ret.size(); ++i)
      ret[i] = ret[i] / total;

  return ret;
}

#endif

