#ifndef gGabor_H
#define gGabor_H
#include "myImg.h"
#include "singleImg.h"
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "createGabor.h"
#include "prefilt.h"
#include <vector>
using namespace std;

void gistGabor(myImg gImg,int w,double ***G,int n,int Nfilters,double **&g);
void downN(singleImg x,int NN,double** &);
void getnxny(int h,int w,int num);

vector<double> getGIST(IplImage * src);

#endif