#define OPENCV

#ifdef OPENCV
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include "Util.h"
void cvRectangle(IplImage * img, CvRect & rect, CvScalar & color, int w)
{
	CvPoint left = cvPoint(rect.x, rect.y);
	CvPoint right = cvPoint(rect.x + rect.width - 1, rect.y + rect.height - 1);
	cvRectangle(img, left, right, color, w);
}
#endif

#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <set>
#include <queue>
#include <climits>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>

using namespace std;
#define debug1(x) cout << #x" = " << x << endl;
#define debug2(x, y) cout << #x" = " << x << " " << #y" = " << y << endl;
#define debug3(x, y, z) cout << #x" = " << x << " " << #y" = " << y << " " << #z" = " << z << endl;
#define debug4(x, y, z, w) cout << #x" = " << x << " " << #y" = " << y << " " << #z" = " << z << " " << #w" = " << w << endl;

typedef unsigned char uchar;
inline double sqr(double x) { return x * x; }

#ifndef OPENCV
const int CV_BGR2GRAY = 1;

struct CvScalar
{
	double val[4];
};

CvScalar cvScalar(double a, double b, double c)
{ 
	CvScalar ret;
	ret.val[0] = a;
	ret.val[1] = b;
	ret.val[2] = c;
	return ret;
}

struct CvPoint
{
	int x, y;
};

CvPoint cvPoint(int x, int y) 
{
	CvPoint ret;
	ret.x = x;
	ret.y = y;
	return ret;
}

struct CvMat
{
	int data;
	int height;
	int width;
};

struct CvRect
{
	int x, y;
	int height, width;
};

CvRect cvRect(int x, int y, int w, int h)  
{
	CvRect ret; ret.x = x; ret.y = y; ret.height = h; ret.width = w;
	return ret;
}

struct CvSize
{
	int width;
	int height;
	CvSize(int w, int h)
	{ width = w; height = h; }
};
CvSize cvSize(int w, int h) { return CvSize(w, h); }

struct IplImage
{
	unsigned char * imageData;
	int height;
	int width;
	int widthStep;
	int nChannels;
	int depth;
	IplImage (CvSize size, int depth, int channels)
	{
		height = size.height;
		width = size.width;
		nChannels = channels;
		widthStep = channels * width;
		imageData = new unsigned char[height * width * channels];
	}

	~IplImage()
	{
		delete [] imageData;
	}
};

IplImage * cvCreateImage(CvSize size, int depth, int channels)
{
	IplImage * img = new IplImage(size, depth, channels);
	return img;
}

void cvReleaseImage(IplImage ** img)
{
	delete *img;
}

void cvReleaseMat(CvMat ** mat)
{ 
	delete *mat; 
}

CvSize cvGetSize(IplImage * img) { return cvSize(img->width, img->height); }

void cvCvtColor(IplImage * img, IplImage * gray, int code)
{
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
}

void cvShowImage(char * name, IplImage * srcImage)
{
}

CvScalar CV_RGB(int r, int g, int b)
{
	 return cvScalar(r, g, b);
}

void cvRectangle(IplImage * img, CvRect rect, CvScalar color, int w)
{
	return;
}

#endif
int threshold(IplImage * src, IplImage * dst, int thres, int value, int type)
{
	int stepSrc = src->widthStep;
	int step = dst->widthStep;
	int ret = 0;
	unsigned char * srcData = (unsigned char *) src->imageData;
	unsigned char * data = (unsigned char *) dst->imageData;
	for (int i = 0; i < src->height; ++i)
		for (int j = 0; j < src->width; ++j)
		{
			int d = srcData[i * stepSrc + j];
			if (d >= thres && type == 0 || d <= thres && type == 1) 
			{
				data[i * step + j] = 255;
				ret++;
			}
			else 
				data[i * step + j] = 0;
		}
	return ret;
}

// 判断两个矩形是否相交，返回相交的面积
int getRectIntersect(CvRect & rect1, CvRect & rect2)
{
	int x1 = rect1.x;
	int x2 = rect1.x + rect1.width - 1;
	int y1 = rect1.y;
	int y2 = rect1.y + rect1.height - 1;

	int ox1 = rect2.x;
	int ox2 = rect2.x + rect2.width - 1;
	int oy1 = rect2.y;
	int oy2 = rect2.y + rect2.height - 1;

	int maxx = max(x1, ox1);
	int minx = min(x2, ox2);
	int maxy = max(y1, oy1);
	int miny = min(y2, oy2);

	if (maxx <= minx && maxy <= miny)
		return (minx - maxx + 1) * (miny - maxy + 1);
	else
		return 0;
}

inline double dist(CvPoint & p1, CvPoint & p2)
{
	return sqrt((double) (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

inline int getRectArea(CvRect & rect)
{ return rect.height * rect.width; }

int sum[1600][1600];
void getBinaryImage(IplImage * img, int blocksize, int param, IplImage * ret, bool issmall = true)
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

			if ((issmall && now < average) || (!issmall && now > average))
				retdata[(i - 1) * stepret + j - 1] = 255;
			else
				retdata[(i - 1) * stepret + j - 1] = 0;
		}
}


ostream & operator << (ostream & out, const CvPoint & p)
{
	out << "(" << p.x << "," << p.y << ")";
	return out;
}

ostream & operator << (ostream & out, CvRect & rect)
{
	out << "(" << rect.x << "," << rect.y << ") " << rect.width << " " << rect.height;
	return out;
}

bool operator < (const CvRect & rect1, const CvRect & rect2)
{ 
	return false;
}

//////////////////////// BEGIN SEGMENTATION /////////
template <class T>
class image {
public:
	/* create an image */
	image(const int width, const int height, const bool init = true);

	/* delete an image */
	~image();

	/* init an image */
	void init(const T &val);

	/* copy an image */
	image<T> *copy() const;

	/* get the width of an image. */
	int width() const { return w; }

	/* get the height of an image. */
	int height() const { return h; }

	/* image data. */
	T *data;

	/* row pointers. */
	T **access;

private:
	int w, h;
};

/* use imRef to access image data. */
#define imRef(im, x, y) (im->access[y][x])

/* use imPtr to get pointer to image data. */
#define imPtr(im, x, y) &(im->access[y][x])

template <class T>
image<T>::image(const int width, const int height, const bool init) {
	w = width;
	h = height;
	data = new T[w * h];  // allocate space for image data
	access = new T*[h];   // allocate space for row pointers

	// initialize row pointers
	for (int i = 0; i < h; i++)
		access[i] = data + (i * w);  

	if (init)
		memset(data, 0, w * h * sizeof(T));
}

template <class T>
image<T>::~image() {
	delete [] data; 
	delete [] access;
}

template <class T>
void image<T>::init(const T &val) {
	T *ptr = imPtr(this, 0, 0);
	T *end = imPtr(this, w-1, h-1);
	while (ptr <= end)
		*ptr++ = val;
}


template <class T>
image<T> *image<T>::copy() const {
	image<T> *im = new image<T>(w, h, false);
	memcpy(im->data, data, w * h * sizeof(T));
	return im;
}

template <class T>
void min_max(image<T> *im, T *ret_min, T *ret_max) {
	int width = im->width();
	int height = im->height();

	T min = imRef(im, 0, 0);
	T max = imRef(im, 0, 0);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			T val = imRef(im, x, y);
			if (min > val)
				min = val;
			if (max < val)
				max = val;
		}
	}

	*ret_min = min;
	*ret_max = max;
} 

/* threshold image */
template <class T>
image<uchar> *threshold(image<T> *src, int t) {
	int width = src->width();
	int height = src->height();
	image<uchar> *dst = new image<uchar>(width, height);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			imRef(dst, x, y) = (imRef(src, x, y) >= t);
		}
	}

	return dst;
}



#ifndef M_PI
#define M_PI 3.141592653589793
#endif

struct rgb {
	uchar r, g, b; 
	rgb(uchar R,uchar G,uchar B)
	{
		r=R;g=G;b=B;
	}
	rgb()
	{
		r=0;g=0;b=0;
	}
} ;

inline bool operator==(const rgb &a, const rgb &b) {
	return ((a.r == b.r) && (a.g == b.g) && (a.b == b.b));
}

template <class T>
inline T abs(const T &x) { return (x > 0 ? x : -x); };

template <class T>
inline int sign(const T &x) { return (x >= 0 ? 1 : -1); };

template <class T>
inline T square(const T &x) { return x*x; };

template <class T>
inline T bound(const T &x, const T &min, const T &max) {
	return (x < min ? min : (x > max ? max : x));
}

template <class T>
inline bool check_bound(const T &x, const T&min, const T &max) {
	return ((x < min) || (x > max));
}

inline int vlib_round(float x) { return (int)(x + 0.5F); }

inline int vlib_round(double x) { return (int)(x + 0.5); }

inline double gaussian(double val, double sigma) {
	return exp(-square(val/sigma)/2)/(sqrt(2*M_PI)*sigma);
}

#define	RED_WEIGHT	0.299
#define GREEN_WEIGHT	0.587
#define BLUE_WEIGHT	0.114

static image<uchar> *imageRGBtoGRAY(image<rgb> *input) {
	int width = input->width();
	int height = input->height();
	image<uchar> *output = new image<uchar>(width, height, false);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			imRef(output, x, y) = (uchar)
				(imRef(input, x, y).r * RED_WEIGHT +
				imRef(input, x, y).g * GREEN_WEIGHT +
				imRef(input, x, y).b * BLUE_WEIGHT);
		}
	}
	return output;
}

static image<rgb> *imageGRAYtoRGB(image<uchar> *input) {
	int width = input->width();
	int height = input->height();
	image<rgb> *output = new image<rgb>(width, height, false);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			imRef(output, x, y).r = imRef(input, x, y);
			imRef(output, x, y).g = imRef(input, x, y);
			imRef(output, x, y).b = imRef(input, x, y);
		}
	}
	return output;  
}

static image<float> *imageUCHARtoFLOAT(image<uchar> *input) {
	int width = input->width();
	int height = input->height();
	image<float> *output = new image<float>(width, height, false);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			imRef(output, x, y) = imRef(input, x, y);
		}
	}
	return output;  
}

static image<float> *imageINTtoFLOAT(image<int> *input) {
	int width = input->width();
	int height = input->height();
	image<float> *output = new image<float>(width, height, false);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			imRef(output, x, y) = (float)imRef(input, x, y);
		}
	}
	return output;  
}

static image<uchar> *imageFLOATtoUCHAR(image<float> *input, 
	float min, float max) {
		int width = input->width();
		int height = input->height();
		image<uchar> *output = new image<uchar>(width, height, false);

		if (max == min)
			return output;

		float scale = UCHAR_MAX / (max - min);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				uchar val = (uchar)((imRef(input, x, y) - min) * scale);
				imRef(output, x, y) = bound(val, (uchar)0, (uchar)UCHAR_MAX);
			}
		}
		return output;
}

static image<uchar> *imageFLOATtoUCHAR(image<float> *input) {
	float min, max;
	min_max(input, &min, &max);
	return imageFLOATtoUCHAR(input, min, max);
}

static image<long> *imageUCHARtoLONG(image<uchar> *input) {
	int width = input->width();
	int height = input->height();
	image<long> *output = new image<long>(width, height, false);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			imRef(output, x, y) = imRef(input, x, y);
		}
	}
	return output;  
}

static image<uchar> *imageLONGtoUCHAR(image<long> *input, long min, long max) {
	int width = input->width();
	int height = input->height();
	image<uchar> *output = new image<uchar>(width, height, false);

	if (max == min)
		return output;

	float scale = UCHAR_MAX / (float)(max - min);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			uchar val = (uchar)((imRef(input, x, y) - min) * scale);
			imRef(output, x, y) = bound(val, (uchar)0, (uchar)UCHAR_MAX);
		}
	}
	return output;
}

static image<uchar> *imageLONGtoUCHAR(image<long> *input) {
	long min, max;
	min_max(input, &min, &max);
	return imageLONGtoUCHAR(input, min, max);
}

static image<uchar> *imageSHORTtoUCHAR(image<short> *input, 
	short min, short max) {
		int width = input->width();
		int height = input->height();
		image<uchar> *output = new image<uchar>(width, height, false);

		if (max == min)
			return output;

		float scale = UCHAR_MAX / (float)(max - min);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				uchar val = (uchar)((imRef(input, x, y) - min) * scale);
				imRef(output, x, y) = bound(val, (uchar)0, (uchar)UCHAR_MAX);
			}
		}
		return output;
}

static image<uchar> *imageSHORTtoUCHAR(image<short> *input) {
	short min, max;
	min_max(input, &min, &max);
	return imageSHORTtoUCHAR(input, min, max);
}

unsigned char Random()
{
	return (unsigned char)(rand() % 256);
}

// Random color
rgb Random_rgb(){ 
	rgb c;

	c.r = (unsigned char)Random();
	c.g = (unsigned char)Random();
	c.b = (unsigned char)Random();

	return c;
}

// dissimilarity measure between pixels
static inline float diff(image<float> *r, image<float> *g, image<float> *b,
	int x1, int y1, int x2, int y2) {
		return sqrt(square(imRef(r, x1, y1)-imRef(r, x2, y2)) +
			square(imRef(g, x1, y1)-imRef(g, x2, y2)) +
			square(imRef(b, x1, y1)-imRef(b, x2, y2)));
}


using namespace std;

/* convolve src with mask.  dst is flipped! */
static void convolve_even(image<float> *src, image<float> *dst, 
	std::vector<float> &mask) {
		int width = src->width();
		int height = src->height();
		int len = (int)mask.size();

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				float sum = mask[0] * imRef(src, x, y);
				for (int i = 1; i < len; i++) {
					sum += mask[i] * (imRef(src, max(x-i,0), y) + 
						imRef(src, min(x+i, width-1), y));
				}
				imRef(dst, y, x) = sum;
			}
		}
}

/* convolve src with mask.  dst is flipped! */
static void convolve_odd(image<float> *src, image<float> *dst, 
	std::vector<float> &mask) {
		int width = src->width();
		int height = src->height();
		int len = (int)mask.size();

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				float sum = mask[0] * imRef(src, x, y);
				for (int i = 1; i < len; i++) {
					sum += mask[i] * 
						(imRef(src, max(x-i,0), y) - 
						imRef(src, min(x+i, width-1), y));
				}
				imRef(dst, y, x) = sum;
			}
		}
}


#define WIDTH 4.0

/* normalize mask so it integrates to one */
static void normalize(std::vector<float> &mask) {
	int len = (int)mask.size();
	float sum = 0;
	for (int i = 1; i < len; i++) {
		sum += fabs(mask[i]);
	}
	sum = 2*sum + fabs(mask[0]);
	for (int i = 0; i < len; i++) {
		mask[i] /= sum;
	}
}

/* make filters */
#define MAKE_FILTER(name, fun)                                \
	static std::vector<float> make_ ## name (float sigma) {       \
	sigma = max(sigma, 0.01F);			      \
	int len = (int)ceil(sigma * WIDTH) + 1;                     \
	std::vector<float> mask(len);                               \
	for (int i = 0; i < len; i++) {                             \
	mask[i] = fun;                                            \
	}                                                           \
	return mask;                                                \
}

MAKE_FILTER(fgauss, exp(-0.5*square(i/sigma)));

/* convolve image with gaussian filter */
static image<float> *smooth(image<float> *src, float sigma) {
	std::vector<float> mask = make_fgauss(sigma);
	normalize(mask);

	image<float> *tmp = new image<float>(src->height(), src->width(), false);
	image<float> *dst = new image<float>(src->width(), src->height(), false);
	convolve_even(src, tmp, mask);
	convolve_even(tmp, dst, mask);

	delete tmp;
	return dst;
}

/* convolve image with gaussian filter */
image<float> *smooth(image<uchar> *src, float sigma) {
	image<float> *tmp = imageUCHARtoFLOAT(src);
	image<float> *dst = smooth(tmp, sigma);
	delete tmp;
	return dst;
}

/* compute laplacian */
static image<float> *laplacian(image<float> *src) {
	int width = src->width();
	int height = src->height();
	image<float> *dst = new image<float>(width, height);  

	for (int y = 1; y < height-1; y++) {
		for (int x = 1; x < width-1; x++) {
			float d2x = imRef(src, x-1, y) + imRef(src, x+1, y) -
				2*imRef(src, x, y);
			float d2y = imRef(src, x, y-1) + imRef(src, x, y+1) -
				2*imRef(src, x, y);
			imRef(dst, x, y) = d2x + d2y;
		}
	}
	return dst;
}


// disjoint-set forests using union-by-rank and path compression (sort of).

typedef struct {
	int rank;
	int p;
	int size;
} uni_elt;

class universe {
public:
	universe(int elements);
	~universe();
	int find(int x);  
	void join(int x, int y);
	int size(int x) const { return elts[x].size; }
	int num_sets() const { return num; }

private:
	uni_elt *elts;
	int num;
};

universe::universe(int elements) {
	elts = new uni_elt[elements];
	num = elements;
	for (int i = 0; i < elements; i++) {
		elts[i].rank = 0;
		elts[i].size = 1;
		elts[i].p = i;
	}
}

universe::~universe() {
	delete [] elts;
}

int universe::find(int x) {
	int y = x;
	while (y != elts[y].p)
		y = elts[y].p;
	elts[x].p = y;
	return y;
}

void universe::join(int x, int y) {
	if (elts[x].rank > elts[y].rank) {
		elts[y].p = x;
		elts[x].size += elts[y].size;
	} else {
		elts[x].p = y;
		elts[y].size += elts[x].size;
		if (elts[x].rank == elts[y].rank)
			elts[y].rank++;
	}
	num--;
}

//#include "Seg_disjoint-set.h"

// threshold function
#define THRESHOLD(size, c) (c/size)

typedef struct {
	float w;
	int a, b;
} edge;

bool operator<(const edge &a, const edge &b) {
	return a.w < b.w;
}

/*
* Segment a graph
*
* Returns a disjoint-set forest representing the segmentation.
*
* num_vertices: number of vertices in graph.
* num_edges: number of edges in graph
* edges: array of edges.
* c: constant for treshold function.
*/
universe *segment_graph(int num_vertices, int num_edges, edge *edges, 
	float c) { 
		// sort edges by weight
		std::sort(edges, edges + num_edges);

		// make a disjoint-set forest
		universe *u = new universe(num_vertices);

		// init thresholds
		float *threshold = new float[num_vertices];
		for (int i = 0; i < num_vertices; i++)
			threshold[i] = THRESHOLD(1,c);

		// for each edge, in non-decreasing weight order...
		for (int i = 0; i < num_edges; i++) {
			edge *pedge = &edges[i];

			// components conected by this edge
			int a = u->find(pedge->a);
			int b = u->find(pedge->b);
			if (a != b) {
				if ((pedge->w <= threshold[a]) &&
					(pedge->w <= threshold[b])) {
						u->join(a, b);
						a = u->find(a);
						threshold[a] = pedge->w + THRESHOLD(u->size(a), c);
				}
			}
		}

		// free up
		delete threshold;
		return u;
}

/*
* Segment an image
*
* Returns a color image representing the segmentation.
*
* im: image to segment.
* sigma: to smooth the image.
* c: constant for treshold function.
* min_size: minimum component size (enforced by post-processing stage).
* num_ccs: number of connected components in the segmentation.
*/
image<rgb> *segment_image(image<rgb> *im, float sigma, float c, int min_size,
	int *num_ccs, vector<vector<int> > & comID) {
		int width = im->width();
		int height = im->height();

		image<float> *r = new image<float>(width, height);
		image<float> *g = new image<float>(width, height);
		image<float> *b = new image<float>(width, height);

		// smooth each color channel  
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				imRef(r, x, y) = imRef(im, x, y).r;
				imRef(g, x, y) = imRef(im, x, y).g;
				imRef(b, x, y) = imRef(im, x, y).b;
			}
		}
		image<float> *smooth_r = smooth(r, sigma);
		image<float> *smooth_g = smooth(g, sigma);
		image<float> *smooth_b = smooth(b, sigma);
		delete r;
		delete g;
		delete b;

		// build graph
		edge *edges = new edge[width*height*4];
		int num = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (x < width-1) {
					edges[num].a = y * width + x;
					edges[num].b = y * width + (x+1);
					edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y);
					num++;
				}

				if (y < height-1) {
					edges[num].a = y * width + x;
					edges[num].b = (y+1) * width + x;
					edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y+1);
					num++;
				}

				if ((x < width-1) && (y < height-1)) {
					edges[num].a = y * width + x;
					edges[num].b = (y+1) * width + (x+1);
					edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y+1);
					num++;
				}

				if ((x < width-1) && (y > 0)) {
					edges[num].a = y * width + x;
					edges[num].b = (y-1) * width + (x+1);
					edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y-1);
					num++;
				}
			}
		}
		delete smooth_r;
		delete smooth_g;
		delete smooth_b;

		// segment
		universe *u = segment_graph(width*height, num, edges, c);

		// post process small components
		for (int i = 0; i < num; i++) {
			int a = u->find(edges[i].a);
			int b = u->find(edges[i].b);
			if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
				u->join(a, b);
		}
		delete [] edges;
		*num_ccs = u->num_sets();

		image<rgb> *output = new image<rgb>(width, height);

		map<int,int> regionID;
		int count = 1;

		// pick Random colors for each component
		rgb *colors = new rgb[width*height];
		for (int i = 0; i < width*height; i++)
			colors[i] = Random_rgb();

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int comp = u->find(y * width + x);
				if (regionID.find(comp) == regionID.end()) {
					regionID[comp] = count;
					count ++;
				}
				imRef(output, x, y) = colors[comp];
				comID[y][x] = regionID[comp];
			}
		}  

		delete [] colors;  
		delete u;

		return output;
}

class GraphBasedImageSegmentation
{
public:

	/** 
	* GraphBasedImage Segmentation 算法
	* @param img					输入要分割的图像
	* @param outputdata		分割之后每个像素点所在的component编号
	* @param numCluster		分割之后component的个数
	* @return						返回分割之后的示例图像
	* @return	comID			图像每个像素点属于哪个component编号，编号从1开始
	*/
	static IplImage * GraphBasedImageSeg(IplImage * img, vector<vector<int> > & comID, int & numCluster, float k = 100, float min_size = 20)
	{
		/** sigma the gaussian factor */
		float sigma = (float)0.8;

		/** bigger k prefer bigger component */

		int height = img->height;
		int width = img->width;
		image<rgb> *input = new image<rgb>(width, height);
		comID.resize(height);
		for (int i = 0; i < height; ++i)
			comID[i].resize(width);

		unsigned char * data = (unsigned char *) img->imageData;
		int step = img->widthStep;

		if (img->nChannels == 3)
		{
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					rgb color;
					color.r = data[y * step + x * 3 + 2];
					color.g = data[y * step + x * 3 + 1];
					color.b = data[y * step + x * 3 + 0];
					imRef(input, x, y) = color;
				}
			}
		}
		else
		{
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					rgb color;
					color.r = data[y * step + x];
					color.g = color.r;
					color.b = color.r;
					imRef(input, x, y) = color;
				}
			}
		}

		int num_ccs; 
		long startTime = 0;
		image<rgb> *seg = segment_image(input, sigma, k, (int)min_size, &num_ccs, comID); 
		long stopTime = 0;
		//cout<<stopTime-startTime<<endl;

		IplImage * output = cvCreateImage(cvSize(width,height), 8, 3);
		unsigned char * outputdata = (unsigned char *) output->imageData;
		int outputstep = output->widthStep;

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				outputdata[y * outputstep + x * 3 + 2] =  (imRef(seg, x, y)).r;
				outputdata[y * outputstep + x * 3 + 1] =  (imRef(seg, x, y)).g;
				outputdata[y * outputstep + x * 3 + 0] =  (imRef(seg, x, y)).b;
			}
		}

		//printf("got %d components\n", num_ccs);
		numCluster = num_ccs;
		delete seg;
		delete input;
		return output;
	}
};

//////////////////////////////////////////////// END SEGMENTATION //////

class CvRegion
{
public:
	/** 区域内的点 */
	/** 区域内的点 .x 横坐标(宽度) .y 纵坐标(高度) */
	vector<CvPoint> points;

	/** 最小平行坐标轴矩形覆盖 */
	int minH; 
	int maxH;
	int minW;
	int maxW;
	int rectW;
	int rectH;

	/** 面积 */
	double area;

	/** 周长 */
	double length;

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
	double score;

	/** 构造函数 */
	CvRegion(vector<CvPoint> & ps, int height, int width, IplImage * srcImage, 
		IplImage * gray, 
		IplImage * sobel, 
		IplImage * hPlane, IplImage * sPlane, IplImage * vPlane,
		CvMat * hogG, CvMat * hogDegree, 
		IplImage * lPlane, IplImage * aPlane, IplImage * bPlane, bool calAdvancedFeature)
	{
		srcHeight = height;
		srcWidth = width;

		points = ps;
		sourceImage = srcImage;
		totalEnergy = 0;
		averGray = 0;
		torGray = 0;
		center.x = 0;
		center.y = 0;
		tag = 0;
		score = 0;

		if (points.size() == 0) return;

		// 抽取特征
		calRegionFeature(srcImage, gray, sobel, hPlane, sPlane, vPlane, hogG, hogDegree, lPlane, aPlane, bPlane);
	}

	CvRegion()
	{
		points.clear();
		totalEnergy = 0;
		averGray = 0;
		torGray = 0;
		center.x = 0;
		center.y = 0;
		tag = 0;
		score = 0;
	}

	vector<double> getAllFeature()
	{
		vector<double> ret;
		ret.insert(ret.end(), hog_hist.begin(), hog_hist.end());
		//ret.insert(ret.end(), h_hist.begin(), h_hist.end());
		ret.push_back(averRR);
		ret.push_back(averGG);
		ret.push_back(averBB);
		ret.push_back(averGray);
		ret.push_back(torGray);
		ret.push_back(averEnergy);
		ret.push_back(torEnergy);
		return ret;
	}

	/** 输出 */
	friend ostream & operator << (ostream & out, CvRegion & region)
	{
		out << "Ps = " << region.points.size() << " ";
		out << "" << region.center << " ";
		//out << "MinRect = " << "(" << region.box.center.x << 
		//	"," << region.box.center.y << ")" << "  H = " << region.box.size.height << " W = " << region.box.size.width << " ";
		out << "" << "H = " << region.rectH << "[" << region.minH << "-" << region.maxH << "]" <<  " W = " << region.rectW << "[" << region.minW << "-" << region.maxW << "]";// << " MinRect = " << 
		//region.box.size.height << " " << region.box.size.width;

		return out;
	}


	/** 计算区域内点的特征 */
	void calRegionFeature(IplImage * srcImage, 
		IplImage * gray, 
		IplImage * sobel, 
		IplImage * hPlane, IplImage * sPlane, IplImage * vPlane,
		CvMat * hogG, CvMat * hogDegree, 
		IplImage * lPlane, IplImage * aPlane, IplImage * bPlane)
	{
		int k, i, j;
		averH = 0;
		averS = 0;
		averV = 0;

		// 计算平均Lab
		averL = 0;
		averA = 0;
		averB = 0;
		if (lPlane != NULL)
		{
			unsigned char * lData = (unsigned char *) lPlane->imageData;
			unsigned char * aData = (unsigned char *) aPlane->imageData;
			unsigned char * bData = (unsigned char *) bPlane->imageData;
			int lStep = lPlane->widthStep;
			for (k = 0; k < points.size(); ++k)
			{
				i = points[k].y;
				j = points[k].x;

				averL += lData[i * lStep + j];
				averA += aData[i * lStep + j];
				averB += bData[i * lStep + j];
			}

			averL /= points.size();
			averA /= points.size();
			averB /= points.size();
		}

		// 计算平均RGB
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

		// 计算能量等特征，中心坐标
		float * sobelFloat = sobel != NULL ? (float *) sobel->imageData : NULL;
		unsigned char * grayData = gray != NULL ? (unsigned char *) gray->imageData : NULL;
		int grayStep = 0;
		if (gray != NULL) grayStep = gray->widthStep;
		for (k = 0; k < points.size(); ++k)
		{
			i = points[k].y;
			j = points[k].x;
			if (sobel != NULL) totalEnergy += sobelFloat[i * sobel->width + j];
			if (gray != NULL) averGray += grayData[i * grayStep + j];
			center.x += j;
			center.y += i;
		}

		// 计算平均灰度
		averGray /= points.size();
		center.x /= points.size();
		center.y /= points.size();

		/** 区域内的平均能量 */
		averEnergy = totalEnergy / points.size();
		torEnergy = 0;

		if (sobel != NULL)
		{
			for (k = 0; k < points.size(); ++k)
			{
				i = points[k].y;
				j = points[k].x;
				torEnergy += sqr(sobelFloat[i * sobel->width + j] - averEnergy);
			}
		}
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

		/** 计算区域内的平行坐标轴矩形覆盖 */
		minH = 9999;
		maxH = -1;
		minW = 9999;
		maxW = -1;
		for (k = 0; k < points.size(); ++k)
		{
			int nowW = points[k].x;
			int nowH = points[k].y;
			minH = min(minH, nowH);
			maxH = max(maxH, nowH);
			minW = min(minW, nowW);
			maxW = max(maxW, nowW);
		}
		rectW = maxW - minW + 1;
		rectH = maxH - minH + 1;
		area = rectW * rectH;

		/** 计算HOG */
	}

	//////////////////////////////////////////////////////////////
	// 绘图函数
	//////////////////////////////////////////////////////////////

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
		IplImage * xSobel = NULL;
		IplImage * ySobel = NULL;
		IplImage * sobel = NULL;
		IplImage * hsv_img = NULL;
		IplImage * h_plane = NULL;
		IplImage * s_plane = NULL;
		IplImage * v_plane = NULL;
		IplImage * lab_img = NULL;
		IplImage * l_plane = NULL;
		IplImage * a_plane = NULL;
		IplImage * b_plane = NULL;

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
		vector<vector<CvPoint> > points(coms);
		vector<CvRegion> ret;

		for (i = 0; i < height; ++i)
			for (j = 0; j < width; ++j)
			{
				int id = comID[i][j] - 1;
				points[id].push_back(cvPoint(j, i));
			}

			// 构造每个区域
			// 是否计算凸包等高级特征
			bool calAdvanced = (features.find("geometry") != features.end());
			for (i = 0; i < coms; ++i)
				if (points[i].size() < 100000 && points[i].size() > 0) // 避免对大物体的无意义抽取，谨慎优化
					ret.push_back(CvRegion(points[i], height, width, img, gray, sobel, h_plane, s_plane, v_plane, gradEnergy, theda, l_plane, a_plane, b_plane, calAdvanced));

			if (gradEnergy != NULL) cvReleaseMat(&gradEnergy);
			if (theda != NULL) cvReleaseMat(&theda);
			if (gray != NULL) cvReleaseImage(&gray);
			if (xSobel != NULL) cvReleaseImage(&xSobel);
			if (ySobel != NULL) cvReleaseImage(&ySobel);
			if (sobel != NULL) cvReleaseImage(&sobel);
			if (hsv_img != NULL) cvReleaseImage(&hsv_img);
			if (h_plane != NULL) cvReleaseImage(&h_plane);
			if (s_plane != NULL) cvReleaseImage(&s_plane);
			if (v_plane != NULL) cvReleaseImage(&v_plane);
			if (lab_img != NULL) cvReleaseImage(&lab_img);
			if (l_plane != NULL) cvReleaseImage(&l_plane);
			if (a_plane != NULL) cvReleaseImage(&a_plane);
			if (b_plane != NULL) cvReleaseImage(&b_plane);

			return ret;
	}
};

////////////////////////////////////////////////////////////////////////////////////////////
// FLOOD FILL 
// 根据输入的单通道二值化图像，获得类似Segmentation的 comID二维数组，
// 即每个像素点属于哪个连通分量 
// 对于每个非零元素进行FLOOD_FILL，comID从0开始编号，获得CvRegion表示
////////////////////////////////////////////////////////////////////////////////////////////
int dir[8][2] = { {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {-1, -1}, {1, 1}, {-1, 1}, {1, -1} };
void floodFill(IplImage * img, vector<vector<int> > & comID, vector<CvRegion> & regions, int dirs, int maxSize)
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
			if (comID[i][j] == -1)
			{
				if (data[i * step + j] != 0)
				{
					comID[i][j] = coms;
					vector<CvPoint> points;
					queue<pair<int, int> > q;
					q.push(make_pair(i, j));

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
							}
						}
					}

					if (points.size() < maxSize && points.size() > 0) // 只抽取适度大小的物体
						regions.push_back(CvRegion(points, height, width, img, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, false));
					else
						regions.push_back(CvRegion());
					coms++;
				}
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
//////////////////////////////////////////////////////////////////////
int floodFill(IplImage * img, vector<vector<int> > & comID, int dirs)
{
	int i, j;
	int height = img->height;
	int width = img->width;
	comID = vector<vector<int> >(height, vector<int>(width, -1));

	int coms = 0;
	unsigned char * data = (unsigned char *)img->imageData;
	int step = img->widthStep;
	for (i = height - 1; i >= 0; --i)
		for (j = 0; j < width; ++j)
			if (comID[i][j] == -1)
			{
				int value = data[i * step + j];
				coms++;
				comID[i][j] = coms;
				queue<pair<int, int> > q;
				q.push(make_pair(i, j));

				while (q.size() > 0)
				{
					pair<int, int> top = q.front();
					q.pop();
					int x = top.first;
					int y = top.second;

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
			}
			return coms;
}

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


//////////////////// MAIN CODE ///////////////////////////

/** Union Find Set 
 * O(1) to merge two set
 * O(1) to find which set the given element belong to 
 */
class UnionFindSet
{
    private:
        vector<int> belongTo;

        /** How many different sets exist now */
        int setCount;

    public:
        UnionFindSet(int size)
        {
            belongTo = vector<int>(size);
            clear();
        }

        /** Reset the union find set data */
        void clear()
        {
            for (int i = 0; i < belongTo.size(); ++i) 
                belongTo[i] = -1;
            setCount = belongTo.size();
        }

        /** Return element i's set's representive element */
        int find(int elementID)
        {
            if (belongTo[elementID] < 0)
                return elementID;
            else 
                return belongTo[elementID] = find(belongTo[elementID]);
        }

        /** Return the element i's set's size */
        int getSize(int elementID)
        { 
            int representiveElement = find(elementID);
            return -belongTo[representiveElement];
        }

        /** merge two element */
        void Union(int elementID1, int elementID2)
        {
            int representiveElement1 = find(elementID1);
            int representiveElement2 = find(elementID2);
            if (representiveElement1 == representiveElement2) 
                return;

            setCount--;
            int size1 = -belongTo[representiveElement1];
            int size2 = -belongTo[representiveElement2];

            if (size1 <= size2) {
                belongTo[representiveElement1] = representiveElement2;
                belongTo[representiveElement2] -= size1;
            }
            else {
                belongTo[representiveElement2] = representiveElement1;
                belongTo[representiveElement1] -= size2;
            }
        }

        /** Return different set count */
        inline int getSetCount() 
        {
            return setCount;
        }

        void output(ostream & out)
        {
            out << setCount << endl;
            for (int i = 0; i < belongTo.size(); ++i)
                out << find(i) << "\t";
            out << endl;
        }

        vector<vector<int> > getAllSets() 
        {
            map<int, vector<int> > sets;
            for (int i = 0; i < belongTo.size(); ++i) 
                sets[find(i)].push_back(i);

            vector<vector<int> > ret;
            for (map<int, vector<int> >::iterator itr = sets.begin(); itr != sets.end(); ++itr) 
                ret.push_back(vector<int>(itr->second.begin(), itr->second.end()));
            return ret;
        }
};

map<string, vector<CvRect> > ans;

class CraterDetection
{
public:
	IplImage * srcImage;
	IplImage * grayImage;
	IplImage * binaryImage;
	vector<pair<double, pair<string, CvRect> > > carters;
	string nowfilename;
	vector<pair<double, CvRect> > nowrects;
	set<string> cals;

    bool addRect(CvRect & rect, double score)
    {
		//if (score < 15) return false;
		/**
        for (int i = 0; i < nowrects.size(); ++i)
            if (getRectIntersect(rect, nowrects[i].second) > rect.height * rect.width / 3)
            {
                nowrects[i].first = max(nowrects[i].first, score);
                return false;
            }
			*/
        nowrects.push_back(make_pair(score, rect));
        return true;
    }

	void clearRect(){
		nowrects.clear();
	}

	int useful()
    {
		int l, r, t;

		l = r = 0;
        for (int i = 0; i < nowrects.size(); ++i)
			if (nowrects[i].second.x < 716)
				l++;
			else
				r++;
		t = l + r;
		//debug2(l, r);
		if (l < t / 5 || r < t / 5 || t < 50){
			clearRect();
			return 0;
		}
        return 1;
    }

	///// 处理完一个文件后进行最终的善后操作
	void finishnowfile()
	{
		// 计算矩形相交构成的等价关系
		/**
		int N = nowrects.size();
		if (N <= 3000)
		{
			UnionFindSet ufset(N);
			for (int i = 0; i < N; ++i)
				for (int j = i + 1; j < N; ++j)
					if (getRectIntersect(nowrects[i].second, nowrects[j].second) > 0)
						ufset.Union(i, j);

			for (int i = 0; i < N; ++i)
			{
				int idsize = ufset.getSize(i);
				if (idsize >= 2)
				nowrects[i].first -= (idsize - 1) * 10;
			}
		}
		*/

		// 将当前识别出的陨石坑加入最终答案
		for (int i = 0; i < nowrects.size(); ++i)
		{
			CvRect & rect = nowrects[i].second;
			if (rect.x < 0) rect.x = 0;
			if (rect.y < 0) rect.y = 0;
			if (rect.x + rect.width - 1 >= srcImage->width - 1) rect.width = srcImage->width - rect.x;
			if (rect.y + rect.height - 1 >= srcImage->height - 1) rect.height = srcImage->height - rect.y;
			carters.push_back(make_pair(nowrects[i].first, make_pair(nowfilename, nowrects[i].second)));

#ifdef OPENCV
			bool ok = false;
			vector<CvRect> & ansrects = ans[nowfilename];
			int area1 = nowrects[i].second.height * nowrects[i].second.width;
			for (int j = 0; j < ansrects.size(); ++j)
			{
				int area2 = ansrects[j].height * ansrects[j].width;
				if (getRectIntersect(nowrects[i].second, ansrects[j]) > (area1 + area2) * 3 / 10)
					ok = true;
			}

			CvFont font = cvFont(1, 1);
			char temp[20];
			sprintf(temp, "%d", (int)nowrects[i].first);
			if (srcImage->height < 1000)
				cvPutText(srcImage, temp, cvPoint(nowrects[i].second.x, nowrects[i].second.y + 15), &font, ok ? CV_RGB(255, 0, 0) : CV_RGB(255, 255, 0));
			else
			{
				CvFont fontb = cvFont(2, 2);
				cvPutText(srcImage, temp, cvPoint(nowrects[i].second.x, nowrects[i].second.y + 15), &fontb, ok ? CV_RGB(255, 0, 0) : CV_RGB(255, 255, 0));
			}
			cvRectangle(srcImage, nowrects[i].second, CV_RGB(0, 255, 0), 2);
#endif
		}

		// 绘制调试信息
		#ifdef OPENCV
			IplImage * resizesrc, * resizebinary, * resizegray;
			if (srcImage->height > 1000)
			{
				resizesrc = cvCreateImage(cvSize(600, 600), 8, 3);
				resizebinary = cvCreateImage(cvSize(600, 600), 8, 1);
				resizegray = cvCreateImage(cvSize(600, 600), 8, 1);
			}
			else
			{
				resizesrc = cvCreateImage(cvSize(600, 400), 8, 3);
				resizebinary = cvCreateImage(cvSize(600, 400), 8, 1);
				resizegray = cvCreateImage(cvSize(600, 400), 8, 1);
			}
			cvResize(srcImage, resizesrc);
			cvResize(binaryImage, resizebinary);
			cvResize(grayImage, resizegray);
			cvShowImage("debug", resizesrc);
			cvShowImage("binary", resizebinary);
			cvShowImage("gray", resizegray);
			cvReleaseImage(&resizesrc);
			cvReleaseImage(&resizebinary);
			cvReleaseImage(&resizegray);
		#endif
	}

	CraterDetection()
	{
		srcImage = NULL;
		grayImage = NULL;
		binaryImage = NULL;
        cals.insert("gray");
        cals.insert("rgb");
	}

    int processLargeImage(bool mustBlack = false, bool mustWhite = false, int scale = 12, bool dirright = true)
    {
        vector<CvRegion> bregions;
        vector<vector<int> > comID;
        int coms = floodFill(binaryImage, comID, 4);
        bregions = CvRegion::getRegionFromSegment(srcImage, comID, coms, cals);

        for (int i = 0; i < bregions.size(); ++i)
        {
            int whmax = max(bregions[i].rectH, bregions[i].rectW);
            int whmin = min(bregions[i].rectH, bregions[i].rectW);
            double arearatio = (double) bregions[i].points.size() / bregions[i].rectH / bregions[i].rectW;
            double lenratio = (double) bregions[i].rectH / bregions[i].rectW;
            if (bregions[i].rectH > 4 * bregions[i].rectW) continue;
            if (bregions[i].rectW > 4 * bregions[i].rectH) continue;
            if (bregions[i].rectH < 8 || bregions[i].rectW < 8) continue;
            //if (arearatio < 0.4) continue;// && !(lenratio >= 0.8 && lenratio <= 1.2 && bregions[i].rectH > 100)) continue;
            //if (arearatio < 0.3) continue;

            double avergray = bregions[i].averGray;
            if (mustBlack && avergray > 60) continue;
			if (mustWhite && avergray < 100) continue; 
            double score = arearatio * 100;
            if (bregions[i].rectH > 4 * bregions[i].rectW) score /= 2;
            if (bregions[i].rectW > 4 * bregions[i].rectH) score /= 2;
            //cout << bregions[i] << " " << avergray << " " << arearatio << " " << score << endl;
			CvRect rect;
            if (dirright)
				rect = cvRect(bregions[i].minW - 5, bregions[i].minH - 5, bregions[i].rectH * scale / 10, bregions[i].rectH * scale / 10);
            else
				rect = cvRect(bregions[i].maxW - bregions[i].rectH, bregions[i].minH, bregions[i].rectH * scale / 10, bregions[i].rectH * scale / 10);

			score /= 3;
            if (addRect(rect, score))
                cvRectangle(srcImage, rect, CV_RGB(0, 255, 0), 2);
        }
		return 1;
    }

    void processSmallImage()
    {
		/**
        vector<vector<int> > feature;
        IplImage * image = GraphBasedImageSegmentation::GraphBasedImageSeg(srcImage, feature, coms, 350, 20);
		cvShowImage("seg", image);
        cvReleaseImage(&image);
        vector<CvRegion> regions = CvRegion::getRegionFromSegment(srcImage, feature, coms, cals);

        for (int i = 0; i < regions.size(); ++i)
        {
			CvRect rect = cvRect(regions[i].minW, regions[i].minH, regions[i].rectW * 12 / 10, regions[i].rectH * 12 / 10);
			if (rect.height < 27) rect.height = 27;
			//cvRectangle(srcImage, rect, CV_RGB(255, 255, 255), 2);

            int whmax = max(regions[i].rectH, regions[i].rectW);
            int whmin = min(regions[i].rectH, regions[i].rectW);
            double arearatio = (double) regions[i].points.size() / regions[i].rectH / regions[i].rectW;
            if (regions[i].rectH > 5 * regions[i].rectW) continue;
            if (regions[i].rectW > 5 * regions[i].rectH) continue;
			if (regions[i].rectH < 5 || regions[i].rectW < 5) continue;
			if (regions[i].averGray > 60) continue;
            if (arearatio < 0.5) continue;
            if (whmax < 20 || whmin > 200) continue;
			if (whmax < 25 && regions[i].averGray > 40) continue;
            double score = arearatio * (100 - regions[i].averGray);
            if (regions[i].rectH > 3 * regions[i].rectW) score /= 2;
            if (regions[i].rectW > 3 * regions[i].rectH) score /= 2;
			if (regions[i].averGray < 8) score /= 2;
			if (regions[i].averGray < 8 && regions[i].torGray < 3) continue;
			
			//cout << regions[i] << " " << regions[i].averGray << " " << regions[i].torGray << " " << arearatio << " " << score << endl;
            if (score < 25) continue;
            cvRectangle(srcImage, rect, CV_RGB(255, 0, 0), 2);
            addRect(rect, score);
        }
		*/
		
        vector<CvRegion> bregions;
        vector<vector<int> > comID;
        int coms = floodFill(binaryImage, comID, 4);
        bregions = CvRegion::getRegionFromSegment(srcImage, comID, coms, cals);

		int lastsize = nowrects.size();
        //cout << "--------" << endl;
        for (int i = 0; i < bregions.size(); ++i)
        {
			CvRect rect = cvRect(bregions[i].minW - 5, bregions[i].minH - 5, bregions[i].rectH * 13 / 10, bregions[i].rectH * 13 / 10);

            int whmax = max(bregions[i].rectH, bregions[i].rectW);
            int whmin = min(bregions[i].rectH, bregions[i].rectW);
            double arearatio = (double) bregions[i].points.size() / bregions[i].rectH / bregions[i].rectW;
            if (bregions[i].rectH > 4 * bregions[i].rectW) continue;
            if (bregions[i].rectW > 4 * bregions[i].rectH) continue;
            if (bregions[i].rectH < 5 || bregions[i].rectW < 5) continue;
            if (arearatio < 0.3) continue;
            double avergray = bregions[i].averGray;
            if (avergray > 60) continue;
            double score = arearatio * (100 - avergray);
            if (bregions[i].rectH > 2 * bregions[i].rectW) score /= 2;
            if (bregions[i].rectW > 2 * bregions[i].rectH) score /= 2;
			if (bregions[i].points.size() > 20000) continue;
			if (bregions[i].points.size() > 1000) score *= 2;
			if (whmax <= 16 || whmin > 200) continue;
			if (bregions[i].rectH > 200) continue;
            //cout << bregions[i] << " " << avergray << " " << arearatio << " " << score << endl;
            
			if (score < 40) continue;
            if (addRect(rect, score))
				cvRectangle(srcImage, rect, CV_RGB(0, 255, 0), 2);
        }        
    }

	void processLargeImage1()
    {
        vector<CvRegion> bregions;
        vector<vector<int> > comID;
        int coms = floodFill(binaryImage, comID, 4);
        bregions = CvRegion::getRegionFromSegment(srcImage, comID, coms, cals);

        //cout << "--------" << endl;
        for (int i = 0; i < bregions.size(); ++i)
        {
            int whmax = max(bregions[i].rectH, bregions[i].rectW);
            int whmin = min(bregions[i].rectH, bregions[i].rectW);
            double arearatio = (double) bregions[i].points.size() / bregions[i].rectH / bregions[i].rectW;
            double lenratio = (double) bregions[i].rectH / bregions[i].rectW;
            if (bregions[i].rectH > 3 * bregions[i].rectW) continue;
            if (bregions[i].rectW > 3 * bregions[i].rectH) continue;
            if (bregions[i].rectH < 8 || bregions[i].rectW < 8) continue;
            if (arearatio < 0.5) continue;// && !(lenratio >= 0.8 && lenratio <= 1.2 && bregions[i].rectH > 100)) continue;
            if (arearatio < 0.3) continue;

            double avergray = bregions[i].averGray;
            if (avergray > 60) continue;
            double score = arearatio * (100 - avergray);
            if (bregions[i].rectH > 2 * bregions[i].rectW) score /= 2;
            if (bregions[i].rectW > 2 * bregions[i].rectH) score /= 2;
            //cout << bregions[i] << " " << avergray << " " << arearatio << " " << score << endl;
            CvRect rect = cvRect(bregions[i].minW, bregions[i].minH, bregions[i].rectH, bregions[i].rectH);
        
            if (addRect(rect, score))
                cvRectangle(srcImage, rect, CV_RGB(0, 255, 0), 2);
        }
    }

    void processLargeImage2()
    {
        vector<CvRegion> bregions;
        vector<vector<int> > comID;
        int coms = floodFill(binaryImage, comID, 4);
        bregions = CvRegion::getRegionFromSegment(srcImage, comID, coms, cals);

        //cout << "--------" << endl;
        for (int i = 0; i < bregions.size(); ++i)
        {
            int whmax = max(bregions[i].rectH, bregions[i].rectW);
            int whmin = min(bregions[i].rectH, bregions[i].rectW);
            double arearatio = (double) bregions[i].points.size() / bregions[i].rectH / bregions[i].rectW;
            double lenratio = (double) bregions[i].rectH / bregions[i].rectW;
            if (bregions[i].rectH > 2 * bregions[i].rectW) continue;
            if (bregions[i].rectW > 2 * bregions[i].rectH) continue;
            if (bregions[i].rectH < 20 || bregions[i].rectW < 20) continue;
            //if (arearatio < 0.5) continue;// && !(lenratio >= 0.8 && lenratio <= 1.2 && bregions[i].rectH > 100)) continue;
            if (arearatio < 0.2) continue;

            double avergray = bregions[i].averGray;
            if (avergray < 200) continue;
            double score = arearatio * (40);
            if (bregions[i].rectH > 2 * bregions[i].rectW) score /= 2;
            if (bregions[i].rectW > 2 * bregions[i].rectH) score /= 2;
            //cout << bregions[i] << " " << avergray << " " << arearatio << " " << score << endl;
            CvRect rect = cvRect(bregions[i].minW, bregions[i].minH, bregions[i].rectH, bregions[i].rectH);
       

            if (addRect(rect, score))
                cvRectangle(srcImage, rect, CV_RGB(0, 0, 255), 2);
        }
    }

	int basicFilter(vector<CvRegion> & regions, bool mustDark, double minratio = 0.3, int maxGray = 255, bool couldsmall = false)
	{
		int ret = 0;
        for (int i = 0; i < regions.size(); ++i)
        {
			int H = regions[i].rectH;
			int W = regions[i].rectW;
            int whmax = max(H, W);
            int whmin = min(H, W);
            double arearatio = (double) regions[i].points.size() / H / W;
            double lenratio = (double) H / W;
			if (lenratio > 1) lenratio = 1 / lenratio;
			double avergray = regions[i].averGray;

			if (mustDark && avergray > 160) continue;
			if (avergray > maxGray) continue;
            if (H > 3 * W) continue;
            if (W > 2 * H) continue;
            if (H < 15 || W < 15 && !couldsmall) continue;
            if (arearatio < minratio) continue;
			regions[i].tag = 1;
			regions[i].score = 100 * arearatio * lenratio;
			ret++;
        }
		return ret;
	}

	int intersectFilter(vector<CvRegion> & regions)
	{
		vector<int> closecnt(regions.size(), 0);
		for (int i = 0; i < regions.size(); ++i)
			if (regions[i].tag == 1)
			{
				int H = regions[i].rectH;
				int W = regions[i].rectW;
				double arearatio = (double) regions[i].points.size() / H / W;
				double lenratio = (double) H / W;
				if (lenratio > 1) lenratio = 1 / lenratio;
				if (arearatio > 0.6 && H > 30) continue;
				if (lenratio > 0.8 && lenratio < 1.2 && H > 30) continue;
				for (int j = 0; j < regions.size(); ++j)
					if (regions[j].tag == 1 && j != i)
					{
						double dis = dist(regions[i].center, regions[j].center);
						double r = max(regions[i].rectH, regions[i].rectW);
						if (dis <= r * 1.5 && r < 100)
						{
							closecnt[i]++;
							regions[i].score -= 30;
						}
					}
			}
		return 0;
	}

	void addRegionToRect(vector<CvRegion> & regions, int scale = 12)
	{
		for (int i = 0; i < regions.size(); ++i)
			if (regions[i].tag == 1 && regions[i].score > 15)
			{
				CvRect rect = cvRect(regions[i].minW, regions[i].minH, regions[i].rectH * scale / 10, regions[i].rectH * scale / 10);
				addRect(rect, regions[i].score);
			}
	}

	bool processLargeImageByBlackHole()
	{
		//cout << "process b" << endl;
		int cntblack = threshold(grayImage, binaryImage, 25, 255, 1);

		vector<CvRegion> regions;
        vector<vector<int> > comID;
        int coms = floodFill(binaryImage, comID, 4);
        regions = CvRegion::getRegionFromSegment(srcImage, comID, coms, cals);

		int validblack = basicFilter(regions, true);
		//debug2(cntblack, validblack);

		
		intersectFilter(regions);
		addRegionToRect(regions);
		if (validblack < 50) return false;
		return true;
	}

	bool processLargeImageByBlackHole2()
	{
		//cout << "process b2" << endl;
		int cntblack = threshold(grayImage, binaryImage, 50, 255, 1);

		vector<CvRegion> regions;
        vector<vector<int> > comID;
        int coms = floodFill(binaryImage, comID, 4);
        regions = CvRegion::getRegionFromSegment(srcImage, comID, coms, cals);

		int validblack = basicFilter(regions, true, 0.3, 60);
		//debug2(cntblack, validblack);

		intersectFilter(regions);
		addRegionToRect(regions, 14);
		if (validblack < 50) return false;
		return true;
	}

	/**
	bool processLargeImageBySobel()
	{
		//cout << "sobel" << endl;
		IplImage * xSobel = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_32F, 1);
		IplImage * ySobel = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_32F, 1);
		cvSobel(grayImage, xSobel, 1, 0, 3);
		cvSobel(grayImage, ySobel, 0, 1, 3);

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

		cvThreshold(sobel, binaryImage, 110, 255, CV_THRESH_BINARY);
		cvReleaseImage(&xSobel);
		cvReleaseImage(&ySobel);
		cvReleaseImage(&sobel);
			
		vector<CvRegion> regions;
        vector<vector<int> > comID;
        int coms = floodFill(binaryImage, comID, 8);
        regions = CvRegion::getRegionFromSegment(srcImage, comID, coms, cals);

		basicFilter(regions, true);
		intersectFilter(regions);
		addRegionToRect(regions, 12);
		return true;
	}
	*/

	bool processLargeImageByTianXiang()
	{
		//cout << "process tianxiang" << endl;
		getBinaryImage(grayImage, 11, 5, binaryImage);
		vector<CvRegion> regions;
        vector<vector<int> > comID;
        int coms = floodFill(binaryImage, comID, 4);
        regions = CvRegion::getRegionFromSegment(srcImage, comID, coms, cals);

		basicFilter(regions, true, 0.4, 60, nowrects.size() < 50);
		intersectFilter(regions);
		addRegionToRect(regions, 10);
		return true;
	}

	bool processLargeImageByBright()
	{
		//cout << "bright" << endl;
		getBinaryImage(grayImage, 41, -10, binaryImage, false);

		vector<CvRegion> regions;
        vector<vector<int> > comID;
        int coms = floodFill(binaryImage, comID, 4);
        regions = CvRegion::getRegionFromSegment(srcImage, comID, coms, cals);

		basicFilter(regions, false, 0.6);
		intersectFilter(regions);
		addRegionToRect(regions, 12);
		return true;
	}

	bool processLargeImageByBright2()
	{
		//cout << "bright2" << endl;
		//int cntwhite = threshold(grayImage, binaryImage, 200, 255, 0);
		getBinaryImage(grayImage, 201, 60, binaryImage);

		vector<CvRegion> regions;
        vector<vector<int> > comID;
        int coms = floodFill(binaryImage, comID, 4);
        regions = CvRegion::getRegionFromSegment(srcImage, comID, coms, cals);

		basicFilter(regions, false, 0.4);
		intersectFilter(regions);
		addRegionToRect(regions, 12);
		return true;
	}

	int processImage(string name, int W, int H, vector <int> data)
	{
		nowfilename = name;
		nowrects.clear();
		if (srcImage != NULL) cvReleaseImage(&srcImage);
		if (grayImage != NULL) cvReleaseImage(&grayImage);
		if (binaryImage != NULL) cvReleaseImage(&binaryImage);
		srcImage = cvCreateImage(cvSize(W, H), 8, 3);
	
		unsigned char * srcdata = (unsigned char *) srcImage->imageData;
		int step = srcImage->widthStep;
		for (int i = 0; i < H; ++i)
			for (int j = 0; j < W; ++j)
				srcdata[i * step + j * 3 + 0] = srcdata[i * step + j * 3 + 1] = srcdata[i * step + j * 3 + 2] = data[i + j * H];

		grayImage = cvCreateImage(cvSize(W, H), 8, 1);
		binaryImage = cvCreateImage(cvSize(W, H), 8, 1);
		cvCvtColor(srcImage, grayImage, CV_BGR2GRAY);

		if (srcImage->height < 1000)
		{
			int cnt = threshold(grayImage, binaryImage, 25, 255, 1);
			if (cnt > srcImage->height * srcImage->width / 6) 
				cnt = threshold(grayImage, binaryImage, 10, 255, 1);
			processSmallImage();
		}
		else
		{
			int cntwhite = threshold(grayImage, binaryImage, 200, 255, 0);
			int cntblack = threshold(grayImage, binaryImage, 25, 255, 1);
			//debug2(cntwhite, cntblack);
			if (!processLargeImageByBlackHole()) 
			{
				processLargeImageByBlackHole2();
				if (cntblack > 50000 && cntwhite < 300000) processLargeImageByTianXiang();
				//processLargeImageBySobel();
				//debug2(nowrects.size(), cntwhite);
				if (nowrects.size() <= 30 && cntwhite > 300000 || nowrects.size() <= 20 && cntwhite > 100000)
				{
					nowrects.clear();
					processLargeImageByBright();
				}

				// 如果数据偏移过大
				int l = 0;
				int r = 0;
				int t = 0; int b = 0;
				for (int i = 0; i < nowrects.size(); ++i)
				{
					if (nowrects[i].second.x < srcImage->width / 2) l++;
					else r++;
					if (nowrects[i].second.y < srcImage->height / 2) t++;
					else b++;
				}
				//debug4(l, r, t, b);
				if (l > nowrects.size() * 0.9 || r > nowrects.size() * 0.9)
					nowrects.clear();
				if (t > nowrects.size() * 0.8 || b > nowrects.size() * 0.8) 
					nowrects.clear();

				if (nowrects.size() == 0)
					processLargeImageByBright2();
			}

			//for (int i = 0; i < nowrects.size(); ++i)
			//	if (nowrects[i].first > 30) nowrects[i].first = 30;

			//getBinaryImage(grayImage, 11, 5, binaryImage);
			
			/**
			getBinaryImage(grayImage, 201, 60, binaryImage);
			
			
            if (nowrects.size() <= 10)
            {
                //cout << "mie" << endl;
                grayContrastEnhance(grayImage);
                //cvThreshold(grayImage, binaryImage, 200, 255, CV_THRESH_BINARY);
                getBinaryImage(grayImage, 21, -10, binaryImage, false);
                processLargeImage2();
            }
			*/

			/**
			IplImage * xSobel = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_32F, 1);
			IplImage * ySobel = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_32F, 1);
			cvSobel(grayImage, xSobel, 1, 0, 3);
			cvSobel(grayImage, ySobel, 0, 1, 3);

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
			cvThreshold(sobel, binaryImage, 100, 255, CV_THRESH_BINARY);
			processLargeImage(false, false, 12, true);
			
			cvReleaseImage(&xSobel);
			cvReleaseImage(&ySobel);
			cvReleaseImage(&sobel);
			*/

			
            //getBinaryImage(grayImage, 201, 60, binaryImage);
			/**
			int cntwhite = threshold(grayImage, binaryImage, 200, 255, 0);
			int cntblack = threshold(grayImage, binaryImage, 25, 255, 1);
			//debug2(cntwhite, cntblack);
			if (cntblack < 100000)
			{
				//cout << "tianxiang" << endl;
				getBinaryImage(grayImage, 11, 5, binaryImage);
				processLargeImage(true, false, 12, false);
				if (!useful())
				{
					//cout << "!useful" << endl;
					getBinaryImage(grayImage, 201, 60, binaryImage);
					processLargeImage(false, false, 14, true);
				}
			}
			else
			{
				//cout << "blackthres" << endl;
				processLargeImage(false, false, 12, true);
			}
			
            if (nowrects.size() <= 15)
            {
				//cout << "less" << endl;
                //grayContrastEnhance(grayImage);
                //cvThreshold(grayImage, binaryImage, 210, 255, CV_THRESH_BINARY);
                getBinaryImage(grayImage, 51, -20, binaryImage, false);
                processLargeImage(false, true, 12, true);
            }
			*/
			//

        }
		finishnowfile();

		return 0;
	}

	vector <string> getCraters()
	{
		sort(carters.begin(), carters.end(), greater<pair<double, pair<string, CvRect> > >());
		vector<string> ans;
		for (vector<pair<double, pair<string, CvRect> > >::iterator itr = carters.begin(); itr != carters.end(); ++itr)
		{
			pair<string, CvRect> now = itr->second;
			char temp[200];
			sprintf(temp, "%s %d %d %d %d", now.first.data(), now.second.x, now.second.y, now.second.x + now.second.width - 1, 
				now.second.y + now.second.height - 1);
			ans.push_back(string(temp));
			//cout << temp << endl;
		}
		return ans;
	}

	int init()
	{
		if (srcImage != NULL) cvReleaseImage(&srcImage);
		if (grayImage != NULL) cvReleaseImage(&grayImage);
		if (binaryImage != NULL) cvReleaseImage(&binaryImage);
		srcImage = NULL;
		grayImage = NULL;
		binaryImage = NULL;
		carters.clear();
		return 0;
	}
};


#ifdef OPENCV

void loadAns(const string & path)
{
	ifstream fin(path.data());
	string filename; int K;
	string t;
	while (fin >> filename >> t >> K >> t)
	{
		//cout << filename << " " << K << endl;
		vector<CvRect> rects;
		for (int i = 0; i < K; ++i)
		{
			CvRect rect;
			fin >> rect.x >> t >> rect.y >> t >> rect.width >> t >> rect.height >> t;
			rect.width = rect.width - rect.x + 1;
			rect.height = rect.height - rect.y + 1;
			rects.push_back(rect);
		}
		ans[filename] = rects;
	}
	fin.close();
}

int autotest()
{
	CraterDetection detect;
	detect.init();
	cout << 0 << endl;

	int cmd;
	while (cin >> cmd)
	{
		if (cmd == 1)
		{
			string name; int W, H; int len;
			cin >> name >> W >> H >> len;
			vector<int> data(len);
			for (int i = 0; i < len; ++i) cin >> data[i];
			cout << detect.processImage(name, W, H, data) << endl;
		}
		else
		{
			vector<string> ans = detect.getCraters();
			cout << ans.size() << endl;
			for (int i = 0; i < ans.size(); ++i)
				cout << ans[i] << endl;
			break;
		}
	}
	return 0;
}

int main(int argc, char ** argv)
{
	if (argc == 1)
	{
		autotest();
		return 0;
	}

	loadAns("E:\\Coder\\marathon\\crater\\training\\A15\\GTF.lms");
	loadAns("E:\\Coder\\marathon\\crater\\training\\LRO\\GTF.lms");

	cvNamedWindow("ans");
	cvNamedWindow("debug");
	cvNamedWindow("binary");
	cvNamedWindow("gray");
	cvNamedWindow("seg");
	cvMoveWindow("ans", 0, 0);
	cvMoveWindow("debug", 600, 0);
	cvMoveWindow("binary", 600, 600);
	cvMoveWindow("gray", 0, 600);
	cvMoveWindow("seg", 1000, 600);
	//FileFinder finder("E:\\Coder\\marathon\\crater\\training\\LRO\\");
	FileFinder finder("E:\\Coder\\marathon\\crater\\training\\A15\\");
	//for (int i = 0; i < 10; ++i) finder.next();
	while (finder.hasNext())
	{
		string file = finder.next();
		if (file.find("jpg") == string::npos && file.find("JPG") == string::npos) continue;
		cout << file << endl;

		IplImage * img = cvLoadImage(file.data());
		vector<CvRect> & rects = ans[Util::getFileName(file)];
		for (int i = 0; i < rects.size(); ++i)
			cvRectangle(img, rects[i], CV_RGB(255, 0, 0), 2);
		IplImage * resizeimg = cvCreateImage(cvSize(600, img->height > 1000 ? 600 : 400), 8, 3);
		cvResize(img, resizeimg);
		cvShowImage("ans", resizeimg);
		string fileans = Util::getFilePath(file) + "\\" + "anal" + "\\" + Util::getFileTrueName(file) + "_ans.jpg";
		cvSaveImage(fileans.data(), img);
		cvReleaseImage(&img);
		cvReleaseImage(&resizeimg);
		img = cvLoadImage(file.data());

		// 调用识别接口
		CraterDetection detect;
		detect.init();
		vector<int> data;
		for (int x = 0; x < img->width; ++x)
			for (int y = 0; y < img->height; ++y)
				data.push_back(cvGet2D(img, y, x).val[0]);
		detect.processImage(Util::getFileName(file), img->width, img->height, data);

		cvWaitKey(0);
	}

	return 0;
}
#endif
