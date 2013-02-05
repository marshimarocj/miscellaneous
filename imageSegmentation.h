#ifndef SEGMENT
#define SEGMENT

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
#include "stdafx.h"

using namespace std;
#define debug1(x) cout << #x" = " << x << endl;
#define debug2(x, y) cout << #x" = " << x << " " << #y" = " << y << endl;
#define debug3(x, y, z) cout << #x" = " << x << " " << #y" = " << y << " " << #z" = " << z << endl;
#define debug4(x, y, z, w) cout << #x" = " << x << " " << #y" = " << y << " " << #z" = " << z << " " << #w" = " << w << endl;

typedef unsigned char uchar;

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

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				rgb color;
				color.r = data[y * step + x * 3 + 2];
				color.g = data[y * step + x * 3 + 1];
				color.b = data[y * step + x * 3 + 0];
				imRef(input, x, y) = color;
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
				outputdata[y * step + x * 3 + 2] =  (imRef(seg, x, y)).r;
				outputdata[y * step + x * 3 + 1] =  (imRef(seg, x, y)).g;
				outputdata[y * step + x * 3 + 0] =  (imRef(seg, x, y)).b;
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

#endif


