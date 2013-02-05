#ifndef PIG_GEOMETRY_H
#define PIG_GEOMETRY_H

#include "stdafx.h"

using namespace std;

const double eps = 1e-7;
const double pi = acos(-1.0);
const double INF = 1e20;

/** Return a float number's sign */
inline int sign(double d) {
	return d < -eps ? -1 : d > eps;
}

inline double sqr(double x) {
	return x * x;
}

struct Point {
	double x, y;
	Point() { x = 0.0; y = 0.0; }
	Point(double nx, double ny) : x(nx), y(ny) {}
	Point turnLeft() const {
		return Point(-y, x);
	}
	Point turnRight() const {
		return Point(y, -x);
	} 
	Point rotate(double ang) const {
		return Point(x * cos(ang) - y * sin(ang), x * sin(ang) + y * cos(ang));
	}

	inline double length() {
		return sqrt(x * x + y * y);
	}

	void normalize() {
		double r = length();
		if (sign(r) != 0)
			x /= r,y /= r;
	}

	bool operator < (const Point & other) const {
		return y < other.y || y == other.y == 0 && x < other.x;
	}

	bool operator == (const Point & other) const {
		return x == other.x && y == other.y;
	}

	/** Consider the float precision to judge whether two points are the same */
	inline bool equal(const Point & other) const {
		return sign(x - other.x) == 0 && sign(y - other.y) == 0;
	}

	friend ostream & operator << (ostream & out , const Point & point)
	{
		out << "(" << point.x << "," << point.y << ")";
		return out;
	}

};

inline Point operator + (const Point & a, const Point & b) {
	return Point(a.x + b.x, a.y + b.y);
}

inline Point operator - (const Point & a , const Point & b) {
	return Point(a.x - b.x, a.y - b.y);
}

inline Point operator * (const Point & a , double d) {
	return Point(a.x * d, a.y * d);
}

inline Point operator / (const Point & a , double d) {
	if (sign(d) == 0) return Point();
	return Point(a.x / d, a.y / d);
}

inline bool operator == (const Point & a , const Point & b) {
	return a.x == b.x && a.y == b.y;
}

inline bool operator != (const Point & a , const Point & b) {
	return a.x != b.x || a.y != b.y;
}

inline double dist(const Point & a , const Point & b) {
	return sqrt(sqr(a.x - b.x) + sqr(a.y - b.y));
}

inline double sqrdist(const Point & a , const Point & b)
{
	return sqr(a.x - b.x) + sqr(a.y - b.y);
}

/** Cross Product */
inline double operator ^ (const Point & a , const Point & b) {
	return a.x * b.y - a.y * b.x;
}

inline double cross(const Point & p , const Point & a , const Point & b) {
	return (a - p) ^ (b - p);
}

/** Dot product */
inline double operator * (const Point & a , const Point & b) {
	return a.x * b.x + a.y * b.y;
}

inline double dot(const Point & p , const Point & a , const Point & b) {
	return (a - p) * (b - p);
}

/** Whether Point p is on the segment [a, b] or not */
inline bool onSeg(const Point & p , const Point & a , const Point & b) {
	return ( sign(cross(p, a, b)) == 0 && sign(dot(p, a, b)) <= 0) ;
}

/** Whether Point p is on the ray from a to b */
inline bool onRay(const Point & p , const Point & a , const Point & b) {
	return ( sign(cross(p, a, b)) == 0 && sign(dot(a, p, b)) >= 0) ;
}

/** Whether Point p is on the straight line a-b or not */
inline bool onLine(const Point & p , const Point & a , const Point & b) {
	return sign(cross(p,a,b))==0;
}

/** Find the intersection point of two strait lines a-b and c-d, 
storing the result in p */
vector<Point> interLine(const Point & a , const Point & b , const Point & c , const Point & d) 
{
	vector<Point> inters;
	double u = cross(a, c, b), v = cross(a, b, d);
	if (sign(u + v) == 0) return inters;
	Point p = (c * v + d * u) / (u + v);
	inters.push_back(p);
	return inters;
}

/** Find the intersection point of two segment a-b and c-d,
storing the result in p */
vector<Point> interSeg(const Point & a , const Point & b , const Point & c , const Point & d)
{
	vector<Point> inters = interLine(a, b, c, d);
	if (inters.size() == 0) return inters;
	const Point & p = inters[0];
	if (onSeg(p, a, b) && onSeg(p, c, d)) 
		return inters;
	else {
		inters.clear();
		return inters;
	}       
}

/** Intersection of two circles, 
whose centers are a and b, radius are r1 and r2, respectively
Return the number of intersections, storing the result to array p */
vector<Point> interCir(const Point & a , const Point & b , double r1 , double r2) {
	vector<Point> ret;
	double d = dist(a, b), d1 = ((sqr(r1) - sqr(r2)) / d + d) / 2;
	if (sign(r1 + r2 - d) < 0 || sign(abs(r1 - r2) - d) > 0) return ret;
	Point mid = a + ((b - a) / d) * d1;
	if (sign(r1 + r2 - d) == 0) {
		ret.push_back(mid);
		return ret;
	}
	Point incr = (b - a).turnLeft() / d * sqrt(sqr(r1) - sqr(d1));
	ret.push_back(mid - incr);
	ret.push_back(mid + incr);
	return ret;
}

/** Intersection of a line and a circle. The circle's center is o, radius is r, 
while the straight line is a-b
Return the number of intersections, storing the result to array p
*/
vector<Point> interLineCir(const Point & o , double r , const Point & a , 
	const Point & b) {
		vector<Point> ret;
		double d = dist(a, b), h = fabs(cross(a, b, o)) / d;
		if (sign(r - h) < 0) return ret;
		double d1 = ((sqr(dist(o, a)) - sqr(dist(o, b))) / d + d) / 2;
		Point mid = a + (b - a) / d * d1;
		if (sign(r - h) == 0) {
			ret.push_back(mid);
			return ret;
		}
		Point incr = (b - a) / d * sqrt(sqr(r) - sqr(h));
		ret.push_back(mid - incr);
		ret.push_back(mid + incr);
		return ret;
}

/** Tangent from Point a, to circle whose center is o, radius is r
Return the number of intersections, storing the result to array p
*/
vector<Point> tangentCirPoint(const Point & o , double r , const Point & a) {
	vector<Point> ret;
	double d = dist(a, o);
	if (sign(d - r) < 0) return ret;
	if (sign(r - d) == 0) {
		ret.push_back(a);
		return ret;
	}
	return interCir(o, a, r, sqrt(sqr(d) - sqr(r)));
}

inline bool inCircle(const Point & p, const pair<Point, double> & circle) {
	double dis = dist(p, circle.first);
	return sign(dis - circle.second) <= 0;
}


struct Line
{
	Point start;
	Point stop;
	Point ori;

	void Refine()
	{
		if (stop < start) swap(start,stop);
		ori = stop - start;
	}

	Line() {;}
	Line(Point & startPoint , Point & stopPoint)
	{
		start = startPoint;
		stop = stopPoint;
		Refine();
	}

	Line(double startX,double startY,double stopX,double stopY)
	{
		start = Point(startX,startY);
		stop = Point(stopX,stopY);
		Refine();
	}

	bool operator < (const Line & other) const
	{
		int signL = sign (this->ori ^ other.ori);
		if (signL > 0) return true;
		if (signL < 0) return false;
		return sign(cross(start, stop, other.start)) > 0;
	}

	bool operator == (const Line & other) const {
		return sign(this->ori ^ other.ori) == 0;
	}

	friend ostream & operator << (ostream & out , Line line)
	{
		out << line.start << " -> " << line.stop << " (" << line.ori << ") ";
		return out;
	}
};


/** A Convex ploygon */
class PigPolygon 
{
public:
	/** Points */
	vector<Point> pt;

	void clear() { 
		pt.clear(); 
	}

	inline int size() const { return (int)pt.size(); }
	inline void addPoint(Point a) { pt.push_back(a); }
	inline void removePoint() { pt.pop_back(); }

	/** Whether a Point is inside a PigPolygon, 
	* 1 : inside, 
	* -1 : outside, 
	*  0 : on the edge
	*/
	int inside(const Point & p) 
	{
		int N = pt.size();
		bool positiveSign = false;
		bool negativeSign = false;

		for (int i = 0; i < N; ++i)
		{
			Point & now = pt[i];
			Point & next = pt[(i + 1) % N];
			Point t1 = next - now;
			Point t2 = p - next;
			int s = sign(t1 ^ t2);
			if (s == 0) 
			{
				if (onSeg(p, now, next))
					return 0;
				else 
					return -1;
			}
			if (s > 0) positiveSign = true;
			if (s < 0) negativeSign = true;
		}

		if (positiveSign && negativeSign) 
			return -1;
		else
			return 1;
	}



	/** Cut the PigPolygon by the strait line a-b, 
	preserving the left side of the straight line
	Provided that the PigPolygon is convex
	*/
	PigPolygon cut(const Point & a, const Point & b) {
		int size = this->size();
		addPoint(pt[0]);
		PigPolygon newPoly;
		Point temp;
		for (int i = 0; i < size; i++) {
			if (sign(cross(a, b, pt[i])) >= 0) newPoly.addPoint(pt[i]);
			if (sign(cross(a, b, pt[i])) * sign(cross(a, b, pt[i + 1])) < 0) {
				vector<Point> inters = interLine(pt[i], pt[i + 1], a, b);
				if (inters.size() > 0) 
					newPoly.addPoint(inters[0]);
			}
		}
		removePoint();
		return newPoly;
	}

	/**
	* Intersect a line with this PigPolygon
	* return the intersect points 
	*/
	vector<Point> polyInterLine(const Point & a, const Point & b)
	{
		int N = pt.size();
		vector<Point> ret;
		Point temp;
		for (int i = 0; i < N; ++i)
		{
			Point & now = pt[i];
			Point & next = pt[(i + 1) % N];
			if (sign(cross(a, b, now)) * sign(cross(a, b, next)) <= 0)
			{
				vector<Point> inters = interLine(a, b, now, next);
				if (inters.size() > 0)
					ret.push_back(inters[0]);
			}
		}
		return ret;
	}

	/** return the area of this PigPolygon */
	double area()
	{
		int size = this->size();
		if (size <= 2) return 0.0;

		addPoint(this->pt[0]);
		double ans = 0;
		for (int i = 0; i < size; ++i) ans += (pt[i] ^ pt[i+1]);
		removePoint();

		return fabs(ans / 2);
	}

	/** return the length of this PigPolygon */
	double length()
	{
		int size = this->size();
		addPoint(pt[0]);
		double ans = 0;
		for (int i = 0; i < size; ++i) ans += dist(pt[i], pt[i+1]);
		removePoint();
		return ans;
	}

	inline int next(int x)
	{
		return (x + 1) % this->size();
	}

	/** Get the Diameter of this PigPolygon */
	pair<Point, Point>  getDiameter()
	{
		if (this->size() == 0) 
			return make_pair(Point(0,0), Point(0,0));
		if (this->size() == 1)
			return make_pair(pt[0], pt[0]);
		if (this->size() == 2)
			return make_pair(pt[0], pt[1]);

		pair<Point, Point> answer;
		double diameter = 0.0;

		for (unsigned int i = 0; i < pt.size(); ++i)
			for (unsigned int j = i + 1; j < pt.size(); ++j) {
				double now = sqrdist(pt[i], pt[j]); 
				if (now > diameter) {
					diameter = now;
					answer = make_pair(pt[i], pt[j]);
				}
			}
			return answer;
	}

	/** Given three points, find a circle that cover these three points which has
	* minimal ares
	*/
	pair<Point, double> minCoverCircle(const Point & a, const Point & b, const
		Point & c) 
	{
		if (a.equal(b)) 
			return make_pair((b + c) / 2, dist(b, c) / 2);
		if (b.equal(c)) 
			return make_pair((c + a) / 2, dist(c, a) / 2);
		if (c.equal(a))
			return make_pair((a + b) / 2, dist(a, b) / 2);

		if (onLine(a, b, c)) {
			if (onSeg(a, b, c)) 
				return make_pair((b + c) / 2, dist(b, c) / 2);
			if (onSeg(b, a, c))
				return make_pair((a + c) / 2, dist(a, c) / 2);
			return make_pair((a + b) / 2, dist(a, b) / 2);
		}

		pair<Point, double> minCircle;
		minCircle.second = INF;

		pair<Point, double> cir1 = make_pair((a + b) / 2, dist(a, b) / 2);
		pair<Point, double> cir2 = make_pair((b + c) / 2, dist(b, c) / 2);
		pair<Point, double> cir3 = make_pair((a + c) / 2, dist(a, c) / 2);

		if (inCircle(c, cir1) && cir1.second < minCircle.second)
			minCircle = cir1;
		if (inCircle(a, cir2) && cir2.second < minCircle.second)
			minCircle = cir2;
		if (inCircle(b, cir3) && cir3.second < minCircle.second)
			minCircle = cir3;

		Point mid1 = (a + b) / 2;
		Point mid2 = (b + c) / 2;
		Point turn1 = (b - a).turnLeft();
		Point turn2 = (c - b).turnLeft();
		turn1 = mid1 + turn1;
		turn2 = mid2 + turn2;

		vector<Point> inters = interLine(mid1, turn1, mid2, turn2);
		pair<Point, double> cir4 = make_pair(inters[0], dist(inters[0], a));
		if (cir4.second < minCircle.second)
			minCircle = cir4;

		return minCircle;
	}


	/** Find a circle that cover the PigPolygon with minimal area 
	*/
	pair<Point, double> minCoverCircle()
	{
		vector<Point> & points = pt;

		if (points.size() == 0)
			return make_pair(Point(0, 0), 0);
		if (points.size() == 1)
			return make_pair(points[0], 0);
		if (points.size() == 2)
			return make_pair((points[0] + points[1]) / 2, dist(points[0], points[1]) / 2);

		Point a = points[0];
		Point b = points[1];
		Point c = points[2];
		pair<Point, double> nowCircle = minCoverCircle(a, b, c);

		while (true) {
			bool found = false;
			double maxDis = nowCircle.second;
			Point maxDisPoint;
			for (unsigned int i = 0; i < points.size(); ++i) {
				double dis = dist(points[i], nowCircle.first);
				if (sign(dis - maxDis) > 0) {
					maxDis = dis;
					maxDisPoint = points[i];
					found = true;
				}
			}

			if (!found) break;
			Point d = maxDisPoint;

			Point newa, newb, newc;
			pair<Point, double> cir1 = minCoverCircle(a, b, d);
			pair<Point, double> cir2 = minCoverCircle(a, c, d);
			pair<Point, double> cir3 = minCoverCircle(b, c, d);
			pair<Point, double> minCircle;
			minCircle.second = INF;
			if (inCircle(c, cir1) && cir1.second < minCircle.second) {
				minCircle = cir1;
				c = d;
			}
			else {
				if (inCircle(b, cir2) && cir2.second < minCircle.second) {
					minCircle = cir2;
					b = d;
				}
				else {
					if (inCircle(a, cir3) && cir3.second < minCircle.second) {
						minCircle = cir3;
						a = d;
					}
				}
			}
			nowCircle = minCircle;
		}
		return nowCircle;
	}

	friend ostream & operator << (ostream & out, const PigPolygon & poly) 
	{
		int size = poly.size();
		for (int i = 0; i < size; ++i) 
			out << poly.pt[i] << "\t";
		return out;
	}
};

struct ConvexHull
{
	static inline bool cmp(const Point &a, const Point &b) {
		return a.y < b.y || (a.y == b.y && a.x < b.x);
	}

	/** Calculate the convex hull represented by array p */
	static PigPolygon convexHull(const vector<Point> & points, bool needInnerEdgePoint = false) 
	{
		int i;
		vector<Point> p(points.begin(), points.end());
		vector<int> stack(p.size() + 1);
		PigPolygon poly;
		poly.clear();
		if (p.size() == 0) 
			return poly;
		if (p.size() == 1) {
			poly.addPoint(points[0]);
			return poly;
		}

		sort(p.begin(), p.end(), ConvexHull::cmp); 

		int threshold = 0;
		if (needInnerEdgePoint) 
			threshold = 1;

		int N = (int)p.size();
		int top = -1;
		stack[++top] = 0; stack[++top] = 1;
		for (i = 2; i < N; i++)
		{
			while (top >= 1 && 
				sign(cross(p[stack[top - 1]], p[i], p[stack[top]])) >= threshold) top--; 
			stack[++top] = i;
		}
		int temp_top = top; 
		stack[++top] = N - 2;
		for (i = N - 3; i >= 0; i--)
		{
			while (top >= temp_top + 1 && 
				sign(cross(p[stack[top - 1]], p[i], p[stack[top]])) >= threshold) top--; 
			stack[++top] = i;
		}

		for (i = 0; i < top; ++i) 
			poly.addPoint(p[stack[i]]);
		return poly;
	}
};

// 计算一个点到矩形的最短距离 
int getMinDisToRect(CvRect & rect, int x, int y)
{
	int x1 = rect.x;
	int x2 = rect.x + rect.width - 1;
	int y1 = rect.y;
	int y2 = rect.y + rect.height - 1;
	int dx = 0;
	if (x < x1) dx = x1 - x; if (x > x2) dx = x - x2;
	int dy = 0;
	if (y < y1) dy = y1 - y; if (y > y2) dy = y - y2;
	return dx + dy;
}

// 计算一个矩形到另一个矩形的最短距离
int getMinDisToRect(CvRect & rect1, CvRect & rect2)
{
	int ret = 99999999;
	ret = min(ret, getMinDisToRect(rect2, rect1.x, rect1.y));
	ret = min(ret, getMinDisToRect(rect2, rect1.x, rect1.y + rect1.height - 1));
	ret = min(ret, getMinDisToRect(rect2, rect1.x + rect1.width - 1, rect1.y));
	ret = min(ret, getMinDisToRect(rect2, rect1.x + rect1.width - 1, rect1.y + rect1.height - 1));
	return ret;
}

// 判断一个矩形是否在另一个矩形内部
bool isRectInRect(CvRect & rect1, CvRect & rect2)
{
	return (rect1.x >= rect2.x && rect1.x + rect1.width <= rect2.x + rect2.width &&
		rect1.y >= rect2.y && rect1.y + rect1.height <= rect2.y + rect2.height);
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

void rectFade(CvRect & r, CvRect & orz)
{
  int nh = (r.height + orz.height) / 2;
  int nw = (r.width + orz.width) / 2;
  r.y = r.y + r.height - nh;
  r.width = nw;
  r.height = nh;
}

void rectRestrict(CvRect & r, CvRect & orz)
{
  int nh = min(r.height, orz.height);
  int nw = min(r.width, orz.width);
  r.y = r.y + r.height - nh;
  r.width = nw;
  r.height = nh;
}

void rectFade(CvRect & r, list<CvRect> & rects)
{
  int sel = min((int)rects.size(), 3);
  int nh = r.height;
  int nw = r.width;
  list<CvRect>::reverse_iterator itr = rects.rbegin();
  for (int i = 0; i < sel; ++i)
  {
    nh += (*itr).height;
    nw += (*itr).width;
    itr++;
  }

  nh /= (sel + 1);
  nw /= (sel + 1);
  r.y = r.y + r.height - nh;
  r.width = nw;
  r.height = nh;
}

double dist(CvScalar & s1, CvScalar & s2)
{
	return sqrt(sqr(s1.val[0] - s2.val[0]) + sqr(s1.val[1] - s2.val[1]) + sqr(s1.val[2] - s2.val[2]));
}

double dist(CvPoint & t1, CvPoint & t2)
{
	return sqrt(sqr(t1.x - t2.x) + sqr(t1.y - t2.y));
}

#endif