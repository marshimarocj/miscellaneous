#ifndef STDAFX
#define STDAFX

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string.h>
#include <stack>
#include <string>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <ctime>
#include <queue>
#include <list>
#include <bitset>
#include <vector>
#include <queue>
#include <map>
#include <cmath>
#include <set>
#include <map>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <set>

//#define DSP
//#define MAC
#define WINDOWS
#include "OpenCvDsp.h"

#define debug1(x) cout << #x" = " << x << endl;
#define debug2(x, y) cout << #x" = " << x << " " << #y" = " << y << endl;
#define debug3(x, y, z) cout << #x" = " << x << " " << #y" = " << y << " " << #z" = " << z << endl;
#define debug4(x, y, z, w) cout << #x" = " << x << " " << #y" = " << y << " " << #z" = " << z << " " << #w" = " << w << endl;


using namespace std;

template <class T>
ostream & operator << (ostream & out, const vector<T> & data)
{ out << "["; for (int i = 0; i < data.size(); ++i) out << data[i] << (i == data.size() - 1 ? "" : ","); out << "]"; return out; }

template <class T>
ostream & operator << (ostream & out, const set<T> & s)
{ out << "{"; for (typename set<T>::iterator itr = s.begin(); itr != s.end(); ++itr) out << *itr << " "; out << "}"; return out; }

template <class T>
ostream & operator << (ostream & out, const multiset<T> & s)
{ out << "{"; for (typename multiset<T>::iterator itr = s.begin(); itr != s.end(); ++itr) out << *itr << " "; out << "}"; return out; }

template <class T1, class T2>
ostream & operator << (ostream & out, const pair<T1, T2> & p)
{ out << "(" << p.first << "," << p.second << ")"; return out; }

template <class T1, class T2>
ostream & operator << (ostream & out, const map<T1, T2> & m)
{ 
  for (typename map<T1, T2>::const_iterator itr = m.begin(); itr != m.end(); ++itr)
    out << itr->first << "->" << itr->second << " ";
  return out;
}

// 程序编译开关
const int DEMO = 0;
const int DEBUG = 1;
const int ALLOW_LARGE_MOTOR = 0;
const int DALI_SPEC = 0;

// 
const int ADDITIONAL_RULE = 1;
const int USING_SVM = 1;

// SVM 车辆标注字符及意义
const char SVM_NONE = '_';
const char SVM_MOTOR = 'm';
const char SVM_CAR = 's';
const char SVM_BIGCAR = 'b';

// SVM 车辆分类ID
const int SVM_NONE_ID = 1;
const int SVM_CAR_ID = 2;

#endif
