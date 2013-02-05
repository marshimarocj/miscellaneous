#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <set>
#include <algorithm>
#include <queue>
#include <cassert>
#include <fstream>
#include <sstream>
#include <bitset>
#include <stack>
#include <list>
using namespace std;
#define debug1(x) cout << #x" = " << x << endl;
#define debug2(x, y) cout << #x" = " << x << " " << #y" = " << y << endl;
#define debug3(x, y, z) cout << #x" = " << x << " " << #y" = " << y << " " << #z" = " << z << endl;
#define debug4(x, y, z, w) cout << #x" = " << x << " " << #y" = " << y << " " << #z" = " << z << " " << #w" = " << w << endl;

ostream & operator << (ostream & out, const vector<int> & v)
{
  out << v.size() << "[";
  for (int i = 0; i < v.size(); ++i)
    out << v[i] << " ";
  out << "]";
  return out;
}

class PrimeJudger
{
public:
  const static int bits = 19, mask = (1<<bits) - 1;
  static unsigned long long mul(unsigned long long x, unsigned long long y, unsigned long long M) 
  {
    if (x <= UINT_MAX && y <= UINT_MAX) return x * y % M;
    unsigned long long rslt = y * (x & mask) % M;
    while (x >>= bits) rslt = (rslt + (y = (y << bits) % M) * (x & mask)) % M;
    return rslt;
  }
  
  static int mtest(unsigned long long n, unsigned long long a) {
    unsigned long long s, t, nmin1 = n - 1;
    int r;
    for (s = nmin1, r = 0; !(s & 1); s >>= 1, r++) ;
    for(t = a; s >>= 1; ) 
    {
      a = mul(a, a, n);
      if (s & 1) t = mul(t, a, n);
    }
    if (t == 1 || t == nmin1) return 1;
    while (--r) if ((t = mul(t, t, n)) == nmin1) return 1;
    return 0;
  }
  
  static bool isPrime(unsigned long long n) {
    if(n<29) return n==2 || n==3 || n==5 || n==7 || n==11 || n==13 || n==17 || n==19 || n==23;
    if(!(n&1 && n%3 && n%5 && n%7 && n%11 && n%13 && n%17 && n%19 && n%23)) return 0;
    return mtest(n, 2) && mtest(n, 1215) &&
    (n==17431 || mtest(n, 34862) && (n==3281359 || mtest(n, 574237825)));
  }
};

class MinHash
{
public:
  long long A, B;
  // vocab size, must be a prime 
  long long P;
  
  MinHash(long long _P)
  {
    P = _P;
    A = (rand() * rand()) % P;
    A = (A + P) % P;
    B = (rand() * rand()) % P;
    B = (B + P) % P;
  }
  
  MinHash() {}
  
  inline int hash(int x) { return (int) ((A * x + B) % P); }
  
  int minhash(vector<int> & x)
  {
    int minValue = (int) (P + 1);
    for (unsigned int i = 0; i < x.size(); ++i)
    {
      int v = hash(x[i]);
      minValue = min(minValue, v);
    }
    return minValue;
  }
};

const double lamdastar = 1.5;
const double alpha = 0.05;
const double r0 = 10;
class CosetMiner
{
public:
  static const int SKETCHES = 85;
  static const int SKETCHES_SIZE = 3;
  static const long long BIGP = (1LL << 31) - 1;
  
  // 每个Visual Word出现在image中的概率 P(A) = |A| / D
  vector<double> PA;
  
  // 文档总数
  int D;
  
  // 文档总数的上界素数
  int DP;
  
  // 单词总数
  int W;
  
  // 每个单词，针对每组MinHash所得到的hash值
  vector<vector<int> > hashSValues;
  vector<vector<int> > hashValues;
  
  // Hash倒排表，每个Hash值，在那些单词中被Hash到了
  map<int, vector<int> > revertHash[SKETCHES];
  
  // 根据lamda计算的无向图
  vector<vector<int> > graph;
  
  // 在无向图中，每个单词属于哪个联通分量，从0开始编号
  vector<int> comID;
  int coms;
  
  // 生成的随机hash函数
  vector<MinHash> hashFuncs;
  
  // 最终找到的core-fri集合
  vector<pair<vector<int>, vector<int> > > ans;
  
  
  // 对Hash函数进行初始化
  void initHashFunc()
  {
    hashFuncs.resize(SKETCHES * SKETCHES_SIZE);
    for (int i = 0; i < hashFuncs.size(); ++i)
      hashFuncs[i] = MinHash(DP);
  }
  
  // 开始Hash
  void startHash(vector<vector<int> > & wordocc)
  {
    hashValues.resize(W);
    hashSValues.resize(W);
    
    for (int i = 0; i < W; ++i)
    {
      if (i % 1000 == 0) cout << "Already hash " << i << " word, total = " << W << endl;
      
      for (int j = 0; j < SKETCHES; ++j)
      {
        long long hash = 0;
        for (int k = 0; k < SKETCHES_SIZE; ++k)
        {
          long long nowhash = hashFuncs[j * 3 + k].minhash(wordocc[i]);
          hash = (hash * DP + nowhash) % BIGP;
          hashValues[i].push_back((int)nowhash);
        }
        hashSValues[i].push_back((int)hash);
        revertHash[j][(int)hash].push_back(i);
      }
    }
    
    map<int, vector<int> > hash0 = revertHash[0];
  }
  
  // 根据公式6计算lamda
  inline double getLamda(int A, int B)
  {
    double ret = log(PA[A] * PA[B]);
    double ovr = 0;
    for (int i = 0; i < hashValues[A].size(); ++i)
      if (hashValues[A][i] == hashValues[B][i]) 
        ovr++;
    ovr /= hashValues[A].size();
    ret /= (log(PA[A] + PA[B]) + log(ovr) - log(ovr + 1));
    return ret;
  }
  
  // 生成无向图
  void genGraph()
  {
    graph.resize(W);
    vector<set<int> > tempgraph(W);
    
    cout << "Generating Edge..." << endl;
    int addedges = 0;
    for (int i = 0; i < SKETCHES; ++i)
    {
      cout << i << " " << addedges << " ";
      flush(cout);
      for (map<int, vector<int> >::iterator itr = revertHash[i].begin(); 
           itr != revertHash[i].end(); ++itr)
      {   
        vector<int> & ids = itr->second;
        for (int a = 0; a < ids.size(); ++a)
          for (int b = a + 1; b < ids.size(); ++b)
          {
            int A = ids[a];
            int B = ids[b];
            if (A > B) continue;
            if (tempgraph[A].find(B) != tempgraph[A].end()) continue;
            double l = getLamda(A, B);
            if (l > lamdastar) 
            {
              addedges++;
              tempgraph[A].insert(B);
              tempgraph[B].insert(A);
            }
          }
      }
    }
    cout << endl;
    
    int edges = 0;
    for (int i = 0; i < W; ++i)
    {
      for (set<int>::iterator itr = tempgraph[i].begin(); itr != tempgraph[i].end(); ++itr)
        graph[i].push_back(*itr);
      edges += tempgraph[i].size();
    }
    cout << "Totally " << edges << " edges" << endl;
  }
  
  // 填充一个联通分量
  void floodfill(int now, int id)
  {
    comID[now] = id;
    for (int i = 0; i < graph[now].size(); ++i)
    {
      int nextid = graph[now][i];
      if (comID[nextid] == -1) floodfill(nextid, id);
    }
  }
  
  // 生成所有的co-ocset cores
  void genComponent()
  {
    coms = 0;
    comID = vector<int>(W, -1);
    for (int i = 0; i < W; ++i)
      if (comID[i] == -1) 
      {
        floodfill(i, coms);
        coms++;
      }
  }
  
  // 对于每个cores，寻找fringes 
  void findFringers(vector<vector<int> > & wc)
  {
    cout << "Dealing x core " << endl;
    for (int k = 0; k < coms; ++k)
    {
      cout << k << " ";
      vector<int> cores;
      vector<int> fringes;
      
      int coreSize = 0;
      vector<int> count(D, 0);
      for (int i = 0; i < W; ++i)
        if (comID[i] == k)
        {
          coreSize++;
          cores.push_back(i);
          for (int j = 0; j < wc[i].size(); ++j)
          {
            int docID = wc[i][j];
            count[docID]++;
          }
        }
      
      vector<bool> isContainCoreDoc(D, false);
      int KiSize = 0;
      for (int i = 0; i < D; ++i)
        if (count[i] > coreSize * alpha)
        {
          KiSize++;
          isContainCoreDoc[i] = true;
        }
      
      // 对于每个单词，如果他所在的文档，包含core，且他不是core，则检查他是否是fringe
      for (int i = 0; i < W; ++i)
        if (comID[i] != k)
        {
          // 统计有多少个文档包含这个单词，且是包含core的文档
          int cnt = 0;
          for (int j = 0; j < wc[i].size(); ++j)
          {
            int docID = wc[i][j];
            if (isContainCoreDoc[docID]) 
              cnt++;
          }
          
          double PA_Ki = (double) cnt / coreSize;
          double p = PA_Ki / PA[i];
          if (p > r0) 
            fringes.push_back(i);
        }
      
      ans.push_back(make_pair(cores, fringes));
    }
    cout << endl;
  }
  
  // 构造函数，输入的是倒排表，每个单词对应他出现的文档列表，文档ID请从0开始编号。
  CosetMiner(vector<vector<int> > & wordoccur)
  {
    // 初始化参数
    D = 0;
    W = wordoccur.size();
    
    for (int i = 0; i < W; ++i)
      for (int j = 0; j < wordoccur[i].size(); ++j)
      {
        int docID = wordoccur[i][j];
        D = max(D, docID);
      }
    
    // 自动找到比当前文档个数大的素数作为模
    DP = D;
    while (!PrimeJudger::isPrime(DP)) DP++;
    cout << "DP Success" << endl;
    
    PA.resize(W);
    for (int i = 0; i < W; ++i)
      PA[i] = (double) wordoccur[i].size() / D;
    cout << "PA Success" << endl;
    
    cout << "Documents: " << D << " " << DP << endl;
    cout << "Words: " << W << endl;
    
    // 初始化Hash 函数，修改HashFunc数据
    initHashFunc();
    cout << "HashFunc Success" << endl;
    
    // 开始Hash，修改HashValue, HashSValue, revertHash表
    startHash(wordoccur);
    cout << "Hash Success" << endl;
    
    // 生成无向图，修改graph，记录无向图
    genGraph();
    cout << "Undirect graph success" << endl;
    
    // 计算联通分量，修改comID，记录每个点所在的联通分量编号
    genComponent();
    cout << "Connected Component Success" << endl;
    
    // 计算core-fringe，计算最后的ans
    findFringers(wordoccur);
    cout << "Data Mining Complete" << endl;
  }
};

// 读入文件中的数据，docword是正排表，wordocc是倒排表
vector<vector<int> > docword;
vector<vector<int> > wordocc;
void read(string file, vector<vector<int> > & docword, vector<vector<int> > & wordocc)
{
  ifstream fin(file.data());
  string line;
  
  cout << "read " << file << "..." << endl;
  
  // 读入数据文档从1开始编号，单词从0开始
  int minDocID = 99999999;
  int maxDocID = 0;
  int maxWordID = 0;
  int minWordID = 99999999;
  int lineCnt = 0;
  while (getline(fin, line))
  {
    lineCnt++;
    
    if (lineCnt % 100000 == 0) 
    {
      cout << lineCnt << endl;
      //break;
    }
    
    if (lineCnt % 10 != 0) continue;
    istringstream sin(line);
    string docName;
    sin >> docName;
    int docID;
    docID = lineCnt - 1;
    
    maxDocID = max(maxDocID, docID);
    minDocID = min(minDocID, docID);
    if (docID + 1 > docword.size())
      docword.resize(docID + 1);
    
    int wordID;
    while (sin >> wordID)
    {
      maxWordID = max(maxWordID, wordID);
      minWordID = min(minWordID, wordID);
      docword[docID].push_back(wordID);
      
      if (wordID + 1 > wordocc.size())
        wordocc.resize(wordID + 1);
      wordocc[wordID].push_back(docID);
    }
  }
  
  debug2(minDocID, maxDocID);
  debug2(minWordID, maxWordID);
}

void rand(vector<vector<int> > & wordocc);

void rand(vector<vector<int> > & wordocc)
{
  wordocc.resize(100);
  for (int i = 0; i < wordocc.size(); ++i)
  {
    int lower = i / 10 * 10;
    int upper = i / 10 * 10 + 10;
    for (int j = lower; j < upper; ++j)
      wordocc[i].push_back(j);
  }
}

map<string, string> imgname2url;
map<int, vector<pair<string, int> > > occur;
void readocc()
{
  ifstream fin("G:\\data\\denseSiftMerge.dat");
  string name, url;
  int num;
  int cnt = 0;
  string line;
  while (fin >> name >> url >> num)
  {
    cnt ++;
    if (cnt % 100 != 0)
    {
      getline(fin, line);
      continue;
    }
    if (cnt % 10000 == 0) cout << cnt << endl;

    vector<int> posis;
    for (int i = 0; i < num; ++i)
    {
      int x, y;
      fin >> x >> y;
      posis.push_back(x * 10000 + y);
    }
    vector<int> ids;
    for (int i = 0; i < num; ++i)
    {
      int id;
      fin >> id;
      ids.push_back(id);
    }

    for (int i = 0; i < ids.size(); ++i)
    {
      int id = ids[i];
      if (occur[id].size() < 5)
        occur[id].push_back(make_pair(url, posis[i]));
    }
  }

  cout << occur.size() << endl;
}

int main()
{
  /**
  read("C:\\Users\\yhzhu\\Downloads\\densesift", docword, wordocc);
  //rand(wordocc);
  CosetMiner cm(wordocc);
  
  vector<pair<vector<int>, vector<int> > > & ans = cm.ans;
  debug1(ans.size());
  
  ofstream fout("D:\\result_minhash.txt");  
  for (int i = 0; i < ans.size(); ++i)
    fout << ans[i].first << endl << ans[i].second << endl;
  fout.close();
  */

  readocc();

  ifstream fin("D:\\result_minhash.txt");
  ofstream fout("D:\\minhash_cores.txt");
  int coresize, frisize;
  map<int, int> cores;
  map<int, int> fris;
  while (fin >> coresize)
  {
    string line1;
    string line2;
    getline(fin, line1);
    fin >> frisize;
    getline(fin, line2);

    cores[coresize]++;
    fris[frisize]++;

    if (coresize >= 3)
    {
      fout << line1 << endl;
      fout << line2 << endl;
    }
  }
  fout.close();

  //ofstream fout2("D:\\fri.txt");

  //for (map<int, int>::iterator itr = cores.begin(); itr != cores.end(); ++itr)
    //cout << itr->first << " " << itr->second << endl;

  cout << "Please input wordID" << endl;
  int wordid;
  
  while (cin >> wordid)
  {
    vector<pair<string, int> > & occ = occur[wordid];
    for (int i = 0; i < occ.size(); ++i)
      cout << occ[i].first << " " << occ[i].second << endl;
  }

	return 0;
}