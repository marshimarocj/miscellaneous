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
  
  // ÿ��Visual Word������image�еĸ��� P(A) = |A| / D
  vector<double> PA;
  
  // �ĵ�����
  int D;
  
  // �ĵ��������Ͻ�����
  int DP;
  
  // ��������
  int W;
  
  // ÿ�����ʣ����ÿ��MinHash���õ���hashֵ
  vector<vector<int> > hashSValues;
  vector<vector<int> > hashValues;
  
  // Hash���ű�ÿ��Hashֵ������Щ�����б�Hash����
  map<int, vector<int> > revertHash[SKETCHES];
  
  // ����lamda���������ͼ
  vector<vector<int> > graph;
  
  // ������ͼ�У�ÿ�����������ĸ���ͨ��������0��ʼ���
  vector<int> comID;
  int coms;
  
  // ���ɵ����hash����
  vector<MinHash> hashFuncs;
  
  // �����ҵ���core-fri����
  vector<pair<vector<int>, vector<int> > > ans;
  
  
  // ��Hash�������г�ʼ��
  void initHashFunc()
  {
    hashFuncs.resize(SKETCHES * SKETCHES_SIZE);
    for (int i = 0; i < hashFuncs.size(); ++i)
      hashFuncs[i] = MinHash(DP);
  }
  
  // ��ʼHash
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
  
  // ���ݹ�ʽ6����lamda
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
  
  // ��������ͼ
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
  
  // ���һ����ͨ����
  void floodfill(int now, int id)
  {
    comID[now] = id;
    for (int i = 0; i < graph[now].size(); ++i)
    {
      int nextid = graph[now][i];
      if (comID[nextid] == -1) floodfill(nextid, id);
    }
  }
  
  // �������е�co-ocset cores
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
  
  // ����ÿ��cores��Ѱ��fringes 
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
      
      // ����ÿ�����ʣ���������ڵ��ĵ�������core����������core���������Ƿ���fringe
      for (int i = 0; i < W; ++i)
        if (comID[i] != k)
        {
          // ͳ���ж��ٸ��ĵ�����������ʣ����ǰ���core���ĵ�
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
  
  // ���캯����������ǵ��ű�ÿ�����ʶ�Ӧ�����ֵ��ĵ��б��ĵ�ID���0��ʼ��š�
  CosetMiner(vector<vector<int> > & wordoccur)
  {
    // ��ʼ������
    D = 0;
    W = wordoccur.size();
    
    for (int i = 0; i < W; ++i)
      for (int j = 0; j < wordoccur[i].size(); ++j)
      {
        int docID = wordoccur[i][j];
        D = max(D, docID);
      }
    
    // �Զ��ҵ��ȵ�ǰ�ĵ��������������Ϊģ
    DP = D;
    while (!PrimeJudger::isPrime(DP)) DP++;
    cout << "DP Success" << endl;
    
    PA.resize(W);
    for (int i = 0; i < W; ++i)
      PA[i] = (double) wordoccur[i].size() / D;
    cout << "PA Success" << endl;
    
    cout << "Documents: " << D << " " << DP << endl;
    cout << "Words: " << W << endl;
    
    // ��ʼ��Hash �������޸�HashFunc����
    initHashFunc();
    cout << "HashFunc Success" << endl;
    
    // ��ʼHash���޸�HashValue, HashSValue, revertHash��
    startHash(wordoccur);
    cout << "Hash Success" << endl;
    
    // ��������ͼ���޸�graph����¼����ͼ
    genGraph();
    cout << "Undirect graph success" << endl;
    
    // ������ͨ�������޸�comID����¼ÿ�������ڵ���ͨ�������
    genComponent();
    cout << "Connected Component Success" << endl;
    
    // ����core-fringe����������ans
    findFringers(wordoccur);
    cout << "Data Mining Complete" << endl;
  }
};

// �����ļ��е����ݣ�docword�����ű�wordocc�ǵ��ű�
vector<vector<int> > docword;
vector<vector<int> > wordocc;
void read(string file, vector<vector<int> > & docword, vector<vector<int> > & wordocc)
{
  ifstream fin(file.data());
  string line;
  
  cout << "read " << file << "..." << endl;
  
  // ���������ĵ���1��ʼ��ţ����ʴ�0��ʼ
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