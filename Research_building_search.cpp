#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
using namespace std;

vector<vector<double> > features;
vector<string> files;

double dist(vector<double> & f1, vector<double> & f2)
{
  double ret = 0;
  for (int i = 0; i < f1.size(); ++i)
    ret += fabs(f1[i] - f2[i]);
  return ret;
}

void loadGIST(string file)
{
  ifstream fin(file.data());
  string filename;
  while (fin >> filename)
  {
    files.push_back(filename);
    vector<double> fea(960);
    for (int i = 0; i < 960; ++i)
      fin >> fea[i];
    features.push_back(fea);
  }
}

void search(string file)
{
  for (int i = 0; i < files.size(); ++i)
    if (files[i] == file)
    {
      vector<pair<double, int> > diss;
      for (int j = 0; j < files.size(); ++j)
        diss.push_back(make_pair(dist(features[i], features[j]), j));
      sort(diss.begin(), diss.end());
      
      for (int j = 0; j < 10; ++j)
        cout << diss[j].first << " " << files[diss[j].second] << endl;
      return;
    }
}


int main()
{
  loadGIST("/Volumes/pigpass/data/Building/buildings/104/feature_gist.txt");
  
  string query = "G:\\data\\Building\\buildings\\104\\1eec27560115ba48363600430c9ca1f2_build0.jpg";
  cout << features.size() << endl;
  search(query);
  return 0;
}



