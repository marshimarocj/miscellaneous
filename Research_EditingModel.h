#ifndef RESEARCH_EDITINGMODEL_H
#define RESEARCH_EDITINGMODEL_H

#include "stdafx.h"
#include <string>
#include <iostream>
#include <fstream>

double vpq_cloning(double p, double q)
{ return p - q; }

double f1p_cloning(double p)
{ return p; }

int sdir[4][2] = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };

class EditingModel
{
public:
	vector<vector<double> > f0, f1;
	vector<vector<bool> > omiga0, omiga1;
	int Height, Width;
	int Rows, Cols;

	int getID(int x, int y) { return x * Width + y + 1; }

	double (*vpq) (double p, double q);
	double (*f1p) (double p);

	EditingModel(vector<vector<double> > _f0, vector<vector<double> > _f1, 
		double (*_vpq) (double p, double q), double (*_f1p) (double p),
		vector<vector<bool> > _omiga0, vector<vector<bool> > _omiga1)
	{
		f0 = _f0;
		f1 = _f1;
		vpq = _vpq;
		f1p = _f1p;
		omiga0 = _omiga0;
		omiga1 = _omiga1;

		Height = f0.size();
		Width = f0[0].size();
	}

	vector<pair<pair<int, int>, double> > matrixA;
	vector<double> matrixB;

	void addEle(int row, int col, double value)
	{ matrixA.push_back(make_pair(make_pair(row, col), value)); }

	void genMatrix()
	{
		matrixB.resize(Height * Width);
		for (int i = 0; i < Height; ++i)
			for (int j = 0; j < Width; ++j)
			{
				int row = getID(i, j);
				int nowid = getID(i, j);
				double b = 0;
				int cnt = 0;

				if (omiga1[i][j])
				{
					double f = (*f1p) (f1[i][j]);
					cnt++;
					b += f;
				}

				if (omiga0[i][j])
				{
					for (int k = 0; k < 4; ++k)
					{
						int ni = i + sdir[k][0];
						int nj = j + sdir[k][1];
						if (ni >= 0 && ni < Height && nj >= 0 && nj < Width)
						{
							double fp = f0[i][j];
							double fq = f0[ni][nj];
							double v = (*vpq) (fp, fq);
							b += v;
							addEle(row, getID(ni, nj), -1);
							cnt++;
						}
					}
				}

				matrixB[row] = b;
				addEle(row, nowid, cnt);
			}
	}

	void outputMatrix(const string & file1, const string & file2)
	{
		ofstream fout(file1.data());
		for (int i = 0; i < matrixA.size(); ++i)
			fout << matrixA[i].first.first << " " << matrixA[i].first.second << " " << matrixA[i].second << endl;
		fout.close();

		ofstream fout2(file2.data());
		for (int i = 0; i < matrixB.size(); ++i)
			fout2 << i + 1 << " " << 1 << " " << matrixB[i] << endl;
		fout2.close();
	}
};

#endif