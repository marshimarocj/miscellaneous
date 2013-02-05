#ifndef ALGORITHM
#define ALGORITHM

#include "stdafx.h"

vector<int> K_MEANS(vector<vector<double> > & s)
{
  int R = (int) s.size();
  int C = (int) s[0].size();
  CvMat * samples = cvCreateMat(R, C, CV_32FC1);
  CvMat * ids = cvCreateMat(R, 1, CV_32SC1);

  for (int i = 0; i < R; ++i)
    for (int j = 0; j < C; ++j)
      cvmSet(samples, i, j, s[i][j]);

  vector<int> ret(R, 0);

  double lastcompat = 1e10;
  for (int cluster_count = 2; ; ++cluster_count)
  {
    double compat;
    cvKMeans2( samples, cluster_count, ids,
      cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1.0 ), 10, 0, 0, 0, &compat);

    debug2(cluster_count, compat);

    for (int i = 0; i < R; ++i)
      ret[i] = ids->data.i[i];
    
    if (compat > lastcompat && cluster_count > 10) break;
    lastcompat = compat;
  }

  cvReleaseMat( &samples );
  cvReleaseMat( &ids );

  return ret;
}

#endif
