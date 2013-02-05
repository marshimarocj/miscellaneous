#ifndef BIPARTITE_MAX_MATCH
#define BIPARTITE_MAX_MATCH

#include "stdafx.h"
using namespace std;

/** id from 0 */
const int BM_MAXN = 208;
const int BM_INFI = 99999999;
struct BipartiteMaxMatch
{
	/** 
	@see http://en.wikipedia.org/wiki/Hungarian_algorithm
	*/
	int N;
	int w[BM_MAXN][BM_MAXN];
	int label[BM_MAXN];
	bool g[BM_MAXN][BM_MAXN];
	int match[BM_MAXN];
	int nowMatch;

	BipartiteMaxMatch()
	{
		nowMatch = 0;
	}

	// �������ƥ�䣬������ߵ�ÿ���� first ƥ��id, second ƥ��ı�Ȩ
	vector<pair<int, int> > getMatch(int _N, int _M, vector<vector<int> > _w)
	{
		int i,j;
		N = max(_N, _M);
    for (i = 0; i < N * 2; ++i)
      memset(w[i], 0, sizeof(w[i]));

		for (i = 0; i < _N; ++i)
			for (j = 0; j < _M; ++j)
			{
				w[i][j + N] = _w[i][j];
				w[j + N][i] = _w[i][j];
			}

			initMatch();
			startMatch();

			vector<pair<int, int> > ret(_N);
			for (i = 0; i < _N; ++i)
			{
				ret[i].first = match[i] - N;
				if (ret[i].first < 0 || ret[i].first >= _M) ret[i].first = -1;
				if (ret[i].first >= 0)
					ret[i].second = w[i][match[i]];
			}
			return ret;
	}

	void initMatch()
	{
		int i, j;
		for (j = N; j < 2 * N; ++j)
			label[j] = 0;
		for (i = 0; i < N; ++i)
		{
			label[i] = 0;
			for (j = N; j < 2 * N; ++j)
				label[i] = max(label[i], w[i][j]);
		}

    for (i = 0; i < N * 2; ++i)
      memset(g[i], 0, sizeof(g[i]));
    
		//memset(g, false, sizeof(g));

		for (i = 0; i < N; ++i)
			for (j = N; j < 2 * N; ++j)
				g[i][j] = true;

		memset(match, -1, sizeof(match));
	}

	bool visited[BM_MAXN];
	int last[BM_MAXN];

	void dfs(int now)
	{
		visited[now] = true;

		for (int i = 0; i < N * 2; ++i)
			if (g[now][i] && label[now] + label[i] == w[now][i] && !visited[i])
			{
				last[i] = now;
				dfs(i);
			}
	}

	void update(int now)
	{
		while (true)
		{
			int prev = last[now];
			g[prev][now] = false;
			g[now][prev] = true;
			match[prev] = now;
			match[now] = prev;
			now = last[prev];
			g[prev][now] = true;
			g[now][prev] = false;
			if (now == -1) break;
		}
	}

	void startMatch()
	{
		int i, j;
		int nowMatch = 0;
		while (true)
		{
			if (nowMatch == N) break;

			memset(visited, false, sizeof(visited));
			memset(last, -1, sizeof(last));
			for (i = 0; i < N; ++i)
				if (match[i] == -1)
				{
					last[i] = -1;
					dfs(i);
				}

			int tar = -1;
			for (j = N; j < 2 * N; ++j)
				if (match[j] == -1 && visited[j])
				{
					tar = j;
					break;
				}

				if (tar != -1)
				{
					update(tar);
					nowMatch++;
				}
				else
				{
					int minDelta = BM_INFI;
					for (i = 0; i < N; ++i)
						if (visited[i])
							for (j = N; j < 2 * N; ++j)
								if (!visited[j])
									minDelta = min(minDelta, label[i] + label[j] - w[i][j]);
					for (i = 0; i < N; ++i)
						if (visited[i])
							label[i] -= minDelta;
					for (j = N; j < 2 * N; ++j)
						if (visited[j])
							label[j] += minDelta;
				}
		}
	}
};


#endif