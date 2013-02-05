/**
 Class:	PatentLabeling
 Method:	getFigures
 Parameters:	int, int, vector <int>, vector <string>
 Returns:	vector <string>
 Method signature:	vector <string> getFigures(int H, int W, vector <int> image, vector <string> text)
 
 Method:	getPartLabels
 Parameters:	int, int, vector <int>, vector <string>
 Returns:	vector <string>
 Method signature:	vector <string> getPartLabels(int H, int W, vector <int> image, vector <string> text)
 (be sure your methods are public)
 */

#define LOCAL

#ifdef LOCAL
#include "stdafx.h"
#endif

#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <sstream>
#include <set>
#include <map>
#include <ctime>
#include <string>
#include <vector>
#include <utility>
#include <string.h>
#include <cassert>
#include <queue>
#include <bitset>

using namespace std;

string toString(int x)
{
  char temp[20];
  sprintf(temp, "%d", x);
  return string(temp);
}


#include <iostream>
#include <utility>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

/**
Copyright (c) 2000-2010 Chih-Chung Chang and Chih-Jen Lin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither name of copyright holders nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define LIBSVM_VERSION 300

struct svm_node
{
	int index;
	double value;
};

struct svm_problem
{
	int l;
	double *y;
	struct svm_node **x;
};

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

struct svm_parameter
{
	int svm_type;
	int kernel_type;
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */

	/* these are for training only */
	double cache_size; /* in MB */
	double eps;	/* stopping criteria */
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	int nr_weight;		/* for C_SVC */
	int *weight_label;	/* for C_SVC */
	double* weight;		/* for C_SVC */
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	double p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */
	int probability; /* do probability estimates */
};

//
// svm_model
// 
struct svm_model
{
	struct svm_parameter param;	/* parameter */
	int nr_class;		/* number of classes, = 2 in regression/one class svm */
	int l;			/* total #SV */
	struct svm_node **SV;		/* SVs (SV[l]) */
	double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
	double *probA;		/* pariwise probability information */
	double *probB;

	/* for classification only */

	int *label;		/* label of each class (label[k]) */
	int *nSV;		/* number of SVs for each class (nSV[k]) */
				/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
	/* XXX */
	int free_sv;		/* 1 if svm_model is created by svm_load_model*/
				/* 0 if svm_model is created by svm_train */
};

struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

int svm_save_model(const char *model_file_name, const struct svm_model *model);
struct svm_model *svm_load_model(const char *model_file_name);

int svm_get_svm_type(const struct svm_model *model);
int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);
double svm_get_svr_probability(const struct svm_model *model);

double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
double svm_predict(const struct svm_model *model, const struct svm_node *x);
double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

void svm_free_model_content(struct svm_model *model_ptr);
void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
void svm_destroy_param(struct svm_parameter *param);

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
int svm_check_probability_model(const struct svm_model *model);

void svm_set_print_string_function(void (*print_func)(const char *));

// deprecated
// this function will be removed in future release
void svm_destroy_model(struct svm_model *model_ptr); 

///////////////////////////////
// END SVM HEADER
///////////////////////////////

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
int libsvm_version = LIBSVM_VERSION;
typedef float Qfloat;
typedef signed char schar;
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void (*svm_print_string) (const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*svm_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
public:
	Cache(int l,long int size);
	~Cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	int get_data(const int index, Qfloat **data, int len);
	void swap_index(int i, int j);	
private:
	int l;
	long int size;
	struct head_t
	{
		head_t *prev, *next;	// a circular list
		Qfloat *data;
		int len;		// data[0,len) is cached in this entry
	};

	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};

Cache::Cache(int l_,long int size_):l(l_),size(size_)
{
	head = (head_t *)calloc(l,sizeof(head_t));	// initialized to 0
	size /= sizeof(Qfloat);
	size -= l * sizeof(head_t) / sizeof(Qfloat);
	size = max(size, 2 * (long int) l);	// cache must be large enough for two columns
	lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache()
{
	for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
		free(h->data);
	free(head);
}

void Cache::lru_delete(head_t *h)
{
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h)
{
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len)
{
	head_t *h = &head[index];
	if(h->len) lru_delete(h);
	int more = len - h->len;

	if(more > 0)
	{
		// free old space
		while(size < more)
		{
			head_t *old = lru_head.next;
			lru_delete(old);
			free(old->data);
			size += old->len;
			old->data = 0;
			old->len = 0;
		}

		// allocate new space
		h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*len);
		size -= more;
		swap(h->len,len);
	}

	lru_insert(h);
	*data = h->data;
	return len;
}

void Cache::swap_index(int i, int j)
{
	if(i==j) return;

	if(head[i].len) lru_delete(&head[i]);
	if(head[j].len) lru_delete(&head[j]);
	swap(head[i].data,head[j].data);
	swap(head[i].len,head[j].len);
	if(head[i].len) lru_insert(&head[i]);
	if(head[j].len) lru_insert(&head[j]);

	if(i>j) swap(i,j);
	for(head_t *h = lru_head.next; h!=&lru_head; h=h->next)
	{
		if(h->len > i)
		{
			if(h->len > j)
				swap(h->data[i],h->data[j]);
			else
			{
				// give up
				lru_delete(h);
				free(h->data);
				size += h->len;
				h->data = 0;
				h->len = 0;
			}
		}
	}
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
public:
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual ~QMatrix() {}
};

class Kernel: public QMatrix {
public:
	Kernel(int l, svm_node * const * x, const svm_parameter& param);
	virtual ~Kernel();

	static double k_function(const svm_node *x, const svm_node *y,
				 const svm_parameter& param);
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const	// no so const...
	{
		swap(x[i],x[j]);
		if(x_square) swap(x_square[i],x_square[j]);
	}
protected:

	double (Kernel::*kernel_function)(int i, int j) const;

private:
	const svm_node **x;
	double *x_square;

	// svm_parameter
	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

	static double dot(const svm_node *px, const svm_node *py);
	double kernel_linear(int i, int j) const
	{
		return dot(x[i],x[j]);
	}
	double kernel_poly(int i, int j) const
	{
		return powi(gamma*dot(x[i],x[j])+coef0,degree);
	}
	double kernel_rbf(int i, int j) const
	{
		return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
	}
	double kernel_sigmoid(int i, int j) const
	{
		return tanh(gamma*dot(x[i],x[j])+coef0);
	}
	double kernel_precomputed(int i, int j) const
	{
		return x[i][(int)(x[j][0].value)].value;
	}
};

Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& param)
:kernel_type(param.kernel_type), degree(param.degree),
 gamma(param.gamma), coef0(param.coef0)
{
	switch(kernel_type)
	{
		case LINEAR:
			kernel_function = &Kernel::kernel_linear;
			break;
		case POLY:
			kernel_function = &Kernel::kernel_poly;
			break;
		case RBF:
			kernel_function = &Kernel::kernel_rbf;
			break;
		case SIGMOID:
			kernel_function = &Kernel::kernel_sigmoid;
			break;
		case PRECOMPUTED:
			kernel_function = &Kernel::kernel_precomputed;
			break;
	}

	clone(x,x_,l);

	if(kernel_type == RBF)
	{
		x_square = new double[l];
		for(int i=0;i<l;i++)
			x_square[i] = dot(x[i],x[i]);
	}
	else
		x_square = 0;
}

Kernel::~Kernel()
{
	delete[] x;
	delete[] x_square;
}

double Kernel::dot(const svm_node *px, const svm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}

double Kernel::k_function(const svm_node *x, const svm_node *y,
			  const svm_parameter& param)
{
	switch(param.kernel_type)
	{
		case LINEAR:
			return dot(x,y);
		case POLY:
			return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
		case RBF:
		{
			double sum = 0;
			while(x->index != -1 && y->index !=-1)
			{
				if(x->index == y->index)
				{
					double d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else
				{
					if(x->index > y->index)
					{	
						sum += y->value * y->value;
						++y;
					}
					else
					{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += y->value * y->value;
				++y;
			}
			
			return exp(-param.gamma*sum);
		}
		case SIGMOID:
			return tanh(param.gamma*dot(x,y)+param.coef0);
		case PRECOMPUTED:  //x: test (validation), y: SV
			return x[(int)(y->value)].value;
		default:
			return 0;  // Unreachable 
	}
}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
public:
	Solver() {};
	virtual ~Solver() {};

	struct SolutionInfo {
		double obj;
		double rho;
		double upper_bound_p;
		double upper_bound_n;
		double r;	// for Solver_NU
	};

	void Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking);
protected:
	int active_size;
	schar *y;
	double *G;		// gradient of objective function
	enum { LOWER_BOUND, UPPER_BOUND, FREE };
	char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	double *alpha;
	const QMatrix *Q;
	const double *QD;
	double eps;
	double Cp,Cn;
	double *p;
	int *active_set;
	double *G_bar;		// gradient, if we treat free variables as 0
	int l;
	bool unshrink;	// XXX

	double get_C(int i)
	{
		return (y[i] > 0)? Cp : Cn;
	}
	void update_alpha_status(int i)
	{
		if(alpha[i] >= get_C(i))
			alpha_status[i] = UPPER_BOUND;
		else if(alpha[i] <= 0)
			alpha_status[i] = LOWER_BOUND;
		else alpha_status[i] = FREE;
	}
	bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
	bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
	bool is_free(int i) { return alpha_status[i] == FREE; }
	void swap_index(int i, int j);
	void reconstruct_gradient();
	virtual int select_working_set(int &i, int &j);
	virtual double calculate_rho();
	virtual void do_shrinking();
private:
	bool be_shrunk(int i, double Gmax1, double Gmax2);	
};

void Solver::swap_index(int i, int j)
{
	Q->swap_index(i,j);
	swap(y[i],y[j]);
	swap(G[i],G[j]);
	swap(alpha_status[i],alpha_status[j]);
	swap(alpha[i],alpha[j]);
	swap(p[i],p[j]);
	swap(active_set[i],active_set[j]);
	swap(G_bar[i],G_bar[j]);
}

void Solver::reconstruct_gradient()
{
	// reconstruct inactive elements of G from G_bar and free variables

	if(active_size == l) return;

	int i,j;
	int nr_free = 0;

	for(j=active_size;j<l;j++)
		G[j] = G_bar[j] + p[j];

	for(j=0;j<active_size;j++)
		if(is_free(j))
			nr_free++;

	if(2*nr_free < active_size)
		info("\nWarning: using -h 0 may be faster\n");

	if (nr_free*l > 2*active_size*(l-active_size))
	{
		for(i=active_size;i<l;i++)
		{
			const Qfloat *Q_i = Q->get_Q(i,active_size);
			for(j=0;j<active_size;j++)
				if(is_free(j))
					G[i] += alpha[j] * Q_i[j];
		}
	}
	else
	{
		for(i=0;i<active_size;i++)
			if(is_free(i))
			{
				const Qfloat *Q_i = Q->get_Q(i,l);
				double alpha_i = alpha[i];
				for(j=active_size;j<l;j++)
					G[j] += alpha_i * Q_i[j];
			}
	}
}

void Solver::Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking)
{
	this->l = l;
	this->Q = &Q;
	QD=Q.get_QD();
	clone(p, p_,l);
	clone(y, y_,l);
	clone(alpha,alpha_,l);
	this->Cp = Cp;
	this->Cn = Cn;
	this->eps = eps;
	unshrink = false;

	// initialize alpha_status
	{
		alpha_status = new char[l];
		for(int i=0;i<l;i++)
			update_alpha_status(i);
	}

	// initialize active set (for shrinking)
	{
		active_set = new int[l];
		for(int i=0;i<l;i++)
			active_set[i] = i;
		active_size = l;
	}

	// initialize gradient
	{
		G = new double[l];
		G_bar = new double[l];
		int i;
		for(i=0;i<l;i++)
		{
			G[i] = p[i];
			G_bar[i] = 0;
		}
		for(i=0;i<l;i++)
			if(!is_lower_bound(i))
			{
				const Qfloat *Q_i = Q.get_Q(i,l);
				double alpha_i = alpha[i];
				int j;
				for(j=0;j<l;j++)
					G[j] += alpha_i*Q_i[j];
				if(is_upper_bound(i))
					for(j=0;j<l;j++)
						G_bar[j] += get_C(i) * Q_i[j];
			}
	}

	// optimization step

	int iter = 0;
	int counter = min(l,1000)+1;

	while(1)
	{
		// show progress and do shrinking

		if(--counter == 0)
		{
			counter = min(l,1000);
			if(shrinking) do_shrinking();
			info(".");
		}

		int i,j;
		if(select_working_set(i,j)!=0)
		{
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size = l;
			info("*");
			if(select_working_set(i,j)!=0)
				break;
			else
				counter = 1;	// do shrinking next iteration
		}
		
		++iter;

		// update alpha[i] and alpha[j], handle bounds carefully
		
		const Qfloat *Q_i = Q.get_Q(i,active_size);
		const Qfloat *Q_j = Q.get_Q(j,active_size);

		double C_i = get_C(i);
		double C_j = get_C(j);

		double old_alpha_i = alpha[i];
		double old_alpha_j = alpha[j];

		if(y[i]!=y[j])
		{
			double quad_coef = QD[i]+QD[j]+2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (-G[i]-G[j])/quad_coef;
			double diff = alpha[i] - alpha[j];
			alpha[i] += delta;
			alpha[j] += delta;
			
			if(diff > 0)
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = diff;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = -diff;
				}
			}
			if(diff > C_i - C_j)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = C_i - diff;
				}
			}
			else
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = C_j + diff;
				}
			}
		}
		else
		{
			double quad_coef = QD[i]+QD[j]-2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (G[i]-G[j])/quad_coef;
			double sum = alpha[i] + alpha[j];
			alpha[i] -= delta;
			alpha[j] += delta;

			if(sum > C_i)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = sum - C_i;
				}
			}
			else
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = sum;
				}
			}
			if(sum > C_j)
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = sum - C_j;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = sum;
				}
			}
		}

		// update G

		double delta_alpha_i = alpha[i] - old_alpha_i;
		double delta_alpha_j = alpha[j] - old_alpha_j;
		
		for(int k=0;k<active_size;k++)
		{
			G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
		}

		// update alpha_status and G_bar

		{
			bool ui = is_upper_bound(i);
			bool uj = is_upper_bound(j);
			update_alpha_status(i);
			update_alpha_status(j);
			int k;
			if(ui != is_upper_bound(i))
			{
				Q_i = Q.get_Q(i,l);
				if(ui)
					for(k=0;k<l;k++)
						G_bar[k] -= C_i * Q_i[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_i * Q_i[k];
			}

			if(uj != is_upper_bound(j))
			{
				Q_j = Q.get_Q(j,l);
				if(uj)
					for(k=0;k<l;k++)
						G_bar[k] -= C_j * Q_j[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_j * Q_j[k];
			}
		}
	}

	// calculate rho

	si->rho = calculate_rho();

	// calculate objective value
	{
		double v = 0;
		int i;
		for(i=0;i<l;i++)
			v += alpha[i] * (G[i] + p[i]);

		si->obj = v/2;
	}

	// put back the solution
	{
		for(int i=0;i<l;i++)
			alpha_[active_set[i]] = alpha[i];
	}

	// juggle everything back
	/*{
		for(int i=0;i<l;i++)
			while(active_set[i] != i)
				swap_index(i,active_set[i]);
				// or Q.swap_index(i,active_set[i]);
	}*/

	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	info("\noptimization finished, #iter = %d\n",iter);

	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	delete[] G_bar;
}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set(int &out_i, int &out_j)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
	
	double Gmax = -INF;
	double Gmax2 = -INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)	
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmax)
				{
					Gmax = -G[t];
					Gmax_idx = t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmax)
				{
					Gmax = G[t];
					Gmax_idx = t;
				}
		}

	int i = Gmax_idx;
	const Qfloat *Q_i = NULL;
	if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
		Q_i = Q->get_Q(i,active_size);

	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))
			{
				double grad_diff=Gmax+G[j];
				if (G[j] >= Gmax2)
					Gmax2 = G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff= Gmax-G[j];
				if (-G[j] >= Gmax2)
					Gmax2 = -G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if(Gmax+Gmax2 < eps)
		return 1;

	out_i = Gmax_idx;
	out_j = Gmin_idx;
	return 0;
}

bool Solver::be_shrunk(int i, double Gmax1, double Gmax2)
{
	if(is_upper_bound(i))
	{
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else
			return(-G[i] > Gmax2);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else	
			return(G[i] > Gmax1);
	}
	else
		return(false);
}

void Solver::do_shrinking()
{
	int i;
	double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
	double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

	// find maximal violating pair first
	for(i=0;i<active_size;i++)
	{
		if(y[i]==+1)	
		{
			if(!is_upper_bound(i))	
			{
				if(-G[i] >= Gmax1)
					Gmax1 = -G[i];
			}
			if(!is_lower_bound(i))	
			{
				if(G[i] >= Gmax2)
					Gmax2 = G[i];
			}
		}
		else	
		{
			if(!is_upper_bound(i))	
			{
				if(-G[i] >= Gmax2)
					Gmax2 = -G[i];
			}
			if(!is_lower_bound(i))	
			{
				if(G[i] >= Gmax1)
					Gmax1 = G[i];
			}
		}
	}

	if(unshrink == false && Gmax1 + Gmax2 <= eps*10) 
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
		info("*");
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}

double Solver::calculate_rho()
{
	double r;
	int nr_free = 0;
	double ub = INF, lb = -INF, sum_free = 0;
	for(int i=0;i<active_size;i++)
	{
		double yG = y[i]*G[i];

		if(is_upper_bound(i))
		{
			if(y[i]==-1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else if(is_lower_bound(i))
		{
			if(y[i]==+1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else
		{
			++nr_free;
			sum_free += yG;
		}
	}

	if(nr_free>0)
		r = sum_free/nr_free;
	else
		r = (ub+lb)/2;

	return r;
}

//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class Solver_NU : public Solver
{
public:
	Solver_NU() {}
	void Solve(int l, const QMatrix& Q, const double *p, const schar *y,
		   double *alpha, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking)
	{
		this->si = si;
		Solver::Solve(l,Q,p,y,alpha,Cp,Cn,eps,si,shrinking);
	}
private:
	SolutionInfo *si;
	int select_working_set(int &i, int &j);
	double calculate_rho();
	bool be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);
	void do_shrinking();
};

// return 1 if already optimal, return 0 otherwise
int Solver_NU::select_working_set(int &out_i, int &out_j)
{
	// return i,j such that y_i = y_j and
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmaxp = -INF;
	double Gmaxp2 = -INF;
	int Gmaxp_idx = -1;

	double Gmaxn = -INF;
	double Gmaxn2 = -INF;
	int Gmaxn_idx = -1;

	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmaxp)
				{
					Gmaxp = -G[t];
					Gmaxp_idx = t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmaxn)
				{
					Gmaxn = G[t];
					Gmaxn_idx = t;
				}
		}

	int ip = Gmaxp_idx;
	int in = Gmaxn_idx;
	const Qfloat *Q_ip = NULL;
	const Qfloat *Q_in = NULL;
	if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
		Q_ip = Q->get_Q(ip,active_size);
	if(in != -1)
		Q_in = Q->get_Q(in,active_size);

	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))	
			{
				double grad_diff=Gmaxp+G[j];
				if (G[j] >= Gmaxp2)
					Gmaxp2 = G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef = QD[ip]+QD[j]-2*Q_ip[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff=Gmaxn-G[j];
				if (-G[j] >= Gmaxn2)
					Gmaxn2 = -G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef = QD[in]+QD[j]-2*Q_in[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if(max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2) < eps)
		return 1;

	if (y[Gmin_idx] == +1)
		out_i = Gmaxp_idx;
	else
		out_i = Gmaxn_idx;
	out_j = Gmin_idx;

	return 0;
}

bool Solver_NU::be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4)
{
	if(is_upper_bound(i))
	{
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else	
			return(-G[i] > Gmax4);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else	
			return(G[i] > Gmax3);
	}
	else
		return(false);
}

void Solver_NU::do_shrinking()
{
	double Gmax1 = -INF;	// max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
	double Gmax2 = -INF;	// max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
	double Gmax3 = -INF;	// max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
	double Gmax4 = -INF;	// max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

	// find maximal violating pair first
	int i;
	for(i=0;i<active_size;i++)
	{
		if(!is_upper_bound(i))
		{
			if(y[i]==+1)
			{
				if(-G[i] > Gmax1) Gmax1 = -G[i];
			}
			else	if(-G[i] > Gmax4) Gmax4 = -G[i];
		}
		if(!is_lower_bound(i))
		{
			if(y[i]==+1)
			{	
				if(G[i] > Gmax2) Gmax2 = G[i];
			}
			else	if(G[i] > Gmax3) Gmax3 = G[i];
		}
	}

	if(unshrink == false && max(Gmax1+Gmax2,Gmax3+Gmax4) <= eps*10) 
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}

double Solver_NU::calculate_rho()
{
	int nr_free1 = 0,nr_free2 = 0;
	double ub1 = INF, ub2 = INF;
	double lb1 = -INF, lb2 = -INF;
	double sum_free1 = 0, sum_free2 = 0;

	for(int i=0;i<active_size;i++)
	{
		if(y[i]==+1)
		{
			if(is_upper_bound(i))
				lb1 = max(lb1,G[i]);
			else if(is_lower_bound(i))
				ub1 = min(ub1,G[i]);
			else
			{
				++nr_free1;
				sum_free1 += G[i];
			}
		}
		else
		{
			if(is_upper_bound(i))
				lb2 = max(lb2,G[i]);
			else if(is_lower_bound(i))
				ub2 = min(ub2,G[i]);
			else
			{
				++nr_free2;
				sum_free2 += G[i];
			}
		}
	}

	double r1,r2;
	if(nr_free1 > 0)
		r1 = sum_free1/nr_free1;
	else
		r1 = (ub1+lb1)/2;
	
	if(nr_free2 > 0)
		r2 = sum_free2/nr_free2;
	else
		r2 = (ub2+lb2)/2;
	
	si->r = (r1+r2)/2;
	return (r1-r2)/2;
}

//
// Q matrices for various formulations
//
class SVC_Q: public Kernel
{ 
public:
	SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
	:Kernel(prob.l, prob.x, param)
	{
		clone(y,y_,prob.l);
		cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
		QD = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i] = (this->*kernel_function)(i,i);
	}
	
	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int start, j;
		if((start = cache->get_data(i,&data,len)) < len)
		{
			for(j=start;j<len;j++)
				data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
		}
		return data;
	}

	double *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(y[i],y[j]);
		swap(QD[i],QD[j]);
	}

	~SVC_Q()
	{
		delete[] y;
		delete cache;
		delete[] QD;
	}
private:
	schar *y;
	Cache *cache;
	double *QD;
};

class ONE_CLASS_Q: public Kernel
{
public:
	ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param)
	{
		cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
		QD = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i] = (this->*kernel_function)(i,i);
	}
	
	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int start, j;
		if((start = cache->get_data(i,&data,len)) < len)
		{
			for(j=start;j<len;j++)
				data[j] = (Qfloat)(this->*kernel_function)(i,j);
		}
		return data;
	}

	double *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(QD[i],QD[j]);
	}

	~ONE_CLASS_Q()
	{
		delete cache;
		delete[] QD;
	}
private:
	Cache *cache;
	double *QD;
};

class SVR_Q: public Kernel
{ 
public:
	SVR_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param)
	{
		l = prob.l;
		cache = new Cache(l,(long int)(param.cache_size*(1<<20)));
		QD = new double[2*l];
		sign = new schar[2*l];
		index = new int[2*l];
		for(int k=0;k<l;k++)
		{
			sign[k] = 1;
			sign[k+l] = -1;
			index[k] = k;
			index[k+l] = k;
			QD[k] = (this->*kernel_function)(k,k);
			QD[k+l] = QD[k];
		}
		buffer[0] = new Qfloat[2*l];
		buffer[1] = new Qfloat[2*l];
		next_buffer = 0;
	}

	void swap_index(int i, int j) const
	{
		swap(sign[i],sign[j]);
		swap(index[i],index[j]);
		swap(QD[i],QD[j]);
	}
	
	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int j, real_i = index[i];
		if(cache->get_data(real_i,&data,l) < l)
		{
			for(j=0;j<l;j++)
				data[j] = (Qfloat)(this->*kernel_function)(real_i,j);
		}

		// reorder and copy
		Qfloat *buf = buffer[next_buffer];
		next_buffer = 1 - next_buffer;
		schar si = sign[i];
		for(j=0;j<len;j++)
			buf[j] = (Qfloat) si * (Qfloat) sign[j] * data[index[j]];
		return buf;
	}

	double *get_QD() const
	{
		return QD;
	}

	~SVR_Q()
	{
		delete cache;
		delete[] sign;
		delete[] index;
		delete[] buffer[0];
		delete[] buffer[1];
		delete[] QD;
	}
private:
	int l;
	Cache *cache;
	schar *sign;
	int *index;
	mutable int next_buffer;
	Qfloat *buffer[2];
	double *QD;
};

//
// construct and solve various formulations
//
static void solve_c_svc(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{
	int l = prob->l;
	double *minus_ones = new double[l];
	schar *y = new schar[l];

	int i;

	for(i=0;i<l;i++)
	{
		alpha[i] = 0;
		minus_ones[i] = -1;
		if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
	}

	Solver s;
	s.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
		alpha, Cp, Cn, param->eps, si, param->shrinking);

	double sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha += alpha[i];

	if (Cp==Cn)
		info("nu = %f\n", sum_alpha/(Cp*prob->l));

	for(i=0;i<l;i++)
		alpha[i] *= y[i];

	delete[] minus_ones;
	delete[] y;
}

static void solve_nu_svc(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int i;
	int l = prob->l;
	double nu = param->nu;

	schar *y = new schar[l];

	for(i=0;i<l;i++)
		if(prob->y[i]>0)
			y[i] = +1;
		else
			y[i] = -1;

	double sum_pos = nu*l/2;
	double sum_neg = nu*l/2;

	for(i=0;i<l;i++)
		if(y[i] == +1)
		{
			alpha[i] = min(1.0,sum_pos);
			sum_pos -= alpha[i];
		}
		else
		{
			alpha[i] = min(1.0,sum_neg);
			sum_neg -= alpha[i];
		}

	double *zeros = new double[l];

	for(i=0;i<l;i++)
		zeros[i] = 0;

	Solver_NU s;
	s.Solve(l, SVC_Q(*prob,*param,y), zeros, y,
		alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
	double r = si->r;

	info("C = %f\n",1/r);

	for(i=0;i<l;i++)
		alpha[i] *= y[i]/r;

	si->rho /= r;
	si->obj /= (r*r);
	si->upper_bound_p = 1/r;
	si->upper_bound_n = 1/r;

	delete[] y;
	delete[] zeros;
}

static void solve_one_class(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l = prob->l;
	double *zeros = new double[l];
	schar *ones = new schar[l];
	int i;

	int n = (int)(param->nu*prob->l);	// # of alpha's at upper bound

	for(i=0;i<n;i++)
		alpha[i] = 1;
	if(n<prob->l)
		alpha[n] = param->nu * prob->l - n;
	for(i=n+1;i<l;i++)
		alpha[i] = 0;

	for(i=0;i<l;i++)
	{
		zeros[i] = 0;
		ones[i] = 1;
	}

	Solver s;
	s.Solve(l, ONE_CLASS_Q(*prob,*param), zeros, ones,
		alpha, 1.0, 1.0, param->eps, si, param->shrinking);

	delete[] zeros;
	delete[] ones;
}

static void solve_epsilon_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l = prob->l;
	double *alpha2 = new double[2*l];
	double *linear_term = new double[2*l];
	schar *y = new schar[2*l];
	int i;

	for(i=0;i<l;i++)
	{
		alpha2[i] = 0;
		linear_term[i] = param->p - prob->y[i];
		y[i] = 1;

		alpha2[i+l] = 0;
		linear_term[i+l] = param->p + prob->y[i];
		y[i+l] = -1;
	}

	Solver s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, param->C, param->C, param->eps, si, param->shrinking);

	double sum_alpha = 0;
	for(i=0;i<l;i++)
	{
		alpha[i] = alpha2[i] - alpha2[i+l];
		sum_alpha += fabs(alpha[i]);
	}
	info("nu = %f\n",sum_alpha/(param->C*l));

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

static void solve_nu_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l = prob->l;
	double C = param->C;
	double *alpha2 = new double[2*l];
	double *linear_term = new double[2*l];
	schar *y = new schar[2*l];
	int i;

	double sum = C * param->nu * l / 2;
	for(i=0;i<l;i++)
	{
		alpha2[i] = alpha2[i+l] = min(sum,C);
		sum -= alpha2[i];

		linear_term[i] = - prob->y[i];
		y[i] = 1;

		linear_term[i+l] = prob->y[i];
		y[i+l] = -1;
	}

	Solver_NU s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, C, C, param->eps, si, param->shrinking);

	info("epsilon = %f\n",-si->r);

	for(i=0;i<l;i++)
		alpha[i] = alpha2[i] - alpha2[i+l];

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

//
// decision_function
//
struct decision_function
{
	double *alpha;
	double rho;	
};

static decision_function svm_train_one(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn)
{
	double *alpha = Malloc(double,prob->l);
	Solver::SolutionInfo si;
	switch(param->svm_type)
	{
		case C_SVC:
			solve_c_svc(prob,param,alpha,&si,Cp,Cn);
			break;
		case NU_SVC:
			solve_nu_svc(prob,param,alpha,&si);
			break;
		case ONE_CLASS:
			solve_one_class(prob,param,alpha,&si);
			break;
		case EPSILON_SVR:
			solve_epsilon_svr(prob,param,alpha,&si);
			break;
		case NU_SVR:
			solve_nu_svr(prob,param,alpha,&si);
			break;
	}

	info("obj = %f, rho = %f\n",si.obj,si.rho);

	// output SVs

	int nSV = 0;
	int nBSV = 0;
	for(int i=0;i<prob->l;i++)
	{
		if(fabs(alpha[i]) > 0)
		{
			++nSV;
			if(prob->y[i] > 0)
			{
				if(fabs(alpha[i]) >= si.upper_bound_p)
					++nBSV;
			}
			else
			{
				if(fabs(alpha[i]) >= si.upper_bound_n)
					++nBSV;
			}
		}
	}

	info("nSV = %d, nBSV = %d\n",nSV,nBSV);

	decision_function f;
	f.alpha = alpha;
	f.rho = si.rho;
	return f;
}

// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
static void sigmoid_train(
	int l, const double *dec_values, const double *labels, 
	double& A, double& B)
{
	double prior1=0, prior0 = 0;
	int i;

	for (i=0;i<l;i++)
		if (labels[i] > 0) prior1+=1;
		else prior0+=1;
	
	int max_iter=100;	// Maximal number of iterations
	double min_step=1e-10;	// Minimal step taken in line search
	double sigma=1e-12;	// For numerically strict PD of Hessian
	double eps=1e-5;
	double hiTarget=(prior1+1.0)/(prior1+2.0);
	double loTarget=1/(prior0+2.0);
	double *t=Malloc(double,l);
	double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
	double newA,newB,newf,d1,d2;
	int iter; 
	
	// Initial Point and Initial Fun Value
	A=0.0; B=log((prior0+1.0)/(prior1+1.0));
	double fval = 0.0;

	for (i=0;i<l;i++)
	{
		if (labels[i]>0) t[i]=hiTarget;
		else t[i]=loTarget;
		fApB = dec_values[i]*A+B;
		if (fApB>=0)
			fval += t[i]*fApB + log(1+exp(-fApB));
		else
			fval += (t[i] - 1)*fApB +log(1+exp(fApB));
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// Update Gradient and Hessian (use H' = H + sigma I)
		h11=sigma; // numerically ensures strict PD
		h22=sigma;
		h21=0.0;g1=0.0;g2=0.0;
		for (i=0;i<l;i++)
		{
			fApB = dec_values[i]*A+B;
			if (fApB >= 0)
			{
				p=exp(-fApB)/(1.0+exp(-fApB));
				q=1.0/(1.0+exp(-fApB));
			}
			else
			{
				p=1.0/(1.0+exp(fApB));
				q=exp(fApB)/(1.0+exp(fApB));
			}
			d2=p*q;
			h11+=dec_values[i]*dec_values[i]*d2;
			h22+=d2;
			h21+=dec_values[i]*d2;
			d1=t[i]-p;
			g1+=dec_values[i]*d1;
			g2+=d1;
		}

		// Stopping Criteria
		if (fabs(g1)<eps && fabs(g2)<eps)
			break;

		// Finding Newton direction: -inv(H') * g
		det=h11*h22-h21*h21;
		dA=-(h22*g1 - h21 * g2) / det;
		dB=-(-h21*g1+ h11 * g2) / det;
		gd=g1*dA+g2*dB;


		stepsize = 1;		// Line Search
		while (stepsize >= min_step)
		{
			newA = A + stepsize * dA;
			newB = B + stepsize * dB;

			// New function value
			newf = 0.0;
			for (i=0;i<l;i++)
			{
				fApB = dec_values[i]*newA+newB;
				if (fApB >= 0)
					newf += t[i]*fApB + log(1+exp(-fApB));
				else
					newf += (t[i] - 1)*fApB +log(1+exp(fApB));
			}
			// Check sufficient decrease
			if (newf<fval+0.0001*stepsize*gd)
			{
				A=newA;B=newB;fval=newf;
				break;
			}
			else
				stepsize = stepsize / 2.0;
		}

		if (stepsize < min_step)
		{
			info("Line search fails in two-class probability estimates\n");
			break;
		}
	}

	if (iter>=max_iter)
		info("Reaching maximal iterations in two-class probability estimates\n");
	free(t);
}

static double sigmoid_predict(double decision_value, double A, double B)
{
	double fApB = decision_value*A+B;
	if (fApB >= 0)
		return exp(-fApB)/(1.0+exp(-fApB));
	else
		return 1.0/(1+exp(fApB)) ;
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
static void multiclass_probability(int k, double **r, double *p)
{
	int t,j;
	int iter = 0, max_iter=max(100,k);
	double **Q=Malloc(double *,k);
	double *Qp=Malloc(double,k);
	double pQp, eps=0.005/k;
	
	for (t=0;t<k;t++)
	{
		p[t]=1.0/k;  // Valid if k = 1
		Q[t]=Malloc(double,k);
		Q[t][t]=0;
		for (j=0;j<t;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=Q[j][t];
		}
		for (j=t+1;j<k;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=-r[j][t]*r[t][j];
		}
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp=0;
		for (t=0;t<k;t++)
		{
			Qp[t]=0;
			for (j=0;j<k;j++)
				Qp[t]+=Q[t][j]*p[j];
			pQp+=p[t]*Qp[t];
		}
		double max_error=0;
		for (t=0;t<k;t++)
		{
			double error=fabs(Qp[t]-pQp);
			if (error>max_error)
				max_error=error;
		}
		if (max_error<eps) break;
		
		for (t=0;t<k;t++)
		{
			double diff=(-Qp[t]+pQp)/Q[t][t];
			p[t]+=diff;
			pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
			for (j=0;j<k;j++)
			{
				Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
				p[j]/=(1+diff);
			}
		}
	}
	if (iter>=max_iter)
		info("Exceeds max_iter in multiclass_prob\n");
	for(t=0;t<k;t++) free(Q[t]);
	free(Q);
	free(Qp);
}

// Cross-validation decision values for probability estimates
static void svm_binary_svc_probability(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn, double& probA, double& probB)
{
	int i;
	int nr_fold = 5;
	int *perm = Malloc(int,prob->l);
	double *dec_values = Malloc(double,prob->l);

	// random shuffle
	for(i=0;i<prob->l;i++) perm[i]=i;
	for(i=0;i<prob->l;i++)
	{
		int j = i+rand()%(prob->l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<nr_fold;i++)
	{
		int begin = i*prob->l/nr_fold;
		int end = (i+1)*prob->l/nr_fold;
		int j,k;
		struct svm_problem subprob;

		subprob.l = prob->l-(end-begin);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);
			
		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<prob->l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		int p_count=0,n_count=0;
		for(j=0;j<k;j++)
			if(subprob.y[j]>0)
				p_count++;
			else
				n_count++;

		if(p_count==0 && n_count==0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = 0;
		else if(p_count > 0 && n_count == 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = 1;
		else if(p_count == 0 && n_count > 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = -1;
		else
		{
			svm_parameter subparam = *param;
			subparam.probability=0;
			subparam.C=1.0;
			subparam.nr_weight=2;
			subparam.weight_label = Malloc(int,2);
			subparam.weight = Malloc(double,2);
			subparam.weight_label[0]=+1;
			subparam.weight_label[1]=-1;
			subparam.weight[0]=Cp;
			subparam.weight[1]=Cn;
			struct svm_model *submodel = svm_train(&subprob,&subparam);
			for(j=begin;j<end;j++)
			{
				svm_predict_values(submodel,prob->x[perm[j]],&(dec_values[perm[j]])); 
				// ensure +1 -1 order; reason not using CV subroutine
				dec_values[perm[j]] *= submodel->label[0];
			}		
			svm_free_and_destroy_model(&submodel);
			svm_destroy_param(&subparam);
		}
		free(subprob.x);
		free(subprob.y);
	}		
	sigmoid_train(prob->l,dec_values,prob->y,probA,probB);
	free(dec_values);
	free(perm);
}

// Return parameter of a Laplace distribution 
static double svm_svr_probability(
	const svm_problem *prob, const svm_parameter *param)
{
	int i;
	int nr_fold = 5;
	double *ymv = Malloc(double,prob->l);
	double mae = 0;

	svm_parameter newparam = *param;
	newparam.probability = 0;
	svm_cross_validation(prob,&newparam,nr_fold,ymv);
	for(i=0;i<prob->l;i++)
	{
		ymv[i]=prob->y[i]-ymv[i];
		mae += fabs(ymv[i]);
	}		
	mae /= prob->l;
	double std=sqrt(2*mae*mae);
	int count=0;
	mae=0;
	for(i=0;i<prob->l;i++)
		if (fabs(ymv[i]) > 5*std) 
			count=count+1;
		else 
			mae+=fabs(ymv[i]);
	mae /= (prob->l-count);
	info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n",mae);
	free(ymv);
	return mae;
}


// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);	
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

//
// Interface functions
//
svm_model *svm_train(const svm_problem *prob, const svm_parameter *param)
{
	svm_model *model = Malloc(svm_model,1);
	model->param = *param;
	model->free_sv = 0;	// XXX

	if(param->svm_type == ONE_CLASS ||
	   param->svm_type == EPSILON_SVR ||
	   param->svm_type == NU_SVR)
	{
		// regression or one-class-svm
		model->nr_class = 2;
		model->label = NULL;
		model->nSV = NULL;
		model->probA = NULL; model->probB = NULL;
		model->sv_coef = Malloc(double *,1);

		if(param->probability && 
		   (param->svm_type == EPSILON_SVR ||
		    param->svm_type == NU_SVR))
		{
			model->probA = Malloc(double,1);
			model->probA[0] = svm_svr_probability(prob,param);
		}

		decision_function f = svm_train_one(prob,param,0,0);
		model->rho = Malloc(double,1);
		model->rho[0] = f.rho;

		int nSV = 0;
		int i;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0) ++nSV;
		model->l = nSV;
		model->SV = Malloc(svm_node *,nSV);
		model->sv_coef[0] = Malloc(double,nSV);
		int j = 0;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0)
			{
				model->SV[j] = prob->x[i];
				model->sv_coef[0][j] = f.alpha[i];
				++j;
			}		

		free(f.alpha);
	}
	else
	{
		// classification
		int l = prob->l;
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);		
		svm_node **x = Malloc(svm_node *,l);
		int i;
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]];

		// calculate weighted C

		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{	
			int j;
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				fprintf(stderr,"warning: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// train k*(k-1)/2 models
		
		bool *nonzero = Malloc(bool,l);
		for(i=0;i<l;i++)
			nonzero[i] = false;
		decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

		double *probA=NULL,*probB=NULL;
		if (param->probability)
		{
			probA=Malloc(double,nr_class*(nr_class-1)/2);
			probB=Malloc(double,nr_class*(nr_class-1)/2);
		}

		int p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				svm_problem sub_prob;
				int si = start[i], sj = start[j];
				int ci = count[i], cj = count[j];
				sub_prob.l = ci+cj;
				sub_prob.x = Malloc(svm_node *,sub_prob.l);
				sub_prob.y = Malloc(double,sub_prob.l);
				int k;
				for(k=0;k<ci;k++)
				{
					sub_prob.x[k] = x[si+k];
					sub_prob.y[k] = +1;
				}
				for(k=0;k<cj;k++)
				{
					sub_prob.x[ci+k] = x[sj+k];
					sub_prob.y[ci+k] = -1;
				}

				if(param->probability)
					svm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],probA[p],probB[p]);

				f[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
				for(k=0;k<ci;k++)
					if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
						nonzero[si+k] = true;
				for(k=0;k<cj;k++)
					if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
						nonzero[sj+k] = true;
				free(sub_prob.x);
				free(sub_prob.y);
				++p;
			}

		// build output

		model->nr_class = nr_class;
		
		model->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model->label[i] = label[i];
		
		model->rho = Malloc(double,nr_class*(nr_class-1)/2);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			model->rho[i] = f[i].rho;

		if(param->probability)
		{
			model->probA = Malloc(double,nr_class*(nr_class-1)/2);
			model->probB = Malloc(double,nr_class*(nr_class-1)/2);
			for(i=0;i<nr_class*(nr_class-1)/2;i++)
			{
				model->probA[i] = probA[i];
				model->probB[i] = probB[i];
			}
		}
		else
		{
			model->probA=NULL;
			model->probB=NULL;
		}

		int total_sv = 0;
		int *nz_count = Malloc(int,nr_class);
		model->nSV = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
		{
			int nSV = 0;
			for(int j=0;j<count[i];j++)
				if(nonzero[start[i]+j])
				{	
					++nSV;
					++total_sv;
				}
			model->nSV[i] = nSV;
			nz_count[i] = nSV;
		}
		
		info("Total nSV = %d\n",total_sv);

		model->l = total_sv;
		model->SV = Malloc(svm_node *,total_sv);
		p = 0;
		for(i=0;i<l;i++)
			if(nonzero[i]) model->SV[p++] = x[i];

		int *nz_start = Malloc(int,nr_class);
		nz_start[0] = 0;
		for(i=1;i<nr_class;i++)
			nz_start[i] = nz_start[i-1]+nz_count[i-1];

		model->sv_coef = Malloc(double *,nr_class-1);
		for(i=0;i<nr_class-1;i++)
			model->sv_coef[i] = Malloc(double,total_sv);

		p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int si = start[i];
				int sj = start[j];
				int ci = count[i];
				int cj = count[j];
				
				int q = nz_start[i];
				int k;
				for(k=0;k<ci;k++)
					if(nonzero[si+k])
						model->sv_coef[j-1][q++] = f[p].alpha[k];
				q = nz_start[j];
				for(k=0;k<cj;k++)
					if(nonzero[sj+k])
						model->sv_coef[i][q++] = f[p].alpha[ci+k];
				++p;
			}
		
		free(label);
		free(probA);
		free(probB);
		free(count);
		free(perm);
		free(start);
		free(x);
		free(weighted_C);
		free(nonzero);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			free(f[i].alpha);
		free(f);
		free(nz_count);
		free(nz_start);
	}
	return model;
}

// Stratified cross validation
void svm_cross_validation(const svm_problem *prob, const svm_parameter *param, int nr_fold, double *target)
{
	int i;
	int *fold_start = Malloc(int,nr_fold+1);
	int l = prob->l;
	int *perm = Malloc(int,l);
	int nr_class;

	// stratified cv may not give leave-one-out rate
	// Each class to l folds -> some folds may have zero elements
	if((param->svm_type == C_SVC ||
	    param->svm_type == NU_SVC) && nr_fold < l)
	{
		int *start = NULL;
		int *label = NULL;
		int *count = NULL;
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

		// random shuffle and then data grouped by fold using the array perm
		int *fold_count = Malloc(int,nr_fold);
		int c;
		int *index = Malloc(int,l);
		for(i=0;i<l;i++)
			index[i]=perm[i];
		for (c=0; c<nr_class; c++) 
			for(i=0;i<count[c];i++)
			{
				int j = i+rand()%(count[c]-i);
				swap(index[start[c]+j],index[start[c]+i]);
			}
		for(i=0;i<nr_fold;i++)
		{
			fold_count[i] = 0;
			for (c=0; c<nr_class;c++)
				fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
		}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		for (c=0; c<nr_class;c++)
			for(i=0;i<nr_fold;i++)
			{
				int begin = start[c]+i*count[c]/nr_fold;
				int end = start[c]+(i+1)*count[c]/nr_fold;
				for(int j=begin;j<end;j++)
				{
					perm[fold_start[i]] = index[j];
					fold_start[i]++;
				}
			}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		free(start);	
		free(label);
		free(count);	
		free(index);
		free(fold_count);
	}
	else
	{
		for(i=0;i<l;i++) perm[i]=i;
		for(i=0;i<l;i++)
		{
			int j = i+rand()%(l-i);
			swap(perm[i],perm[j]);
		}
		for(i=0;i<=nr_fold;i++)
			fold_start[i]=i*l/nr_fold;
	}

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct svm_problem subprob;

		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);
			
		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct svm_model *submodel = svm_train(&subprob,param);
		if(param->probability && 
		   (param->svm_type == C_SVC || param->svm_type == NU_SVC))
		{
			double *prob_estimates=Malloc(double,svm_get_nr_class(submodel));
			for(j=begin;j<end;j++)
				target[perm[j]] = svm_predict_probability(submodel,prob->x[perm[j]],prob_estimates);
			free(prob_estimates);			
		}
		else
			for(j=begin;j<end;j++)
				target[perm[j]] = svm_predict(submodel,prob->x[perm[j]]);
		svm_free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}		
	free(fold_start);
	free(perm);	
}


int svm_get_svm_type(const svm_model *model)
{
	return model->param.svm_type;
}

int svm_get_nr_class(const svm_model *model)
{
	return model->nr_class;
}

void svm_get_labels(const svm_model *model, int* label)
{
	if (model->label != NULL)
		for(int i=0;i<model->nr_class;i++)
			label[i] = model->label[i];
}

double svm_get_svr_probability(const svm_model *model)
{
	if ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
	    model->probA!=NULL)
		return model->probA[0];
	else
	{
		fprintf(stderr,"Model doesn't contain information for SVR probability inference\n");
		return 0;
	}
}

double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
{
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
	{
		double *sv_coef = model->sv_coef[0];
		double sum = 0;
		for(int i=0;i<model->l;i++)
			sum += sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param);
		sum -= model->rho[0];
		*dec_values = sum;

		if(model->param.svm_type == ONE_CLASS)
			return (sum>0)?1:-1;
		else
			return sum;
	}
	else
	{
		int i;
		int nr_class = model->nr_class;
		int l = model->l;
		
		double *kvalue = Malloc(double,l);
		for(i=0;i<l;i++)
			kvalue[i] = Kernel::k_function(x,model->SV[i],model->param);

		int *start = Malloc(int,nr_class);
		start[0] = 0;
		for(i=1;i<nr_class;i++)
			start[i] = start[i-1]+model->nSV[i-1];

		int *vote = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			vote[i] = 0;

		int p=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				double sum = 0;
				int si = start[i];
				int sj = start[j];
				int ci = model->nSV[i];
				int cj = model->nSV[j];
				
				int k;
				double *coef1 = model->sv_coef[j-1];
				double *coef2 = model->sv_coef[i];
				for(k=0;k<ci;k++)
					sum += coef1[si+k] * kvalue[si+k];
				for(k=0;k<cj;k++)
					sum += coef2[sj+k] * kvalue[sj+k];
				sum -= model->rho[p];
				dec_values[p] = sum;

				if(dec_values[p] > 0)
					++vote[i];
				else
					++vote[j];
				p++;
			}

		int vote_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(vote[i] > vote[vote_max_idx])
				vote_max_idx = i;

		free(kvalue);
		free(start);
		free(vote);
		return model->label[vote_max_idx];
	}
}

double svm_predict(const svm_model *model, const svm_node *x)
{
	int nr_class = model->nr_class;
	double *dec_values;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
		dec_values = Malloc(double, 1);
	else 
		dec_values = Malloc(double, nr_class*(nr_class-1)/2);
	double pred_result = svm_predict_values(model, x, dec_values);
	free(dec_values);
	return pred_result;
}

double svm_predict_probability(
	const svm_model *model, const svm_node *x, double *prob_estimates)
{
	if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
	    model->probA!=NULL && model->probB!=NULL)
	{
		int i;
		int nr_class = model->nr_class;
		double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
		svm_predict_values(model, x, dec_values);

		double min_prob=1e-7;
		double **pairwise_prob=Malloc(double *,nr_class);
		for(i=0;i<nr_class;i++)
			pairwise_prob[i]=Malloc(double,nr_class);
		int k=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				pairwise_prob[i][j]=min(max(sigmoid_predict(dec_values[k],model->probA[k],model->probB[k]),min_prob),1-min_prob);
				pairwise_prob[j][i]=1-pairwise_prob[i][j];
				k++;
			}
		multiclass_probability(nr_class,pairwise_prob,prob_estimates);

		int prob_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(prob_estimates[i] > prob_estimates[prob_max_idx])
				prob_max_idx = i;
		for(i=0;i<nr_class;i++)
			free(pairwise_prob[i]);
		free(dec_values);
		free(pairwise_prob);	     
		return model->label[prob_max_idx];
	}
	else 
		return svm_predict(model, x);
}

static const char *svm_type_table[] =
{
	"c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

static const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

int svm_save_model(const char *model_file_name, const svm_model *model)
{
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	const svm_parameter& param = model->param;

	fprintf(fp,"svm_type %s\n", svm_type_table[param.svm_type]);
	fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);

	if(param.kernel_type == POLY)
		fprintf(fp,"degree %d\n", param.degree);

	if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
		fprintf(fp,"gamma %g\n", param.gamma);

	if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
		fprintf(fp,"coef0 %g\n", param.coef0);

	int nr_class = model->nr_class;
	int l = model->l;
	fprintf(fp, "nr_class %d\n", nr_class);
	fprintf(fp, "total_sv %d\n",l);
	
	{
		fprintf(fp, "rho");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->rho[i]);
		fprintf(fp, "\n");
	}
	
	if(model->label)
	{
		fprintf(fp, "label");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->label[i]);
		fprintf(fp, "\n");
	}

	if(model->probA) // regression has probA only
	{
		fprintf(fp, "probA");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->probA[i]);
		fprintf(fp, "\n");
	}
	if(model->probB)
	{
		fprintf(fp, "probB");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->probB[i]);
		fprintf(fp, "\n");
	}

	if(model->nSV)
	{
		fprintf(fp, "nr_sv");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->nSV[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "SV\n");
	const double * const *sv_coef = model->sv_coef;
	const svm_node * const *SV = model->SV;

	for(int i=0;i<l;i++)
	{
		for(int j=0;j<nr_class-1;j++)
			fprintf(fp, "%.16g ",sv_coef[j][i]);

		const svm_node *p = SV[i];

		if(param.kernel_type == PRECOMPUTED)
			fprintf(fp,"0:%d ",(int)(p->value));
		else
			while(p->index != -1)
			{
				fprintf(fp,"%d:%.8g ",p->index,p->value);
				p++;
			}
		fprintf(fp, "\n");
	}
	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}


class Memscanf
{
public:
	Memscanf(const char * data)
	{
		modeldata = data;
		offset = 0;
		len = strlen(data);
	}

	const char * modeldata;
	int offset;
	int len;

	char temp[300];

	int getLen(int x)
	{
		sprintf(temp, "%d", x);
		return strlen(temp);
	}

	void skipEmptyChar()
	{
		while (modeldata[offset] == ' ' || modeldata[offset] == '\n' || modeldata[offset] == '\t') 
			offset++;
	}

	void skipToEmptyChar()
	{
		while (modeldata[offset] != ' ' && modeldata[offset] != '\n' && modeldata[offset] != '\t')
			offset++;
	}

	void skipToClosed()
	{
		while (modeldata[offset] != ')') 
			offset++;
		offset++;
	}

	void mem_scanf_int(int * r)
	{
		skipEmptyChar();
		sscanf(modeldata + offset, "%d", r);
		skipToEmptyChar();
	}

	void mem_scanf_uint(unsigned int * p)
	{
		skipEmptyChar();
		sscanf(modeldata + offset, "%u", p);
		skipToEmptyChar();
	}

	void mem_scanf_float(float * p)
	{ 
		skipEmptyChar();
		sscanf(modeldata + offset, "%f", p);
		skipToEmptyChar();
	}

	void mem_scanf_double(double *p)
	{ 
		skipEmptyChar();
		sscanf(modeldata + offset, "%lf", p);
		skipToEmptyChar();
	}

	bool mem_readline(char * buffer)
	{
		if (offset >= len) return false;
		skipEmptyChar();
		int nowOff = 0;
		while (modeldata[offset] != '\n' && modeldata[offset] != 0) 
			buffer[nowOff++] = modeldata[offset++];
		buffer[nowOff] = 0;
		return true;
	}

	void mem_scanf(char * format, void * p)
	{
		skipEmptyChar();
		sscanf(modeldata + offset, format, p);
		skipToEmptyChar();
	}

	void mem_scanfTriple(unsigned int * a, unsigned int * b, float * c)
	{
		skipEmptyChar();
		sscanf(modeldata + offset, "(%u, %u, %f)", a, b, c);
		skipToClosed();
	}

	void mem_scanfPair(unsigned int * a, float * b)
	{ 
		skipEmptyChar();
		sscanf(modeldata + offset, "(%u, %f)", a, b);
		skipToClosed();
	}

	void mem_skipInt(int count)
	{
		offset += count;
	}

	void mem_skipStr(char * str)
	{
		skipEmptyChar();
		offset += strlen(str);
	}

	void mem_printf()
	{
		int i;
		for (i = offset; i < offset + 10; ++i)
			printf("%c", modeldata[i]);
		printf("\n");
	}
};

svm_model * svm_load_model_char(const char * model_data)
{
	Memscanf ms(model_data);

	svm_model *model = Malloc(svm_model,1);
	svm_parameter& param = model->param;
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->label = NULL;
	model->nSV = NULL;

	char cmd[81];
	while(1)
	{
		ms.mem_scanf("%80s",cmd);

		if(strcmp(cmd,"svm_type")==0)
		{
			ms.mem_scanf("%80s",cmd);
			int i;
			for(i=0;svm_type_table[i];i++)
			{
				if(strcmp(svm_type_table[i],cmd)==0)
				{
					param.svm_type=i;
					break;
				}
			}
			if(svm_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown svm type.\n");
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		}
		else if(strcmp(cmd,"kernel_type")==0)
		{		
			ms.mem_scanf("%80s",cmd);
			int i;
			for(i=0;kernel_type_table[i];i++)
			{
				if(strcmp(kernel_type_table[i],cmd)==0)
				{
					param.kernel_type=i;
					break;
				}
			}
			if(kernel_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown kernel function.\n");
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		}
		else if(strcmp(cmd,"degree")==0)
			ms.mem_scanf_int(&param.degree);
		else if(strcmp(cmd,"gamma")==0)
			ms.mem_scanf_double(&param.gamma);
		else if(strcmp(cmd,"coef0")==0)
			ms.mem_scanf_double(&param.coef0);
		else if(strcmp(cmd,"nr_class")==0)
			ms.mem_scanf_int(&model->nr_class);
		else if(strcmp(cmd,"total_sv")==0)
			ms.mem_scanf_int(&model->l);
		else if(strcmp(cmd,"rho")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->rho = Malloc(double,n);
			for(int i=0;i<n;i++)
				ms.mem_scanf_double(&model->rho[i]);
		}
		else if(strcmp(cmd,"label")==0)
		{
			int n = model->nr_class;
			model->label = Malloc(int,n);
			for(int i=0;i<n;i++)
				ms.mem_scanf_int(&model->label[i]);
		}
		else if(strcmp(cmd,"probA")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probA = Malloc(double,n);
			for(int i=0;i<n;i++)
				ms.mem_scanf_double(&model->probA[i]);
		}
		else if(strcmp(cmd,"probB")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probB = Malloc(double,n);
			for(int i=0;i<n;i++)
				ms.mem_scanf_double(&model->probB[i]);
		}
		else if(strcmp(cmd,"nr_sv")==0)
		{
			int n = model->nr_class;
			model->nSV = Malloc(int,n);
			for(int i=0;i<n;i++)
				ms.mem_scanf_int(&model->nSV[i]);
		}
		else if(strcmp(cmd,"SV")==0)
		{
			/**
			while(1)
			{
				int c = getc(fp);
				if(c==EOF || c=='\n') break;	
			}
			*/
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			free(model->rho);
			free(model->label);
			free(model->nSV);
			free(model);
			return NULL;
		}
	}

	// read sv_coef and SV

	int elements = 0;
	long pos = ms.offset;

	max_line_len = 1024;
	char line[1024];
	char *p,*endptr,*idx,*val;

	while(ms.mem_readline(line))
	{
		if (strlen(line) == 0) break;
		p = strtok(line,":");
		while(1)
		{
			p = strtok(NULL,":");
			if(p == NULL)
				break;
			++elements;
		}
	}
	elements += model->l;

	ms.offset = pos;

	int m = model->nr_class - 1;
	int l = model->l;
	model->sv_coef = Malloc(double *,m);
	int i;
	for(i=0;i<m;i++)
		model->sv_coef[i] = Malloc(double,l);
	model->SV = Malloc(svm_node*,l);
	svm_node *x_space = NULL;
	if(l>0) x_space = Malloc(svm_node,elements);

	int j=0;
	for(i=0;i<l;i++)
	{
		ms.mem_readline(line);
		model->SV[i] = &x_space[j];

		p = strtok(line, " \t");
		model->sv_coef[0][i] = strtod(p,&endptr);
		for(int k=1;k<m;k++)
		{
			p = strtok(NULL, " \t");
			model->sv_coef[k][i] = strtod(p,&endptr);
		}

		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
				break;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			x_space[j].value = strtod(val,&endptr);

			++j;
		}
		x_space[j++].index = -1;
	}
	//free(line);

	model->free_sv = 1;	// XXX
	return model;

}

svm_model *svm_load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"rb");
	if(fp==NULL) return NULL;
	
	// read parameters

	svm_model *model = Malloc(svm_model,1);
	svm_parameter& param = model->param;
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->label = NULL;
	model->nSV = NULL;

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);

		if(strcmp(cmd,"svm_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;svm_type_table[i];i++)
			{
				if(strcmp(svm_type_table[i],cmd)==0)
				{
					param.svm_type=i;
					break;
				}
			}
			if(svm_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown svm type.\n");
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		}
		else if(strcmp(cmd,"kernel_type")==0)
		{		
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;kernel_type_table[i];i++)
			{
				if(strcmp(kernel_type_table[i],cmd)==0)
				{
					param.kernel_type=i;
					break;
				}
			}
			if(kernel_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown kernel function.\n");
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		}
		else if(strcmp(cmd,"degree")==0)
			fscanf(fp,"%d",&param.degree);
		else if(strcmp(cmd,"gamma")==0)
			fscanf(fp,"%lf",&param.gamma);
		else if(strcmp(cmd,"coef0")==0)
			fscanf(fp,"%lf",&param.coef0);
		else if(strcmp(cmd,"nr_class")==0)
			fscanf(fp,"%d",&model->nr_class);
		else if(strcmp(cmd,"total_sv")==0)
			fscanf(fp,"%d",&model->l);
		else if(strcmp(cmd,"rho")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->rho = Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->rho[i]);
		}
		else if(strcmp(cmd,"label")==0)
		{
			int n = model->nr_class;
			model->label = Malloc(int,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%d",&model->label[i]);
		}
		else if(strcmp(cmd,"probA")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probA = Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->probA[i]);
		}
		else if(strcmp(cmd,"probB")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probB = Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->probB[i]);
		}
		else if(strcmp(cmd,"nr_sv")==0)
		{
			int n = model->nr_class;
			model->nSV = Malloc(int,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%d",&model->nSV[i]);
		}
		else if(strcmp(cmd,"SV")==0)
		{
			while(1)
			{
				int c = getc(fp);
				if(c==EOF || c=='\n') break;	
			}
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			free(model->rho);
			free(model->label);
			free(model->nSV);
			free(model);
			return NULL;
		}
	}

	// read sv_coef and SV

	int elements = 0;
	long pos = ftell(fp);

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	char *p,*endptr,*idx,*val;

	while(readline(fp)!=NULL)
	{
		p = strtok(line,":");
		while(1)
		{
			p = strtok(NULL,":");
			if(p == NULL)
				break;
			++elements;
		}
	}
	elements += model->l;

	fseek(fp,pos,SEEK_SET);

	int m = model->nr_class - 1;
	int l = model->l;
	model->sv_coef = Malloc(double *,m);
	int i;
	for(i=0;i<m;i++)
		model->sv_coef[i] = Malloc(double,l);
	model->SV = Malloc(svm_node*,l);
	svm_node *x_space = NULL;
	if(l>0) x_space = Malloc(svm_node,elements);

	int j=0;
	for(i=0;i<l;i++)
	{
		readline(fp);
		model->SV[i] = &x_space[j];

		p = strtok(line, " \t");
		model->sv_coef[0][i] = strtod(p,&endptr);
		for(int k=1;k<m;k++)
		{
			p = strtok(NULL, " \t");
			model->sv_coef[k][i] = strtod(p,&endptr);
		}

		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
				break;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			x_space[j].value = strtod(val,&endptr);

			++j;
		}
		x_space[j++].index = -1;
	}
	free(line);

	if (ferror(fp) != 0 || fclose(fp) != 0)
		return NULL;

	model->free_sv = 1;	// XXX
	return model;
}

void svm_free_model_content(svm_model* model_ptr)
{
	if(model_ptr->free_sv && model_ptr->l > 0)
		free((void *)(model_ptr->SV[0]));
	for(int i=0;i<model_ptr->nr_class-1;i++)
		free(model_ptr->sv_coef[i]);
	free(model_ptr->SV);
	free(model_ptr->sv_coef);
	free(model_ptr->rho);
	free(model_ptr->label);
	free(model_ptr->probA);
	free(model_ptr->probB);
	free(model_ptr->nSV);
}

void svm_free_and_destroy_model(svm_model** model_ptr_ptr)
{
	svm_model* model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		svm_free_model_content(model_ptr);
		free(model_ptr);
	}
}

void svm_destroy_model(svm_model* model_ptr)
{
	fprintf(stderr,"warning: svm_destroy_model is deprecated and should not be used. Please use svm_free_and_destroy_model(svm_model **model_ptr_ptr)\n");
	svm_free_and_destroy_model(&model_ptr);
}

void svm_destroy_param(svm_parameter* param)
{
	free(param->weight_label);
	free(param->weight);
}

const char *svm_check_parameter(const svm_problem *prob, const svm_parameter *param)
{
	// svm_type

	int svm_type = param->svm_type;
	if(svm_type != C_SVC &&
	   svm_type != NU_SVC &&
	   svm_type != ONE_CLASS &&
	   svm_type != EPSILON_SVR &&
	   svm_type != NU_SVR)
		return "unknown svm type";
	
	// kernel_type, degree
	
	int kernel_type = param->kernel_type;
	if(kernel_type != LINEAR &&
	   kernel_type != POLY &&
	   kernel_type != RBF &&
	   kernel_type != SIGMOID &&
	   kernel_type != PRECOMPUTED)
		return "unknown kernel type";

	if(param->gamma < 0)
		return "gamma < 0";

	if(param->degree < 0)
		return "degree of polynomial kernel < 0";

	// cache_size,eps,C,nu,p,shrinking

	if(param->cache_size <= 0)
		return "cache_size <= 0";

	if(param->eps <= 0)
		return "eps <= 0";

	if(svm_type == C_SVC ||
	   svm_type == EPSILON_SVR ||
	   svm_type == NU_SVR)
		if(param->C <= 0)
			return "C <= 0";

	if(svm_type == NU_SVC ||
	   svm_type == ONE_CLASS ||
	   svm_type == NU_SVR)
		if(param->nu <= 0 || param->nu > 1)
			return "nu <= 0 or nu > 1";

	if(svm_type == EPSILON_SVR)
		if(param->p < 0)
			return "p < 0";

	if(param->shrinking != 0 &&
	   param->shrinking != 1)
		return "shrinking != 0 and shrinking != 1";

	if(param->probability != 0 &&
	   param->probability != 1)
		return "probability != 0 and probability != 1";

	if(param->probability == 1 &&
	   svm_type == ONE_CLASS)
		return "one-class SVM probability output not supported yet";


	// check whether nu-svc is feasible
	
	if(svm_type == NU_SVC)
	{
		int l = prob->l;
		int max_nr_class = 16;
		int nr_class = 0;
		int *label = Malloc(int,max_nr_class);
		int *count = Malloc(int,max_nr_class);

		int i;
		for(i=0;i<l;i++)
		{
			int this_label = (int)prob->y[i];
			int j;
			for(j=0;j<nr_class;j++)
				if(this_label == label[j])
				{
					++count[j];
					break;
				}
			if(j == nr_class)
			{
				if(nr_class == max_nr_class)
				{
					max_nr_class *= 2;
					label = (int *)realloc(label,max_nr_class*sizeof(int));
					count = (int *)realloc(count,max_nr_class*sizeof(int));
				}
				label[nr_class] = this_label;
				count[nr_class] = 1;
				++nr_class;
			}
		}
	
		for(i=0;i<nr_class;i++)
		{
			int n1 = count[i];
			for(int j=i+1;j<nr_class;j++)
			{
				int n2 = count[j];
				if(param->nu*(n1+n2)/2 > min(n1,n2))
				{
					free(label);
					free(count);
					return "specified nu is infeasible";
				}
			}
		}
		free(label);
		free(count);
	}

	return NULL;
}

int svm_check_probability_model(const svm_model *model)
{
	return ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
		model->probA!=NULL && model->probB!=NULL) ||
		((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
		 model->probA!=NULL);
}

void svm_set_print_string_function(void (*print_func)(const char *))
{
	if(print_func == NULL)
		svm_print_string = &print_string_stdout;
	else
		svm_print_string = print_func;
}

////////////////////////////////////
// END SVM CPP
////////////////////////////////////
// training svm-train -b 1 data model
class SVMPredictor
{
public:
	svm_model * model;
	svm_node * x;

	// each classes' label and its probility
	double * prob_estimates;
	int * predict_labels;
	int classes;

	// model should be trained by probility model
	SVMPredictor(const char * modelFile, int maxFeatureDim = 128)
	{
		model = svm_load_model_char(modelFile);
		classes = svm_get_nr_class(model);
		prob_estimates = (double *) malloc(classes * sizeof(double));
		predict_labels=(int *) malloc(classes * sizeof(int));
		svm_get_labels(model, predict_labels);
		x = (struct svm_node *) malloc((maxFeatureDim + 1) * sizeof(struct svm_node));
	}

	// model trained from file 
	SVMPredictor(vector<vector<pair<int, double> > > & input, vector<double> & output, string modelsave, int maxFeatureDim = 128)
	{
		// default values
		svm_parameter param;
		param.svm_type = C_SVC;
		param.kernel_type = RBF;
		param.degree = 3;
		param.gamma = 0;	// 1/num_features
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 100;
		param.C = 1;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.nr_weight = 0;
		param.weight_label = NULL;
		param.weight = NULL;
		param.probability = 1;

		int max_index = 0;
	
		svm_problem prob;
		prob.l = input.size();
		prob.y = new double[output.size()];
		for (int i = 0; i < prob.l; ++i)
			prob.y[i] = output[i];
		prob.x = new svm_node*[prob.l];

		for (int i = 0; i < input.size(); ++i)
		{
			prob.x[i] = new svm_node[input[i].size() + 1];
			for (int j = 0; j < input[i].size(); ++j)
			{
				prob.x[i][j].index = input[i][j].first;
				prob.x[i][j].value = input[i][j].second;
				max_index = max(max_index, input[i][j].first);
			}
			prob.x[i][input[i].size()].index = -1;
		}
		param.gamma = 1.0/max_index;

		model = svm_train(&prob, &param);
		classes = svm_get_nr_class(model);
		prob_estimates = (double *) malloc(classes * sizeof(double));
		predict_labels=(int *) malloc(classes * sizeof(int));
		svm_get_labels(model, predict_labels);
		x = (struct svm_node *) malloc((maxFeatureDim + 1) * sizeof(struct svm_node));

		svm_save_model(modelsave.data(), model);
	}

	pair<int, double> predict(vector<pair<int, double> > & feature)
	{
		for (unsigned int i = 0; i < feature.size(); ++i)
		{
			x[i].index = feature[i].first;
			x[i].value = feature[i].second;
		}
		x[feature.size()].index = -1;

		int predict_label = svm_predict_probability(model, x, prob_estimates);
		for (int i = 0; i < classes; ++i) 
			if (predict_labels[i] == predict_label) 
				return make_pair(predict_label, prob_estimates[i]);
	}

	static void outputModelToCode(const char * file, const char * codefile)
	{
		ifstream fin(file);
		ofstream fout(codefile);

		fout << "char modelData[1000000] = {";
		string line;
		while (getline(fin, line))
		{
			for (unsigned int i = 0; i < line.length(); ++i) 
				fout << (int)line[i] << ",";
			fout << (int)'\n' << "," << endl;
		}
		fout << "0};" << endl;
		fin.close();
		fout.close();
	}
};

#ifndef LOCAL
const int CV_BGR2GRAY = 1;
const int CV_BGR2Lab = 2;
//const int IPL_DEPTH_32F = 32;
const int CV_32FC1 = 32;

///////////////////////
// BASIC STRUCTURE
///////////////////////
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


///////////////////////////////////////
// CVMAT
///////////////////////////////////////
struct CvMat
{
	int type;
    int step;

    /* for internal use only */
    int* refcount;
    int hdr_refcount;

    union
    {
        unsigned char * ptr;
        short* s;
        int* i;
        float* fl;
        double* db;
    } data;

    union
    {
        int rows;
        int height;
    };

    union
    {
        int cols;
        int width;
    };


	CvMat(int h, int w)
	{
		data.fl = new float[h * w];
		height = h;
		width = w;
		rows = h;
		cols = w;
	}

	~CvMat()
	{
		delete [] data.fl;
	}
};

CvMat * cvCreateMat(int h, int w, int depth)
{
	CvMat * mat = new CvMat(h, w);
	return mat;
}

void cvReleaseMat(CvMat ** mat) { delete *mat; }

inline double cvmGet(CvMat * mat, int x, int y) { return mat->data.fl[x * mat->width + y]; }
inline void cvmSet(CvMat * mat, int x, int y, double value) { mat->data.fl[x * mat->width + y] = value; }
void cvZero(CvMat * mat) { for (int i = 0; i < mat->height * mat->width; ++i) mat->data.fl[i] = 0; }

/////////////////////////////
// IPLIMAGE
/////////////////////////////
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

void cvZero(IplImage * img) { for (int i = 0; i < img->height * img->width * img->nChannels; ++i) img->imageData[i] = 0; }
void cvSet2D(IplImage * img, int i, int j, CvScalar s) {
	int step = img->widthStep;
	img->imageData[i * step + j * 3 + 0] = s.val[0];
	img->imageData[i * step + j * 3 + 1] = s.val[1];
	img->imageData[i * step + j * 3 + 2] = s.val[2];
}

IplImage * cvCreateImage(CvSize size, int depth, int channels)
{
	IplImage * img = new IplImage(size, depth, channels);
	return img;
}

void cvReleaseImage(IplImage ** img) { delete *img; }
IplImage * cvLoadImage(const char * data) { IplImage * img; return img; }
void cvSaveImage(const char * data, IplImage * img) {}
CvSize cvGetSize(IplImage * img) { return cvSize(img->width, img->height); }
void cvWaitKey(int x) {}
void cvLine(IplImage * img, CvPoint p1, CvPoint p2, CvScalar color, int x, int y, int z) {}
//////////////////////////
// ADVANCED
//////////////////////////
void cvShowImage(char * name, IplImage * srcImage) {}

CvScalar CV_RGB(int r, int g, int b) { return cvScalar(r, g, b); }

void cvRectangle(IplImage * img, CvRect rect, CvScalar color, int w)
{
	return;
}

#endif

void cvMyResize(IplImage * srcImage, IplImage * dstImage)
{
	double srcHeight = srcImage->height;
	double srcWidth = srcImage->width;
	double dstHeight = dstImage->height;
	double dstWidth = dstImage->width;

	double revHeightScale = srcHeight / dstHeight;
	double revWidthScale = srcWidth / dstWidth;

	unsigned char * srcData = (unsigned char *) srcImage->imageData;
	unsigned char * dstData = (unsigned char *) dstImage->imageData;
	int srcStep= srcImage->widthStep;
	int dstStep = dstImage->widthStep;
	int nC = srcImage->nChannels;
	for (int k = 0; k < nC; ++k)
		for (int i = 0; i < dstImage->height; ++i)
			for (int j = 0; j < dstImage->width; ++j)
			{
				int value = 0;
				double u = (double) i * revHeightScale;
				double v = (double) j * revWidthScale;

				double x = u - floor(u);
				double y = v - floor(v);
				double minDis = 1e10;
				int x1 = (int)u;
				int x2 = (int) ceil(u); x2 = min(x2, srcImage->height);
				int y1 = (int)v;
				int y2 = (int) ceil(v); y2 = min(y2, srcImage->width);
				int f00 = srcData[x1 * srcStep + y1 * nC + k];
				int f01 = srcData[x1 * srcStep + y2 * nC + k];
				int f10 = srcData[x2 * srcStep + y1 * nC + k];
				int f11 = srcData[x2 * srcStep + y2 * nC + k];
				value = f00 * (1 - x) * (1 - y) + f10 * x * (1 - y) + f01 * (1 -  x) * y + f11 * x * y;

				dstData[i * dstStep + j * nC + k] = value;
      }
}

IplImage* getSubImage(IplImage * image, CvRect roi)  
{  
  if (roi.x < 0) roi.x = 0;
  if (roi.y < 0) roi.y = 0; 
  if (roi.x + roi.width >= image->width) roi.width = image->width - roi.x;
  if (roi.y + roi.height >= image->height) roi.height = image->height - roi.y;

  IplImage * result = cvCreateImage(cvSize(roi.width, roi.height), image->depth, image->nChannels);
  int step = result->widthStep;
  int stepSrc = image->widthStep;
  unsigned char * data = (unsigned char *) result->imageData;
  for (int i = 0; i < result->height; ++i)
    for (int j = 0; j < result->width; ++j)
      for (int k = 0; k < result->nChannels; ++k)
        data[i * step + j * result->nChannels + k] = 
        image->imageData[(i + roi.y) * stepSrc + (j + roi.x) * result->nChannels + k];
  return result;  
}  

IplImage * constructImage(int H, int W, vector<int> & image)
{
  IplImage * img = cvCreateImage(cvSize(W, H), 8, 1);
  unsigned char * data = (unsigned char *) img->imageData;
  int step = img->widthStep;

  int cnt = 0;
  for (int i = 0; i < H; ++i)
    for (int j = 0; j < W; ++j)
    {
      data[i * step + j] = image[cnt];
      cnt++;
    }
  return img;
}

const int hog_hist_bins = 9;

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

  /** 矩形区域 */
  CvRect rect;

  /** 面积 */
  double area;

  /** 周长 */
  double length;

  /** 像素点矩形面积比 */
  double ratio;

  CvPoint center;

  /** 区域内的边缘能量 */
  double totalEnergy;

  /** 区域内的平均边缘能量，能量方差 */
  double averEnergy;
  double torEnergy;

  /** 区域的额外信息，可设置成自己想要的数据 */
  int label;
  string sLabel;
  int tag;
  int filterTag;

  int validHeng;
  double validHengRatio;
  int validHengAverH;
  double validHengPosi;

  int validShu;
  int validShuAverW;
  double validShuPosi;
  double validShuRatio;

  /** 识别出的标记 */
  string text;
  bool valid;


  /** 构造函数 */
  CvRegion(vector<CvPoint> & ps)
  {
    points = ps;
    totalEnergy = 0;
    center.x = 0;
    center.y = 0;
    tag = 0;

    if (points.size() == 0) return;

    // 抽取特征
    calRegionFeature();
    text = "1";
    valid = true;
  }

  CvRegion()
  {
    points.clear();
    totalEnergy = 0;
    center.x = 0;
    center.y = 0;
    tag = 0;
    text = "1";
    valid = true;
  }

  /** 输出 */
  friend ostream & operator << (ostream & out, CvRegion & region)
  {
    out << "Ps = " << region.points.size() << " Label=" << region.text << " ";
    out << "(" << region.center.x << "," << region.center.y << ") ";

    //out << "MinRect = " << "(" << region.box.center.x << 
    //	"," << region.box.center.y << ")" << "  H = " << region.box.size.height << " W = " << region.box.size.width << " ";
    out << "" << "H = " << region.rectH << "[" << region.minH << "-" << region.maxH << "]" <<  " W = " << region.rectW << "[" << region.minW << "-" << region.maxW << "]";// << " MinRect = " << 
    return out;
  }

  string toString(int H, int W, bool expand)
  {
    if (text.length() > 10) text.resize(10);

    int newminW = minW;
    int newmaxW = maxW;
    int newminH = minH;
    int newmaxH = maxH;

    if (expand)
    {
      newminW = minW - rectW / 4;
      newmaxW = maxW + rectW / 4;
      newminH = minH - rectH / 4;
      newmaxH = maxH + rectH / 4;
    }
    else
    {
      newmaxW = maxW + rectW / 4;
      newmaxH = maxH + rectH / 4;
    }

    if (newminW < 0) newminW = 0;
    if (newmaxW >= W) newminW = W - 1;
    if (newminH < 0) newminH = 0;
    if (newmaxH >= H) newmaxH = H - 1;


    char temp[200];
    sprintf(temp, "4 %d %d %d %d %d %d %d %d %s", newminW, newminH, newmaxW, newminH, newmaxW, newmaxH, newminW, newmaxH, text.c_str());
    return string(temp);
  }


  /** 计算区域内点的特征 */
  void calRegionFeature()
  {
    calRegionBasicFeature();
  }

  void calRegionBasicFeature()
  {
    int i, j, k;

    /** 计算区域内的平行坐标轴矩形覆盖 */
    minH = 9999;
    maxH = -1;
    minW = 9999;
    maxW = -1;
    center.x = center.y = 0;
    for (k = 0; k < points.size(); ++k)
    {
      int nowW = points[k].x;
      int nowH = points[k].y;
      minH = min(minH, nowH);
      maxH = max(maxH, nowH);
      minW = min(minW, nowW);
      maxW = max(maxW, nowW);
      center.x += nowW;
      center.y += nowH;
    }

    rectW = maxW - minW + 1;
    rectH = maxH - minH + 1;
    area = rectW * rectH;
    center.x /= (int) points.size();
    center.y /= (int) points.size();

    ratio = (double) points.size() / rectH / rectW;

    rect = cvRect(minW, minH, rectW, rectH);

    validHeng = 0;
    validHengAverH = 0;
    validHengPosi = 0;
    vector<int> cnt(maxH + 1, 0);

    for (k = 0; k < points.size(); ++k)
    {
      i = points[k].y;
      j = points[k].x;
      cnt[i]++;
    }

    for (i = minH; i <= maxH; ++i)
    {
      if (cnt[i] >= rectW * 7 / 10) 
      {
        validHeng++;
        validHengAverH += i;
      }
    }

    validHengRatio = 0;
    if (rectH > 0) validHengRatio = (double) validHeng / rectH;
    if (validHeng > 0) validHengAverH /= validHeng;
    if (validHeng > 0) validHengPosi = (double) (validHengAverH - minH) / rectH; 

    validShu = 0;
    validShuAverW = 0;
    validShuPosi = 0;
    vector<int> cntW(maxW + 1, 0);

    for (k = 0; k < points.size(); ++k)
    {
      i = points[k].y;
      j = points[k].x;
      cntW[j]++;
    }

    for (i = minW; i <= maxW; ++i)
    {
      if (cntW[i] >= rectH * 7 / 10)
      {
        validShu++;
        validShuAverW += i;
      }
    }

    validShuRatio = 0;
    if (rectW > 0) validShuRatio = (double) validShu / rectW;
    if (validShu > 0) validShuAverW /= validShu;
    if (validShu > 0) validShuPosi = (double) (validShuAverW - minW) / rectW;
  }

  // 简单合并两个区域
  void mergeRegion(CvRegion & r)
  {
    minH = min(minH, r.minH);
    maxH = max(maxH, r.maxH);
    minW = min(minW, r.minW);
    maxW = max(maxW, r.maxW);

    rectW = maxW - minW + 1;
    rectH = maxH - minH + 1;
    area = rectW * rectH;

    this->center.x = center.x * this->points.size() + r.center.x * r.points.size();
    this->center.x /= (this->points.size() + r.points.size());

    this->center.y = center.y * this->points.size() + r.center.y * r.points.size();
    this->center.y /= (this->points.size() + r.points.size());

    points.insert(points.end(), r.points.begin(), r.points.end());
    r.points.clear();

    calRegionBasicFeature();

    text = text + r.text;
    r.valid = false;
  }
};

bool comRegionByX(const CvRegion & r1, const CvRegion & r2)
{
  return r1.minW < r2.minW; 
}

bool comRegionByY(const CvRegion & r1, const CvRegion & r2)
{
  return r1.minH > r2.minH;
}


////////////////////////////////////////////////////////////////////////////////////////////
// FLOOD FILL 
// 根据输入的单通道二值化图像，获得类似Segmentation的 comID二维数组，
// 即每个像素点属于哪个连通分量 
// 对于每个非零元素进行FLOOD_FILL，comID从0开始编号，获得CvRegion表示
////////////////////////////////////////////////////////////////////////////////////////////
int dir[8][2] = { {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {-1, -1}, {1, 1}, {-1, 1}, {1, -1} };
void floodFill(IplImage * img, vector<vector<int> > & comID, vector<CvRegion> & regions, int dirs, int minSize, int maxSize)
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
      if (data[i * step + j] != 0 && comID[i][j] == -1)
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

        if (points.size() <= maxSize && points.size() >= minSize) // 只抽取适度大小的物体
          regions.push_back(CvRegion(points));
        else
          regions.push_back(CvRegion());
        coms++;
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
// id 从0开始编号
//////////////////////////////////////////////////////////////////////
int floodFill(IplImage * img, vector<vector<int> > & comID, vector<int> & comSize, int dirs)
{
  int i, j;
  int height = img->height;
  int width = img->width;
  comID = vector<vector<int> >(height, vector<int>(width, -1));
  comSize.clear();

  int coms = 0;
  unsigned char * data = (unsigned char *)img->imageData;
  int step = img->widthStep;
  for (i = height - 1; i >= 0; --i)
    for (j = 0; j < width; ++j)
      if (comID[i][j] == -1)
      {
        int value = data[i * step + j];
        int comsize = 0;
        comID[i][j] = coms;
        queue<pair<int, int> > q;
        q.push(make_pair(i, j));

        while (q.size() > 0)
        {
          pair<int, int> top = q.front();
          q.pop();
          int x = top.first;
          int y = top.second;
          comsize++;

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
        coms++;
        comSize.push_back(comsize);
      }
      return coms;
}

void getBinaryCharImage(IplImage * gray, IplImage * binary)
{
  int h = gray->height; 
  int w = gray->width;
  unsigned char * data = (unsigned char *) gray->imageData;
  int step = gray->widthStep;
  unsigned char * out = (unsigned char *) binary->imageData;

  for (int i = 0; i < h; ++i)
  {
    for (int j = 0; j < w; ++j)
    {
      int d = data[i * step + j];
      if (d < 80) out[i * step + j] = 255; 
      else out[i * step + j] = 0;
    }
  }
}

const int FEAH = 8;
const int FEAW = 6;
const int FEADIM = FEAH * FEAW;

vector<pair<int, double> > extractFeature(IplImage * img)
{
  int HH = FEAH;
  int WW = FEAW;
  IplImage * resizeImg = cvCreateImage(cvSize(WW, HH), 8, 1);
  cvMyResize(img, resizeImg);
  getBinaryCharImage(resizeImg, resizeImg);

  /**
  cvNamedWindow("debug");
  cvMoveWindow("debug", 200, 200);
  cvShowImage("debug", img);
  cvWaitKey(0);
  cvShowImage("debug", resizeImg);
  cvWaitKey(0);
  */

  vector<pair<int, double> > ret;

  for (int i = 0; i < HH; ++i)
    for (int j = 0; j < WW; ++j)
    {
      unsigned char * data = (unsigned char *) resizeImg->imageData;
      int step = resizeImg->widthStep;
      int v = data[i * step + j];
      if (v != 0) 
        ret.push_back(make_pair(i * WW + j + 1, 1));
    }

  cvReleaseImage(&resizeImg);
  return ret;
}

string pattern = string()+string("(000000000000100000010001010001010011001110001100"
)+string(")001100001110011110010001000001000001100000000000"
)+string("0000100011110010001110001110001110001010001011110"
"0000000011100000010100001000000010000001000000111"
"0001100011110011111110001100000011111011111001110"
"0001100011110011111100000100000011111011111001110"
"0000100001110011111010101000000011111011111001110"
"0000100011110011111100000100000011111011111001110"
"0000000001110011111000000000000011111011111001110"
"0000000011110010001010001010001010001010001011111"
"0000100011010010001110001110001010001010011011110"
"0000100011011010001010001010001010001010011011110"
"0010000011100100010100001000000010000001101000111"
"0001110011011110011110001110001110001010011001110"
"0000011011111000111000111000111000111000111000111"
"0000000011010010011110001110001110001110001011110"
"0000100011011010011110001010001010001010011001111"
"0000100011010010001110001110001110001010001011110"
"0000100011011010001010001110001010001010001001110"
"0000100011011010011010001110001110001010001011011"
"0001100011010010001110001110001110001010001011110"
"0001100011011010001110001110001010001010001011011"
"0000000011110100010100001000000010000001000000111"
"0000110001011010001010001010001010001010001001011"
"0000100011010010001010001010001010001010001001110"
"0000100011110010001010001110001010001010001011010"
"0000000000010000001010000100001010010100010010100"
"0001100110010000011100010000011000100000010000100"
"0001000110010100010000001100010000000000100000000"
"0000000000111001001000000000001010000100001010010"
"0001000011110110111100001100000010000011111001111"
"0001000011110100011100001100000010000011000001111"
"0001100011110100011100001100000010000011101000111"
"0011000011100100011100001000000010000001101000111"
"0001000011110111111100001100001010000011111001111"
"0001000011110100011100001110000010000011101001111"
"0001100010011010001100001100001110001010001011010"
"0001100010001010001110001100001010001010001010010"
"0001100011110010011000001100000110001110011011110"
"0001000011100000010000001000000010000011001001110"
"0001100001111010001010000110000010001010001001110"
"0000100011110000001100000100000100001010010011110"
"0011000011110110010100001100001010001011011001111"
"0001100011110100001100000100000000001011001001110"
"0001000011100000011100001100000000000011101011111"
"0011100010110000011100001100000010000011001001111"
"0000100001110010001010000000000100001011110001100"
"0001110011111010001010000000001000001111011001110"
"0001000011110010001110001110001010001110011011110"
"0000100010011010001010001110001010001010001001010"
"0000000011010010001110001100001110001010001011010"
"0000000011111010001100000100000000001011011011111"
"0000100011111010001010001010001010001010001011110"
"0000100011111110011110001110001110001010011011110"
"0000000011111010001010001000000010001011001011111"
"0001110001110010001000001100001100001010001011110"
"0000100011111010001100000100000010001011011001110"
"0000100011111010001100000100000000001011011011110"
"0001110011111010001100000100000100001011111011111"
"0000100011111010001000001100000100001010011011111"
"0000000011111010001100000100000100001011001011110"
"0001110011111010001000001100000000001010011011110"
"0000100000000010001010001100001010001010001001110"
"0000100011010010001010001110001110001110001001000"
"0000100011110010001110001110001110001010001010010"
"0000110001010010001010001010001010001010001001110"
"0000100010000000001000000000000000000010000010010"
"0000100010001100000100000000000000000010001001111"
"0000000010010000001100001100001100001000001001110"
"0000110011001010001010000110001010000010001001011"
"0000100001011010001010001110001110001010001011011"
"0000110011001010001010001010001010001010001001110"
"0001110011001010001010001010001010001010001001000"
"0000110011011010001010001100001010001010001011011"
"0000100011011010001110001110001110001010001011011"
"0000100011110010000110001110001110001010000011110"
"0001110010001010001110001110001010001010001001010"
"0000100010001000001000000000000000000000000011001"
"0001110011011010001110001100001110001010001001011"
"0001100011111010011000001100000000001011111011110"
"0000100011011010001110001110001010001010001001110"
"0001100011010010011110001110001110001010011011110"
"0001000011111010001100001100000100001011111001110"
"0000000011011010001010001010001010001010001011011"
"0000000011111010001000000000000000000010001011111"
"0001000010011010001010001010001110001010001011111"
"0001100011111000001000000000000000000010011001110"
"0000000011111010001000000100000000000010001001110"
"0000000010010110001100001100001100001110001010011"
"0000100010001010001010001110001010001010001011011"
"0000100010011010001010001110001010001010001010010"
"0000000011011010001010001010001110001010001010011"
"0000000010000010001110001100001110001010001011010"
"0001100010011010001100001100001100001010001010010"
"0001100010010010001010001110001010001010001001010"
"0000100011111010001100000100000010001011111001110"
"0000000010001010001100001100001000001010001011110"
"0000000010001010001110001010001110001010001011011"
"0000100010010000001100001100001100001010001010010"
"0000100010010010001100001000001110001010001010010"
"0001110011011010001000000100000000000010001001110"
"0000010100001101001010001010001110001100010010000"
"0010010100101001000011001010001000010000010010100"
"0000110101001101001010001010001110010000010011100"
"0000100011111010001000000100000000000010001011111"
"0001100011110010001110001110001010001010001011110"
"0000100010010010001000001100001010001010001001010"
"0000100010010010001100001100001000001010001011010"
"0010000100100100010100001010000010000001000000111"
"0010000100100100010100001000000010000001000000111"
"0000000011100000010000001000000010000001000000111"
"0000000011100100010000001010000010000001100000111"
"0010000101100100010100001000000010000001000000111"
"0001100001111010001010001100001110001011011001110"
"0000000001110010011010001110001110001010001011111"
"0000000011110010011000001100001100001010011011110"
"0000100001110010011010001110001110001010001011110"
"0001100011110010011010001010001110001010011011110"
"0001000011110010011110001110001110001110011011110"
"0000100001110011011010001110001110001010011011110"
"0001000011110010011110001110001110001010011011110"
"0000000001110011011010001010001110001010011011110"
"0001000011100110110110011110001010001011011001111"
"0001000011110010011010001110001110001011011001110"
"0000000011110010011010001110001110001011110011110"
"0000000001110010001010001110001110001010001011110"
"0000000001110010011010001010001110001010011011110"
"0000000011110010001110001110001110001010011011110"
"0000100001110010011010001110001110001010011011110"
"0000100001110010011110001110001110001010011001110"
"0000100011011010001110001110001010001010001011011"
"0000100011111010001110001100000010001010001001110"
"0001110011111010001010001000001010001010001001110"
"0000110001001000001010001010001110010100010011100"
"0000010001001001001010001010001100010100010110100"
"0000010001001010001010001110001100001100010111100"
"0000000001001000001010001010001010000100010011100"
"0000110001001000001010001010001010000000010010110"
"0000010001001010001010001010001100011100010011100"
"0000100011010010001010001010001110001010001011110"
"0000100011011010011110001110001110001010011011110"
"0000110011011010001110001110001110001010001011110"
"0000100011010110001110001100001110001110001011110"
"0000100011010010001010001110001010001010001011110"
"0000100001010010001010001110001010001010001011010"
"0000100011010010001110001110001010001010001001110"
"0000100001010010001110001110001110001010001001110"
"0000100011011010001010001110001110001110001011010"
"0000100011011110001110001110001110001110001010010"
"0000100010011010011110001110001110001010011011110"
"0000100010010010001110001010001010001010001001110"
"0000000011010010001010001010001010001010001011010"
"0000100011010010001110001010001010001010001011110"
"0001100010010100011100011100011100011110011011110"
"0001100011010110001110001110001110001010010011110"
"0000100011010110011100001100001100011110011011110"
"0000100001110010001010001100001110001010011011110"
"0001110010001010001100001100001100001010001011111"
"0000100010011010001010000100001010000010001011011"
"0001110010010010001110001110001010001010001011110"
"0000100011110010001000001000001110001010011001110"
"0000100010011010001110001010001010001010001001110"
"0000100010001010001100001100001100001110001011110"
"0000100010011010001110001100001100001010001011011"
"0000100011110010001010001010001010001010001011110"
"0000100010011010001010001100001000001010001011011"
"0000010001111001001010001010011110010110010111100"
"0000011001101011001011001010011110010110010011100"
"0000010001101001001011001010011010010110010011100"
"0000000000101001001011001010001010010110010011100"
"0000100011111010001100001100001100001110001011111"
"0000100011111110001010001010001010001010001011111"
"0001100010000100001100001100001000000000001010001"
"0001100010001100001100001100000100000100001011001"
"0011000011110000011100001100001100011000010011100"
"0011000011110010011110001010001010001111011011110"
"0001110001010010001000000000000100001010011011110"
"0001000011000010100100010100001100001010011001110"
"0011000011100100010100011100001000001010011001110"
"0000000001110000011010011110001110011111110011110"
"0000100011111010001100001100001010001010011011111"
"0000100000000010001110001110001110001010001000010"
"0001100010000110001110001110001110001010001001010"
"0000000111011100011100001100001100001110011001110"
"0001110010010110001100001100001110001010001001110"
"0001100011011110011110001100001110001010011001110"
"0000100011111110001110001100001110001010001001111"
"0000100110010100001100001100001100001110011011110"
"0001100010011100001100001100001100001110001011110"
"0000100011001010001010001110001110001010001011111"
"0000100011001110000100000100000110000011001001111"
"0000100010011110001100001100001100001010001011011"
"0001110011111010001000001100000010001010001011111"
"0000000011010110001100001100001100001110001001110"
"0001100010010010011110001000001110011010010011110"
"0001100011010100011100001100001100001010011011110"
"0000100011010010011110011100001010001010011011110"
"0001100010001100001100001100001100001010001011110"
"0000100011111110001010001010001010001010001001110"
"0000100011110110011110001110001110001110001001110"
"0001100010010100011100011100011100011110010011110"
"0000000001111110001010001010001010001010001001111"
"0000100011111010001010000100000010000010001011111"
"0000100011010010011110001110001110011010010001110"
"0000100010010010011110001110001010001010011011110"
"0000100011001010001110001010001010001010001011001"
"0000110010011010001010001010001010001010001001010"
"0001100010011110001100001100001110001010001011110"
"0001100010011000001100001100001010001010001011110"
"0000100011111010001100000100000100000010001011111"
"0001100011111010001100000100000100000010011011110"
"0001110001110010001100000100000000000011111001110"
"0001110011110010001000001100001110001011111001110"
"0001000010010010001100001000001010001010011011110"
"0001100000010010001110001110001110001010001001010"
"0001110010001010001000000100000000000010001010011"
"0001100010001110001110000110000110001010001011011"
"0000100010001110001100001100001100001110001011110"
"0000100010010010001100001100001100001010001011010"
"0000110010011010001000001000001010001010001001110"
"0001110010011010001110001110001110001010001001110"
"0001100010010010001110001110001010001010011011110"
"0000100011010010001110001110001110001010001001110"
"0001100001111010001010001110001110001010001011111"
"0001100011011010001110001110000010001010001011110"
"0000100011111011011011011110011110011010010011110"
"0000100001111011011011011110011110011010010011110"
"0000110001111011011010011010011110011011111011110"
"0001110010001110001110001110001110001110001011011"
"0000100011010000001100001100001100001000001010010"
"0000100010011010001100001100001110001010001011110"
"0001100010001100001100001100001100001100001011110"
"0001110010011110001110001110001110001110001011111"
"0001100010011010001110001110001110001010001011010"
"0001100010010110001110001110001110001110001010010"
"0001110011011010011110011010001010001011011011111"
"0001100111110110011110011110011110011110011011110"
"0001110011111010001110001010001110001010011011110"
"0000100011110011011010001010011010011011011011110"
"0001000010010110001110001110001110001110001011110"
"0000100111111110011110011110011110011110011011111"
"0001100011111110011110011110011110011010011011111"
"0000100011111110011010011010001010001010011011111"
"0000100011111111011111001111001111001011011011111"
"0001000011111111111110001110001111011011111000100"
"0000000011111011111010001000001010001011111011111"
"0001000011111011111110001100001111001011111011110"
"0001100011110110001100001100001100001110011011110"
"0001110011111011111010101010001011111011111001110"
"0001110011110011111110001110001111111011111011110"
"0001100011111110001100000000000010000011101001111"
"0001100011111000000000000100000010000010001001111"
"0000010001001010001010001110001110001110011010010"
"0000000010000010100000100000101100010100000001000"
"0001100001110010001010001110001010001010001001110"
"0000000010010010010010010110010110011110010010100"
"0001000010100010100000100000100100100100100000000"
"0000100010011010001010001010001110001010001011110"
"0000000011110010001110001110001110001010001011111"
"0000100011011010001010001110001010001010001011110"
"0001000010010010001110001100001000001010001011110"
"0001110011011010001010000110000010001010001001111"
"0001100011011010001110001010001010001010001011111"
"0001100001011010001010001010001010001010001001110"
"0000100011110010001010001110001110001010001011111"
"0001100011110010001000000100000000000010001011111"
"0000000011111011111010001110001110001011111001110"
"0001100011010010000110001110001110001010001011110"
"0000000001110011110010111100011110011011110011000"
"0000000001111011001010001010001010001011011001110"
"0000100001111011111010001010000110001110011011110"
"0001011011101000010100001110001110000010001101111"
"0000100010011010001110001110001110001010001001010"
"0000000011010010001110001110001110001010011001010"
"0000000001110011011011001010001010000011101100111"
"0000100001110010011110001110001010001011111001110"
"0000100011111010001100001100000000001010001011111"
"0001110011111010001100000100000100001010011011111"
"0001100011010010011110001110001110001010001010010"
"0000100011011010001010001110001010001010011001110"
"0001000011110010001010001010001110001010001011010"
"0001100010010010011110001110001110001010011001110"
"0000000011010010001110001110001110001010001011010"
"0001100010010010001010001100001010001010001011111"
"0000100010011010001010001110001010001010001001110"
"0000100010011010001010001010001010001010001001110"
"0000000001110010001010001110001010001010001001010"
"0000100011010010001110001110001010001010001001010"
"0000100011011010001010001110001010001010001011111"
"0001110010010010001110001110001010001010001001110"
"0001100011011010001100001000001010001010001001011"
"0001100010010110011110001110001110011010011001110"
"0000000010011010001110001110001010001010010011110"
"0000000010010010001110001110001110001010001010010"
"0001000011110110001110001110001110001010001001111"
"0001100010010110001100001100001110001010001011110"
"0000100011011010001010001110001010001010001011011"
"0000100001110010001010001110001110001110001011010"
"0000100011011010001010001010001010001010001001110"
"0000100011110010001110001110001110001010001011010"
"0000100011010010001110001110001010001010001011010"
"0000000011010110011100001100001100001010011011010"
"0001110001011110001110000110000110001010001001010"
"0000100010010010001010001110001010001010001010010"
"0001100010011010001110001010001010001010001001110"
"0001100010011010001010001110001010001010001001110"
"0001000011110010001110001110001110001010001001110"
"0001100010001010001010001110001010001010001001010"
"0001100010001010001110001010001010001010001001110"
"0000000101100100001011001000000010011000100010000"
"0000100011110010011010011010001010011011011001110"
"0001100011110010011010011110011010011010011011110"
"0001100010010110001100001100001100001110011011010"
"0001100011110010011110011110011110011010011011110"
"0000100011110010001000000000000000001011111001110"
"0001110011111010001100000100000000000011111001110"
"0000000011110010001100000100000000000011011001110"
"0000100011110011011000000100000000000011111001110"
"0000100011111010001100000100000100000011111001110"
"0000000011010010001010001110001010001010001001010"
"0000000011110010111100000100000100000011011001110"
"0000100011110010001100000100000100000011111001110"
"0001100011110010001100000100000100000011111001110"
"0000100001110011011000000100000000000010011001110"
"0001110011111010001000000000000100001010001011110"
"0000000011110010001100001100000000000010001011110"
"0000010001001011001010001010001110011110011010110"
"0000100010011110001100001100001110001010001011110"
"0000100011000010001010001010001010001010001001010"
"0000110011011010001110001110001110001010001011111"
"0001100010011110001100001100001100001010001011110"
"0000000011011010001100001100001100001010001011111"
"0000100010001000000000000100000100000100000010000"
"0000100011001000000000000000000000000000001000010"
"0000110010001100001100001100001100001100001001010"
"0000100010001100001100001100001100001100001000010"
"0000000010001000000000001100001000001000001001010"
"0000000001010010001110001110001110001010001001110"
"0000100000010100000100000100001100001100001010010"
"0000100010001100000100000100000100000100001001010"
"0000100010001100000100000100000100000100000010000"
"0000000010000100000100001100001100001100001010000"
"0000100010001100001100001100001100001100001010010"
"0000100010001010001010001010001010001100001000010"
"0000100010001100000100000100000100000100000010010"
"0000100010001000000000000000000000000000000010000"
"0000100010000000001000001000001100001000001010000"
"0000100011011010001010001010001010001010001000010"
"0000100010001100000100000000000100000100001010000"
"0000000011111011101100000100000000000011111011111"
"0000100011111011111000001100001010001011111011111"
"0000100011111010011010000100000010000011111011111"
"0000100001001010001010001010001110001010001001110"
"0001100001011010001110001110001110001010001001110"
"0000100011111010011000001100000010001010001011111"
"0001110011111010001100001100000100001010001011111"
"0000000011010010001010001010001010001010001000010"
"0000000011111010001000000100000010001011111001110"
"0000100011111011111010001100000010000011001011111"
"0000110011111010001100000100000110001011111011110"
"0000100011111010001000000100000100000010001011110"
"0001100011110010011100001100001000001011111011110"
"0001100011111010001100001100000100001010001011111"
"0000110011110010001100000100000100000011111011110"
"0000100011111010011100001100000100001011111001110"
"0001010011010110001100001100001100001110001011110"
"0000100010010010001100001100001100001010001011110"
"0000000011010010001110001110001110001010001001110"
"0000100011110011111100001100001100001110001011111"
"0001110011111010001010000010000010000011111001110"
"0000100010011010001100001100001100001010001011010"
"0000000011010010001100001000001010001010001011110"
"0000100010011010001000001100001010001010011011110"
"0001100010010100001100001100001100001110001011110"
"0001100010110100011100001100001100001100011010010"
"0000100010001010001110001110000110001010001001110"
"0000100010001110001110001100001110001010001011110"
"0001100010010010001110001100001110001010011001110"
"0000100011011010001110001010001110001010001011111"
"0000000001011010001110001010001010001010001001110"
"0000100011010110001100001100001110001110011011110"
"0000110011011010001010001010001110001010001001010"
"0100001100111001001001001010000000010100100011000"
"0100001100111000001001001010001010010000010111100"
"0100011000100001000001001010001000010100100111000"
"0000000101000100100100010010001001000000100110011"
"0000010011001100100000010010001001000000100000011"
"0010000001000100100010010010001001000101000110110"
"0000010111001000100010000010001001000000100010011"
"0100001000100101001001001010001000010000000001100"
"0100000000100001000000001010001010010000100011000"
"0100000000101100001001001010001010010000110001100"
"0001110011111010001100001100000000001011111011110"
"0001110011110010001100001100000000001011111001110"
"0000100010010010001110001110001010001010001011010"
"0001100010011010001010001110001010001010001001010"
"0000100010011010001110001110001010001010001011010"
"0001110010000010001010001010001010001010001001010"
"0001110010001110001100001100001100001010001011010"
"0000010011110010001100001100000000001011011011110"
"0000110011111010001000000100000000000010001011111"
"0000100011111010001010001010001010001011001001110"
"0000100011110010011010001110001110001010011001110"
"0001110011111010001100000100000000001010011001110"
"0011110010011000001100000100000000000010011011110"
"0000100000010010000010001010001010001010000001010"
"0001100011111010001100000100000000000010001001110"
"0001100011111010001100000100000000000010001011111"
"0001100011111000000100000100000000000000001011111"
"0011110010001000001100000100000100000000001011111"
"0011110010001000000000000100000000000010001011111"
"0001110010001000000100000100000100000000000011111"
"0011110010001000000100000100000000000010001011111"
"0001110010001000000100000100000100000000001011111"
"0001110010001000000100000100000000000010001011111"
"0011110010001000000100000100000000000000000011111"
"0001110010001000000100000100000000000000000011111"
"0011110000001000001100000100000100000000001011111"
"0001110011111000000100000100000100000000000011111"
"0001110010001000001100000100000100000000001011110"
"0011110011111000000100000100000100000000000011111"
"0011110010001000000100000100000100000000001011111"
"0001110010001000000000000100000000000010001011111"
"0000000011111000001100000100000100000000001011111"
"0011110011111000000100000100000000000000000011111"
"0001110011011000000100000100000100000000000011111"
"0001110010001000000000000100000000000000000011111"
"0001110010001010001010001010001010001010001001010"
"0011010011111000000100000100000100000000001011111"
"0011110011111000000100000100000100000000001011111"
"0001110011111000000100000100000000000000000011111"
"0001110011111000000100000100000100000000001011110"
"0000000011111000000100000100000100000000000011111"
"0000100011111010001010001110001110001010001011111"
"0001000011011010001110001110001110001010001011111"
"0000001000111000001001001010001000010000100011000"
"0000000000111001111011001010001110011011110011100"
"0000000000111001101011001010000010010110100011000"
"0001110000000010001010001110001010001010001001010"
"0000010100111001011000001010001000010110100011000"
"0000010000111001101010001110010100010110100111000"
"0000010000111001001010001010000000010111100011000"
"0000011000111001101010001010001110010110110011100"
"0000010000111001111011001011001111011011110011100"
"0000010000111001001011001010001010010111110011100"
"0000010000111001101011001010001110010111110011000"
"0000011001111011001010001110001110001010111011110"
"0000111001111010111010011100011100011111110111100"
"0000011001111010011010001110011100011111111011100"
"0001100010000010001110001110001110001010001001010"
"0000000001111001111011001010001110001010011011110"
"0000000001111011111010001100001100010111110011100"
"0000000000111001111010001110001100001011110011100"
"0000010000111001011010001010001110010110110011100"
"0000010001111001001010011110011100110111100011100"
"0000010000111001101011001010001100010100110011100"
"0000110000111001001010001110001110011011110011100"
"0010000100111101101010001010001110000110110011100"
"0001100010010010011110001110001100001010011000010"
"0000010000111001001010001110001110000110110011000"
"0000001000011001001001001010001010011111110011100"
"0100011001110001100011000010001100000010010011100"
"0010001100011000000011001110001110010100000111000"
"0100001000111001101001001010001010010100100011100"
"0000000100111101101001001010001010010100110011000"
"0000011100111101000011001010001110011110010011100"
"0100001100011100100001001010001010010110100011000"
"0001100011111110001110001110001010001010001011111"
"0001100011111010001110001110001010001010001011110"
"0000100001010010001110001110001110001010001010010"
"0000100010001010001110001110001110001010001001010"
"0000100011010010001110001110001010001010011001010"
"0000010001001011001010001010001100011100010011100"
"0011100100010100001100001000000011000001001000111"
"0000110001001001001010001010001010010100010010100"
"0000010001001010001010001010001110001110010010100"
"0000110001001010001010001110011110010100010111100"
"0000110001001010001010001010001110010100010011100"
"0000110001011010001010001110001110011100010110100"
"0000100011011010001010001100001010001010001011011"
"0000000010010000001000001100001000001000001011010"
"0000100010001010001100000100000100000010001011011"
"0000110001111011001010001010001110011110010011100"
"0000110101111000001010001010010010010110010011100"
"0011000011110100011100001000000010000011101000111"
"0000010001111010001010001110001110011100010111110"
"0000110001101011001010001010001010011110010011100"
"0000111001001011001010001010001010011110010011110"
"0000111101011011001010001010011010011110010011100"
"0000010001111011001010001010001010011110011011110"
"0000110001111010001010001110001110011100010111100"
"0000110001111011001010001010001110011110010011110"
"0000110001001011001010001010001010010010010011110"
"0000110001011011001010001010001110010110010011100"
"0010000011100100110100001100000010000011000000111"
"0000100011010011011111011111011111011011011011010"
"0001100011010010011010011110011010011010011001010"
"0001100010011110011110011110011110011010011001110"
"0000100010010010011010011110011010011010011001010"
"0000100001011011011111011111011111011011011001010"
"0001100011010011011011011010011010011011011001010"
)+string("1000001000011111111000011000011000011000011000011"
"1100000110000111000011110010011010001010000000000"
"1000000000111111111000011000011000111000011000011"
"1000011011111000111000011000111000111000111000111"
"1000011000111111111000011000011000011000011000011"
"1000001000111010011000011000011000011000011000011"
"1000000001111111111000111000111000011000011000011"
"1000011000111110111000111000111000011000011000011"
"1000010000111011111000011000011000011000111000011"
"1000011000111111111000111000111000111000011000011"
"1100000110000011000011110010011010001010000000000"
"1011111111111111111111111011111010000010000010000"
"1011111111111111111011111010000010000010000000000"
"1000111011111111111011111010000010000010000000000"
"1011111111111111111011111011111010000010000010000"
"1000001000011000110001110001100001000010000010000"
"1000001000001000010001110000100001000011000010000"
"1000000100000011000011110010111000011000000000000"
"1000000000011000010000100000100001000011000010000"
"1000000000001000110000110000100001000011000010000"
"1000001000011000110000100000100001000011000010000"
"1000001000001001110000110000100001000001000010000"
"1000000000011000110000110000100001100001000010000"
"1000001000011000110000100000100001000010000010000"
"1000000001100001100001100001100001100001100011111"
"1000100001100001100001100001100001100001100001110"
"1000111111111010011010000010000011000011000011000"
"1110000011000011110010110010111010001010000000000"
"1000011111111111111011111010000010000010000010000"
"1000001111111011111010000010000010000010000000000"
"1000110011111111111111111011001010000010000010000"
"1001101111111011111010000010000010000010000000000"
"1000001011111111111011111010000010000010000010000"
"1000100011111111111111111010000010000010000010000"
"1000100111100001100001100001100001100001100011100"
"1000110111110000110000110000110000110000110001110"
"1000000111110000110000100000110000100001100001110"
"1000000100000011100011110010011000001000000000000"
"1000100111100001100001100001100001100001100011110"
"1000110001100001100000100001100001100001100001100"
"1000100011100001100001100001100001100001100001100"
"1000110001110000110000110000110000110000110000100"
"1001110001110001110001100001110001100001100001110"
"1000110001110000110000110000110000110000110000110"
"1000000110000011000010111000011000001000000000000"
"1001100111110011110000110000110000111000110000110"
"1000100111100001100001100001100001100001100001110"
"1000000111110001110001110001110001110001110001110"
"1000011000111011110000110001110001110001100011100"
"1000001000111111111000110000110000100001100011000"
"1000001000111011111000110000110001100001100001100"
"1000001000111111111000111000110001100011100011000"
"1000001000111111111000110001110001100001100011000"
"1100000100000011000011110010111010001000000000000"
"1000001000111111111000110000110001100001100001100"
"1000011000011111110000110001110001100011100011000"
"1000001000111000110000110000110000100001100001000"
"1000001000011111111000010000100000100001100011100"
"1000001000111111111000010000100000100001100011100"
"1000001000011111111000110000110001100001100001100"
"1000001000111000111000110000110001110011100011100"
"1000001000011111111000110000110000100001100001100"
"1100000110000111000011110010111010011010000000000"
"1000001000111111111000110000100001100001100011000"
"1000001000011011111000010000110000110001100001100"
"1000001000011111111000111000110001100011100011000"
"1000001000011011111000011000110000100001100011100"
"1000001000011111110000110000110001100011100011100"
"1110000110000011100011110010111010011000001000000"
"1000001000011111110000110000100000100001100001100"
"1000001001111000011000011000011000011000011000011"
"1000011001111000011000011000011000011000011000011"
"1000010001111000011000011000011000011000011000011"
"1100000110000011000010110010111010001010001000000"
"1000011000011111110000110000100000100001100011100"
"1000011000111111111000110000100000100001100011000"
"1000001000111111110000110000100001100001100001100"
"1000001000011111111000111000110000100000100001100"
"1000001000011111111000111000110000100001100011000"
"1000001000111111110000110000110001100001100011100"
"1000001000011111111000111000110000110000100001100"
"1000000000111011111000011000011000011000011000011"
"1000001000111111110000110000110001100001100001000"
"1000001000011111110000110000110001110001100001100"
"1000001000011111111000110000110001100001100001000"
"1000001000111111110000110000100000100001100011000"
"1000001000011011111000011000110000100001100011000"
"1000001000011111110000110000100001100001100011100"
"1110000110000011100011110010111000001000000000000"
"1000011000111111110000110000110000100001100011000"
"1000000000111111110000110000100001100001100011000"
"1000001000011111111000110000100001100001100011100"
"1000001000111111111000111000110000110001100001100"
"1000000110000011000011110010110010011010000000000"
"1000000000111111110000110000110000100001100001100"
"1000001000011111111000011000110000110000100001100"
"1000011000111111110000100000100001100011000011000"
"1000001000111111110000110000110001100001100001100"
"1000000011100001100001100001100001100001100001110"
"1000000011110001110001110001110001110001110001110"
"1000000010000011000011111010011010001000000000000"
"1000100111110001110001110001110001110001110001110"
"1000000001110001110001110001110001110001110001110"
"1000000000000011111011111011111011111010000010000"
"1000000000000111111111111011111010011010000010000"
"1000000000000001111111111011111011111010000010000"
"1000000000000111111011111011111010000010000010000"
"1100000110000111100011110010111010011000000000000"
"1000000000000011111111111011111011111010000000000"
"1000000000000000000111111011111011111010000010000"
"1000010001110011110000111000111000110001111000110"
"1000000011111001111000111000111000111000111000111"
"1100000100000011100011100010111010011000001000000"
"1110011001111001111000011000011000011000011000111"
"1000110001111001111000111000111001111000111000111"
"1000010001110001110000110000111000110000111000111"
"1000010001111000111000111000111000111000111000011"
"1000010001111011111000111000111110111100111000111"
"1100000010000011100010110010011010001010000000000"
"1000010001111011111000111000111000111000111000111"
"1000010001110011110000110000110000110000110000110"
"1000011011111111111000111000111000111001111000111"
"1000011001111001111000111000111000111000111000111"
"1000001011111010011000011000011000011000011000011"
"1000010001110011110000110000110000110001110000110"
"1100000110000111000011110010111000001000000000000"
"1000000001111111111000111000011000011000011000111"
"1000010000000111110000100001000001100011000000000"
"1100000110000011100011111010011000001000000000000"
"1100000111000011100011111011111000001001000001000"
"1100000111100011110011111010111000001001000001000"
"1100000110000111100011111010111010001000000001000"
"1100000111000011110010111000011000000001000001000"
"1100000111000011110011111000011000001001000001000"
"1100000110000111110011111010011001000001000001000"
"1110000011000011100010110010111000001000001000000"
"1100000111000011110011111010111010001001000001000"
"1100000110000011100011110010111000001001000001000"
"1100000111000011100011111010111010001001000001000"
"1100000111000011110011111011111001001001000001000"
"1110000111000011110011111010011000000000000001000"
"1000001000011000011000001000001000001000001000001"
"1100000110000011000011110010011000001000000000000"
"1000001000111111011000011000011000011000001000011"
"1000001000111111001000011000011000001000011000001"
"1000001001111111111000011000011000011000011000011"
"1000000000111011111010111000011000011000011000011"
"1000000110000010000011000011100011111011111011111"
"1000011000111111111000011000011000111000011000111"
"1000001000111011111110011000011000011000011000011"
"1000000011110000110000010000010000110000110000110"
"1100000100000111000011110010111010011010000000000"
"1000000011110011110000110000111000110000110000111"
"1000011111111001111000011001111000011001111001111"
"1000110011110000110001110000111000111001110001110"
"1000010001110011110000110001111001111001110001110"
"1000010001110111110001110001110001110011110001110"
"1000010001111000011000011000011000011000111000011"
"1100000010000011100011110010111010001010001000000"
"1000010001110000111000111000111000111000110000110"
"1000000001111001111000111001111001111001111001111"
"1000011111111000111000011000011000011000111000111"
"1011111111111011111010000010000010000010000010000"
"1000111111111011111011111010000010000010000010000"
"1000001000111000111000011000011000011000011000011"
"1000000000011000011000011000011000011000011000011"
"1000011000111000111000111000111000111000111000111"
"1111111111111011111010000010000010000010000000000"
"1011111111111011111010100010000010000000000000000"
"1010111111111111111010000010000010000010000000000"
"1111111111111011111011111010000010000010000010000"
"1010000111111011111010000010000010000010000000000"
"1101111111111111111011111010000010000010000010000"
"1100000010000011100010110010111010001010000000000"
"1010011111111111111011111010000010000000000000000"
"1111111111111111111011111010000010000010000010000"
"1010011011111010000010000000000000000000000001000"
"1000000110000011000011110010001000001000000000000"
"1000010011111011101010000000000001000001000001000"
"1000000111111011111010000010000000000000000001000"
"1000000111111111111010000010000010000000000001000"
"1100000110000111000011110000011000011000000000000"
"1000001000001000001000011111101000001000001000001"
"1000000011110011110011110011110011110011110011110"
"1000100011110000110000110000110000111000111000011"
"1001110111111000111000111000111000111000111000111"
"1000110111111000111000111000011000011000011000111"
"1000111001111011110111100111100111000110000100000"
"1001110111110001111000111000111000111000111000111"
"1000110111111000111000011000011000011000011000111"
"1000000111111000011000011000011000011000011000011"
"1100000010000011100011110000111000001000000000000"
"1001100111110000110000111000110000110000110000110"
"1000000000000111111111111110000010000010000000000"
"1011111111111111111010000010000011000001000000000"
"1000000000000000010111111010000010000010000000000"
"1000000000000011010111111111111010000010000010000"
"1000000000000000000010110111111010000010000000000"
"1000000000000000000111111010000010000000000000000"
"1000000000000011000011110000111000011000000000000"
"1000000000000111111111111000000000000000000000000"
"1000000000000011111011111101111000000000000000000"
"1000000000000011111111111000000000000000000000000"
"1000100111111111111011111011000001000001000001000"
"1100000010000011000011100010110010010010011000000"
"1000010011111111110011110001110001110001110011110"
"1000111000111000111000111000111000111000111111111"
"1000000000111001111001111011111011111111110111110"
"1000110001110001110001111001110001110001110111110"
"1000010001111001111001110001111001111001110111110"
"1000100001110000111000111000111000111001111111110"
"1001000011100011100111100011100011110011110111110"
"1011100111110011110001111011111011111011111011111"
"1000110000111001111001110001110001110011110011100"
"1000110001110001110011110111110111100111110111100"
"1000100000111000111001110011110011110011100111100"
"1000010001111001111001110001110001110001110111110"
"1000000111100111100111100111100111110011110011111"
"1000010001111001111001111001111001111011110111111"
"1001100011111011111011111011111011111111111111111"
"1001100111100111100011110011110011111011111011110"
"1001000111100011100011100001110001110001111001111"
"1000000110000011000011100010111010011010000000000"
"1011000111110111110111110111111111110111110111110"
"1011000111100011110011110011110011110011110011110"
"1000100011111011111011111111111011111011111111111"
"1011100011111011110011111011111011111011111011111"
"1000100011111011111011111011110011110011110011110"
"1000100011110111111011110011110011111111111011111"
"1111100111100011100111110111111011100111100111100"
"1011100111100111100111100111100111100111111111111"
"1011110011111011111011111111111111111011111011111"
"1100000110000011100011110010111010001010000000000"
"1011110111110011111011110011110111111011110011110"
"1000000011110111110011110011110111110111111111110"
"1000000011111011111111111111111111111111111011110"
"1000001000111100111000110000110001100001100011000"
"1000001000011111110000110000100001100001000011000"
"1000001000111111111000110000110001100011100011000"
"1000001000111111111000110000110000100001100001100"
"1000001000111111110000110000110001100001100011000"
"1000001000011111110000110000100000100001000001000"
"1000001000110000110000110000100000100001000001000"
"1000001000011000110000110000100000100001000001000"
"1100000110000011100010111010011010001010000000000"
"1000001000011111111000110000110000100001100001000"
"1000001000011000111000110000110001100001100011000"
"1000001000011111111000110000110000100001100011000"
"1000001000111000110000110001100001100001100011000"
"1000011000110000110000100000100000100001000011000"
"1000011001111000110000110001100001100011100011000"
"1000000110000011000011100000011000011000000000000"
"1000000000011111110000110000100001100001000001000"
"1000001000011011111000110000110000100001100001000"
"1000001000011111110000110000100000100001100001000"
"1000100001100011100000100000100000100000100000100"
"1000000001100011100000100000100000100000100011111"
"1000000001100011100000100000100000100000100001100"
"1100000110000111000011111010011010001000000000000"
"1001000001100111100001100001100001100001100001100"
"1000000000011011111000011000011000011000011000011"
"1000001001111000110000110001110001100011100011000"
"1000011011111000110000100001100011100011000111000"
"1000000011111000110000100001100011100111100111000"
"1000011011111000111000110000100001100011100011000"
"1000001011111000110001110001110001100011000011000"
"1000001011111000110001110001100011100011100011000"
"1000000011111000110000110001100001100011000011000"
"1000001011110011110001100001100011100011000111000"
"1000011011111000111000110001110001100011000011000"
"1000011011111000110000110001100011100011100011000"
"1000000011110100001000000000000000001000001000001"
"1000000000110110010000010000010000010000010000011"
"1000000000011011001000001000001000001000001000001"
"1000000110000011000011100010111000001000000000000"
"1000000000110110110000010000010000011000011000011"
"1111000111000111000001000111000111000001000001111"
"1011111000111000111000111000100001100111000111000"
"1000000011100011100000100000100000100000100011111"
"1000100011100111100000100000100000100000100111111"
"1000100001100111100000100000100000100000100000100"
"1000100011100011100001100001100001100000100011111"
"1001000001000111000001100001100001100001100011111"
"1000100011100011100000100000100000100000100001111"
"1000100001100000100001100001100001100001100001100"
"1000000110000011000011100010111010001000000000000"
"1000000011100000100000100000100000100000100001111"
"1000000000010011110000011000011000011000011000010"
"1000001000111111111000011000011000011000011000011"
"1000001001111010011000011000011000011000010000011"
"1000000000011111011000001000001000001000001000011"
"1000011011111011011000011000011000011000011000011"
"1000010000111110011000011000011000010000011000011"
"1111111111111111111010000010000010000000000000000"
"1000000011111111111010010010000000000000000000000"
"1000000100000011000011100010111010011010000000000"
"1000000111111111111011111010000010000000000000000"
"1111001111111011111010000010000000000000000000000"
"1001110111111111111011111010000010000010000000000"
"1000000000000000111111111011111000000000000000000"
"1000000000000000000111111011111010000000000000000"
"1000000000000000000111111011111011111000000000000"
"1000000000000000000000000111111011111000000000000"
"1000000000000000000001111011111011111000000000000"
"1111100000100000100000100000100000100000100001110"
"1100000110000011000011110010111010011010000000000"
"1000000001100000100000100000100000100000100001100"
"1001000011100001100001100001100001100001100011110"
"1000100000100000100000100000100000100000100000100"
"1000001000111111111000001000001000001000001000001"
"1000000000011111111000011000011000011000011000011"
"1000001000111011111000011000011000011000011000011"
"1000001000011011111100011000001000011000011000011"
"1000011001111011111110111000110000110000110001110"
"1000001000111111111000011000011000011000011000001"
"1100000110000011110010110010011010001000000000000"
"1000011000111011111110111000111001110001110001110"
"1000000000111011111010111000111000110001110001110"
"1000011000111011111110110000110001110001110001100"
"1000011001111011111110111000111000111000110001110"
"1000011001111011111111111000111000110001110001100"
"1111100001100001100001100001100001100001110001110"
"1011100001110001110001110001110001110001110001111"
"1000000001100001110001110001110001110001110001110"
"1011110001110001110000110000110000110000110001111"
"1000000110000011000011110010111000001000000000000"
"1001100011111000111000111000111000111000111000111"
"1011011111111000111000111000111000111000111000111"
"1111100111111001111001111001111001111001111001111"
"1011110001111001111000111000111000111000111000111"
"1111111000110000110000110000110000111000110001111"
"1011110011111000111000111000111000111000111000111"
"1000010111111001111000111001111001111001111001111"
"1111111011111000111000111000111000111000111000111"
"1011110011111000011000111000111000111000111000111"
"1001110011111000111001111001111001111000111000111"
"1111110011111001111001111001110001110001110001111"
"1111110000111000111000111000110000110000110000111"
"1111110000111000111000111000111000111000111000111"
"1011110011111000111000110000110000110000110000111"
"1000010111111001111000111001111000111000111000111"
"1011110011111000110001111000111000111000111000111"
"1011011111111111111011111010000010000010000000000"
"1011111001111000111000111000111000111000111000111"
"1000110111111000111000111000111000111000111000111"
"1011111011111000111000111000111000111000111000111"
"1000000011111001110001110000111001111001111001111"
"1000000001001011111011111111111111111111101100000"
"1000000011111111111111111111111111011000000000000"
"1000000000001111111011100011100000100000000000000"
"1000011011111111111011100011100001000001000000000"
"1000001011111111000111000011000001000000000000000"
"1000000000011000111111100111100011000001000001000"
"1111111111111011111011111001000001000001000000000"
"1000000000000000011111110011000000100000100000000"
"1011111011111011111011111010011010000010000010000"
"1011111011111111111011111011000001000001000001000"
"1000000010000111111011111001000001000001000000000"
"1010000011000011111011000001000001000000000000000"
"1010110011111111111010000000000000000000000000000"
"1111111111111111111011000011000011000001000000000"
"1000001000111000101000110001110000110001110001100"
"1000001001111000111000111000110001110001110001010"
"1000001000111000111000110000110111110001100001100"
"1000001001111000111000111001110001110001110011110"
"1000001001111000011000111000110000110001110001000"
"1000001000111000011000110000110001110001100001100"
"1000001000111000111000111000111001110001110001110"
"1000001000111000101000101000110000110001110000110"
"1000001000111000010000110000110000110000100001100"
"1000001000111000101000110000110001110001110001010"
"1000001000111000111000110000110000110001110001100"
"1000001000111000011000111000110000110000110001100"
"1000001001111000111000111001111001110001110011100"
"1000000000011000101000111000110000110001010001000"
"1000001000111000111000110000110001110001100001100"
"1000001001111000101000111000111001110001110001010"
"1000001000111000011000111000110000010000110001100"
"1000001000111000111000110000110001110001110001100"
"1000001001111000110000110001110001100001110000110"
"1000000000111000101000110000110000110000110001100"
"1000000001000001000000100001001011000011000011000"
"1000001001111000111000110001110001100001100001100"
"1000000000100000100000100000100000100011100000100"
"1000000001000001000001000001100001001001000010000"
"1000000000110111110000010000010000010000010000010"
"1000011001111111111000011000011000011000011000010"
"1011111111111111111011111010000000000000000000000"
"1000001000111011011000011000011000011000011000011"
"1010000111111011111010000010000010000000000000000"
"1011111111111011111010000010000000000001000001000"
"1000011000111001110001110001110011110011110011100"
"1000000000111000111001110001110001100011100111100"
"1000010000111000111000111001100001100011100111100"
"1100011100111000110001100011100011100111100111100"
"1000010000111000110011100011100011100111100011100"
"1000100001110001110001110011100011100011110011110"
"1001100111111011111001110001110001110001110001110"
"1001100111100011100011100011100011100001110001111"
"1001100111111011110011100011100011110111110111100"
"1010000010000111101111111011111001111001111000011"
"1000000010000011000011111011111001111001111000001"
"1001101111111011111010000010000010000000000001000"
"1011111111111111111011111010000010000010000010000"
"1011111111111011111010000010000010000000000001000"
"1000011111111011111010000010000010000000000001000"
"1000011001111000111000111000111000111000111000011"
"1000000000111000111000111000111000111000111000111"
"1000011111111000110000110000110000110000110000010"
"1000011000111000111000111000111000111000111000011"
"1000000011110000110000110000110000110000111000011"
"1000000001111000111000111000111000111000111001110"
"1000001011111001111000011000011000011000111000011"
"1000001001111000111000111000011000011000011000011"
"1000001011111000111000111000111000111000011000111"
"1000000001110000110000110000110000111000111000111"
"1000011111110000110000110000110000111000110000110"
"1000000111110000110000111000110000110000111000110"
"1000001011111000111000111000110000110000110000110"
"1000000011111000111000111000111000111001110000110"
"1000001001111000011000011000110000110000111000111"
"1000010111111000111000111000111000111000111000111"
"1000000111111000111000111000110000111000111000111"
"1000011011111000011000011000011000011000011000011"
"1000001000111011111000111000111000111000110000110"
"1000001011111000111000111000111000111000111000111"
"1000001111111000111000111000111000111000111000111"
"1000001000111000111000111000111000111000111000111"
"1000001011111000011000011000011000011000011000011"
"1000010011111000111000111000111000111000111000111"
"1000011000111000111000111000111001111001111000111"
"1000011001111000111000111000111000111000111000111"
"1000000001111000010000011000010000011000011000011"
"1000011111111000111000111000111000111000111000111"
"1000001000111000111000111000111000111000111000011"
"1000000000111000111000011000011000011000011000011"
"1000011111111000111000111000011000011000111000111"
"1000011001111001011000011000011000011000011000011"
"1000011001111000111000111000111000011000110000111"
"1000001111111000111000111000111000111000111000011"
"1000001001111000111000111000111000111000111000111"
"1000001011111000011000111000111000111000011000111"
"1000010111111000111000011000011000011000011000111"
"1000010001111000111000111000111000111000111000111"
"1000001111111000011000011000011000011000011000011"
"1000011000111000111000011000011000011000011000011"
"1000000000010000100000010000100001100001000010100"
"1000010001111111111000011000011000011000011000011"
"1000000000000011011011111011111011111010000010000"
"1000000000000111111111111011111010000010000010000"
"1000000000000111111111111011111010011000000010000"
"1000000000000011101011111011111011111010000010000"
"1000000000000111111111111011111011111010000010000"
"1000000000000111111111111011111011110010000010000"
"1000000000000111111011111010111010000010000010000"
"1000101111111111111011110010000010000000000000000"
"1000001011111011111011000010000000000000000000000"
"1111000111111011111011111010000010000000000000000"
"1000111111111000111000111000111000111000111000111"
"1111111111111011111011101010000000000000000000000"
"1000011001111000111000110000110000100000100001100"
"1000001001111000011000110000110000110000110001100"
"1000001000111000011000110000110000110000110001100"
"1000000001111000011000110000110000110001110001110"
"1000011000011111111000011000011000011000011000011"
"1000000111111000111000111000111000111000111000111"
"1000001000011011111000011000011000011000011000011"
"1000000000111111111000011000011000011000011000011"
"1000001000111000011000001000001000001000001000001"
"1000000011100001100001100001100001100001100001100"
"1000000001000001100001100001100001100001100001000"
"1010111111111011111010000010000000000001000001000"
"1000001111111011111011111010000000000001000001000"
"1000011000111111011000011000011000011000011000011"
"1000010111111000111000111000110000111000111000110"
"1000010000111011001000011000011000011000011000011"
"1000001000011011111000011000011000001000011000011"
"1000111001111111111000111000111000111000111000111"
"1000010001111110011000011000011000011000011000011"
"1000111011111001111000111000111000111000111000111"
"1000011011111111111000111000111001111000111000111"
"1000000011111011110000111000111000111000111000111"
"1000000001111011111000111000111000111000111000111"
"1000010111111001111000110000111000110000110000110"
"1000010001111111111000111000111000111000111000111"
"1000011001111001111000111000111000111000111001111"
"1000010011111001111001111001111001110001110001110"
"1000000011111000011000011000011000011000011000111"
"1000100011110001110000110000110000110000110001110"
"1000001000011000011000011000011000011000011000011"
"1000001000111000011000011000011000011000011000011"
"1000010000111000001000001000001000001000001000001"
"1011000111111011111011111010000010000010000010000"
"1111111111111011111010000010000010000010000010000"
"1011111111111011111011111010000010000010000010000"
"1111111111111111111010001010000010000010000010000"
"1011100011111011111011111011111011111011111011111"
"1001110111111111111011111011111011111011111011111"
"1000001000111111011000001100011000001000001000011"
"1000100111111011111001111001111011111011111011111"
"1011100111111011111011111011111011111011111011111"
"1001100111110111110111110111110011110011110011111"
"1001110011111011111011111011111011111011111011111"
"1000001000010111011000010000010000011000011000011"
"1011100011110011111011111011111011111111111111111"
"1000100011110011110011110011110001110011111001111"
"1000010111111011111011111011111001111001111011111"
"1000000111111011111011110001111001111001111011110"
"1000100111110011110011111011111001111011111011111"
"1001110111111011111011111011111011111011111011111"
"1111111111111111111000000000000010000010000000000"
"1000001000111111010000001000010000001000001000111"
"1000000110000011000001100001110000111000011000001"
"1000000000000000000010000111111000000010000010000"
"1000000000000000000011111011111010000010000010000"
"1000000000000000000001110111111000000010000010000"
"1000000000000000000011111111111000000000000010000"
"1000000000000000000000000111111000000000000010000"
"1000011000111010111000111000011000011000011000011"
"1000000000000000000111111111111000000010000010000"
"1000000000000000000000000111111000000010000010000"
"1000000000000000000111111000010000000000000010000"
"1000000000000000000011100011111000000010000010000"
"1000000000000000000000000111111010000010000010000"
"1000000000000000000011011111111000000010000010000"
"1000010000111111111000011000011000011000011000011"
"1000000000000000000011111011111000000010000010000"
"1000000000000000000011111111111000000010000010000"
"1000000000000000000000000011111000000010000010000"
"1000000000000000000000010111111000000010000010000"
"1000000000000000000001111111111000000010000010000"
"1000000000000000000011111111111010000000000010000"
"1000001000111110111000111000111000011000011000111"
"1000000000000000000000000111111000000000000000000"
"1000011000111110011000011000111000111000011000011"
"1000001000001000111000001000001000001000001000001"
"1000000000011011101000001000001000001000001000001"
"1011100011111011111010000000000000000000000000000"
"1000001000011101011000001000001000001000001000001"
"1000001000011011001000001000001000001000001000001"
"1000001000011011011000011000001000001000001000001"
"1000000000011111101000001000001000001000001000001"
"1000001000110011011000011000011000011000011000011"
"1000001000001001111000001000001000001000001000001"
"1000001000001001101000001000001000001000001000001"
"1000000000001001111000001000001000001000001000001"
"1000001001111010111000011000011000011000011000011"
)+string("2000100011011010001000001000011001100010000111111"
"2010000011000100100000100000000010010000001000001"
"2001100011111000001000011000110001100011000011111"
"2000100011110000001000011000110001100011000011111"
"2001100011110000001000011000110001100011000111111"
"2000100011110000011000011000010000100001000011111"
"2001100010011000001000001000010000100000000011111"
"2001100011111000001000001000010000100000000011011"
"2001100011110000001000001000010000100001000011111"
"2000100011111000001000001000010000100001100011111"
"2000110001001001001000011000010001100010000011110"
"2010000011000100101100100000000010010000001000001"
"2000010001111011001000001000010001100010000111110"
"2000010001001011001000001000110001000010000010000"
"2000010101001001001000001000010001000010000011110"
"2000010101101101001000001000010000100010000010110"
"2000010001001011001000011000010001100010000011110"
"2000100011001010001000001000010000100001000010000"
"2000100010011010001000001000010000100001000010000"
"2010000011000100100000100000000010010000000000001"
"2001100010001000001000001000010000100001000010000"
"2001110011001010001000001000010000110001000011000"
"2000100011011010001000001000010000100001000010000"
"2000100010001010001000001000010000100001000010000"
"2010000011000100100100100000000011010000001000001"
"2000110001101001001000001000010001000010000011110"
"2000010001111001001000001000010001100010000111110"
"2000110101101101001000001000010001100010000011110"
"2000000101111001001000001000010001100010000011110"
"2000110101111011001000001000110001100011000011110"
"2000011001101011001000001000010001000010000111111"
"2000110001101011001000001000010001000110000111110"
"2000110101101101001100001000010001100010000011110"
"2000010001101010001000001000110001000010000111110"
"2100111101111001001000001000110001100010000111110"
"2001100011111000011000011000110000100011111011111"
"2001110010111000011000011000110000100001000011111"
"2000100011110000011000011000110000100001001011111"
"2000100011111000011000011000010000100001111011111"
"2000001011001011101111101110011000011010001010000"
"2010000011000100000100100010000010010000001000001"
"2000000011001011101110101110011010001010001010000"
"2000000011001011101111101110001010011010001010000"
"2000000011001011001110101110011000011010001000000"
"2000001011001011101110101110011100011010001010001"
"2000000011001011101110101110011010011010001010001"
"2000001011001011001110101110001100011000001010001"
"2000001011001011101110101110011100001000001010000"
"2010000011000100000100100000000010010000000000001"
"2000100011011010011000011000110001100011000111111"
"2001100011011010001000011000110000100001000111111"
"2000000011011010001000011000110000110011000111111"
"2000100011011010001000011000110001100001000111111"
"2000100001110010011000011000010000100001000011111"
"2000100011011010001000011000110000100011000111111"
"2000100011111110011000011000110000100011000011000"
"2000000011010010001000011000010000100011000111111"
"2011000011000111100100100100110010010010001000001"
"2000110011001010001000001000110000110001000011111"
"2001100011011010011000011000110001100011000011111"
"2000100011011010001000001000010000100001100011101"
"2001110011011010001000001000010000110001000011001"
"2000000011111110001000001000010000110011100011111"
"2000110011011010001000001000010000110011000011111"
"2000100011110110011000010000110000100011000111111"
"2000110011001010001000001000110000110011000111111"
"2000000011011010001000011000010000110001000011111"
"2001100010011110011000010000110001100001000111111"
"2011000011000011100100100100100100010010011010001"
"2000110011011010011000010000010000100001000111111"
"2100010000101101001000001000010000100000000010000"
"2001000011000011100100100100010010010010001000001"
"2011000011001100101100101100011010011010011000001"
"2011000011000110100100100100010010011010001000000"
"2001000011000100100100100100110110010010001000001"
"2011000011000100100100100100010010011010001000001"
"2000000011001100101100101100011010011010001000001"
"2000001011001011000100100100101100001000001010001"
"2011001011001000001100101100001000001010010010011"
"2000001011001000000100100100101100101010001010011"
"2010001011001001001000001100101100001000001010010"
"2000000000001000001011001111101100011100011110011"
"2000001000000000000011000000101100010010001000001"
"2000001000001000000001000100011000001010000001000"
"2000000000011000001010001101101000011000001010000"
"2001110010001010001000001000010001100001000011111"
"2001110011011010001000001000110001100010000111111"
"2011000011000111100100100100100100010010011010001"
"2000100010011010001000001000010001100011000011111"
"2001110011111010001000011000110001100011000111111"
"2000100011111010001000001000011001100011000011111"
"2000100011111010001000001000011001100011000111111"
"2000110011111010001000001000011001100011000111111"
"2001100011111010001000011000110001100011000111111"
"2000100011111010001000001000110001100011000111111"
"2001110011111110001000001000110001110011000111111"
"2001110011111010001000001000110001100011000111111"
"2000100011111110001000001000011000110011000111111"
"2000000011000011100000100100110000010010011010001"
"2000100011111010001000001000011001110011100011111"
"2000100011111110001000011000011001100011100111111"
"2000000011111010001000011000111001110011000111111"
"2000000000000010000000100000000000010000001000000"
"2000100001010000010000010000000001100001000001110"
"2000000000010000010000010000000000100000000001011"
"2000000010010000010000010000100000000001000010001"
"2000000011000011100100100000010110010010011010001"
"2000000011000011100100100100100100010010011010001"
"2000000011001010101000100100010000010010001010001"
"2000001011001011101000001000011010011010001010001"
"2000100010001010001000001000011000100001000011000"
"2001100010001110001000011000010000100001000111000"
"2000100010011000001000000000010000100001000111000"
"2001100011011000001000001000010000100011000111000"
"2001100011010010001000001000010000100001000011000"
"2011110110001100001000001000010000100010000111000"
"2001100110001000001000000000000000100000000010000"
"2010000011000111100100100100110010011010001010001"
"2001100010110110011000011000010001100011000011000"
"2000100011000000001000000000010000100001000011000"
"2000100010011110001000000000010000100001000011000"
"2000100011010010001000011000010000100001000011000"
"2000100011011010001000000000010000100001000011111"
"2001100010011010001000011000010001100001000111111"
"2000000010011000011000010000110000100001000011000"
"2000110010011110001000001000010000110001000011000"
"2001110010010010011000010000010000100001000111111"
"2000000011000011100100100100100000010010011010001"
"2001110011001011001000001000010000100001000111101"
"2000110010001010001000001000010000110001000011000"
"2000000011011110001000001000010000100001000011100"
"2001100010010110011000011000010000100011000111111"
"2000110011011010001000011000010000100001000011111"
"2000000011000011100000100100110000010011011011001"
"2000000011000011100100100100010010011011011011001"
"2000000011000010100100100000010000011011001001001"
"2000000011000011100000100100010000010011001001001"
"2000000011000011100000100100110000010010011001001"
"2000000011000011100100100100010000011011011011001"
"2000000011000011100000100100010000011010001000001"
"2000000011000011100100100100010010010011001001001"
"2000000011000011100000100100010000010011011011001"
"2001110011011010001000011000110001100011000111111"
"2000100011010110011000010000110001100011000111111"
"2000100011011010001000011000110000100011000011000"
"2000000010010010011000011000110001100011000010000"
"2001100011011010001000011000110001100011000111000"
"2001100010011010001000011000110001100011000010000"
"2000000011000011100100100100110100011010011010001"
"2000000011000001100100100100010100011011001011001"
"2000000011000011000100100100110000011010001011001"
"2000000011000011100000100100010010011011001011001"
"2001000011000011100100100000010000010010011000001"
"2001100011011010001000001000111000100010000111011"
"2000100011001010001000001000010000100011000010011"
"2001000011100000100000100100010000010010001010001"
"2000001110000101011000010000100000000010010111110"
"2000000000111001001000010000100001000010010110101"
"2000000001100010001010010000100001000010010000010"
"2000000101001010001010010000000000000000011110010"
"2000010111100101001000110001000010000000010100110"
"2000100000111001001000010000100001000010001010101"
"2001001001101010010000100001000000000010100000101"
"2011000011000010100100100100000000010010001010001"
"2000011001000010001000010000100001000010010011010"
"2000001101101101001010010001000000000010010110100"
"2000011001100000001010100001000001000010010011100"
"2001100111111000011000011000010000100001000011111"
"2001100011111010011000011000110001100001000111111"
"2001100010011110001000001000110011000010000111111"
"2011000011000000100000100100000000010000011010011"
"2011000011000100100100100100000100000100011000001"
"2010000011000100100100100100100100000100011000011"
"2000001011001010100000000100010000000010001000000"
"2000010010011010000000100100100000100010100011001"
"2000110011011110001000001000011001100010000111111"
"2000001011101010100000000100010100000010001000000"
"2000000111000100000100100000000010010000001000001"
"2000000011000000000000000010010000000000001000000"
"2010000001000100100100100000010000000000001000000"
"2000000101000100100000000010010000000000001000000"
"2010000101000100100100000010010000000000001100000"
"2000000011000100100100100000000000010000001000001"
"2010000001000100100000000010010010000000001000001"
"2010000101000100100100100000000010010000001000000"
"2010000101000100000100100100000010010000001100001"
"2001110011111000001000111001100011000110000011111"
"2011100000110000010000110001100011000010000011111"
"2001000010111000001001111011000010000110000111111"
"2001100110011000011000110001100001000011000001111"
"2000110011111111001000111001110011000110000011111"
"2001110111111000011000110001100011000011000011111"
"2001100111110000011000110000100001100001101001111"
"2000100011111000011000110001100011000010000111111"
"2001000001111000001000011001100011000110000011110"
"2111100111110000011000110001100011000110000011111"
"2000000111100000100001100001000010000010000011111"
"2000010001111010001000010001100011000010000111110"
"2011100111110000010000110001100011000011000011111"
"2011000111110000110000100001000011000010000011111"
"2001100011110000110001100001000010000010111111111"
"2001000111100000110001100011000010000110111011111"
"2011000111110000110000100001100011000010000111111"
"2001110010011110001000011001100011000010000111111"
"2000100011110110001000011000110001000010000111111"
"2000000011000111100100100100100100010010011010001"
"2001100011011110001000011001100011000010000011111"
"2001100011010110001000011001100011000010000011111"
"2000000011110110011000011000110011000110000111111"
"2001110011110110001000011000110011000110000011111"
"2000000011000011101000100000100000010010011011001"
"2000000000001011001110101100101100111110011011011"
"2000001011001011101110101100100110111010011011011"
"2000001011001011001110101100101010111010011011001"
"2000000011001111001100101100101110101010011011001"
"2000001011001011101110101000101010101010011011001"
"2011000011000010100100100100010010010010001010001"
"2000111001001001001000001000010000100010000010000"
"2000010001101001001000001000010001100010000011110"
"2000010001101101001000001000010001100010000011110"
"2000011001001001001000001000010001100010000011110"
"2100111101001001001000001000010001000010000111110"
"2000011001101001001000001000010000100010000010000"
"2000111001101001001000001000010001000010000011110"
"2000010001001001001000001000010000000010000111110"
"2000010001001000001000001000010001000010000010000"
"2000010001001001001000001000010000100010000010000"
"2001110010001000001000001000011011000010000110000"
"2001110010001000001000001001110011000010000111110"
"2001100010001000001000001000110010000100000110000"
"2000000110011000001000001000110011000110000110000"
"2000000110011000001000001000110011000100000100000"
"2001110010001000001000001000110011000010000010000"
"2011110010001000001000001000110011000010000011000"
"2000000110011000001000001000110011000100000110000"
"2001100010001000001000001000110011000110000110000"
"2001110010001000001000001000110011000110000111110"
"2000100010011010001000001000010001000010000011111"
"2000000010011100001000001000011001000010000111111"
"2001110010001100001000001000110001000010000111111"
"2001110010001010001000001000110001000010000111111"
"2000100010011000001000001000110001000010000111111"
"2001110010001000001000001000110001100010000111111"
"2000100010001010001000001000010001000010000111111"
"2000110011011000001000001000010001100010000111111"
"2000100011011010001000001000010001100010000011111"
"2000100010011010001000001000010001000010000111111"
"2000100010001000001000001000010011000010000111111"
"2000100010011010001000001000010001100010000110010"
"2001100010011010001000001000010001100010000011111"
"2000110011011010001000001000011001100010000011111"
"2001110010001000001000001000111010000010000010000"
"2000100010001000000000001000110001000010000000000"
"2000110011001000001000001000110011000010000011000"
"2001100010011000001000011000110010000010000011111"
"2000110010001000001000001000110011000110000011111"
"2000100010001000001000001000010001000010000000000"
"2001110010010000001000001000010001000010000010000"
"2000100010001000001000001000110001000000000011111"
"2000100110001000001000001000110001000010000100000"
"2000000110001000000000000000011001000010000100000"
"2000000110001000000000000000010001000010000100001"
"2001100010010000001000001000010001000010000000001"
"2000110110001000000000001000010001000010000100000"
"2000100010001000000000001000010001000010000000000"
"2000110010001000001000001000010011000010000000000"
"2000100010001000000000001000011001100010000010000"
"2000100010001000001000001000010001000010000111111"
"2001100010000000000000001000110001000000000100000"
"2000110010001010001000001000110001100010000011111"
"2000100010001000001000001000110001000010000000000"
"2000100010000000001000001000010001000010000010111"
"2000100010000000001000001000110011000000000000001"
"2010000111100000100000010000010001111010100001100"
"2011000000100000010000011000111001100010100011000"
"2010000001000000100000010000010000010000010001101"
"2001100011011000001000001000011000010111110011100"
"2010000111000000100000010000010000111010100011100"
"2110000001000001000000000000000011110101001011000"
"2100000010000000000001000001000001111111000100000"
"2110000001100000100000110000110110111011100011000"
"2000100010001000001000001000010001000010000100000"
"2010000001100000100000000000100011111111110111100"
"2010000000100000100000100000100111111101000111000"
"2001110110001000001000001000110000000010000011111"
"2001110110001000001000011000110011000010000011000"
"2001100010010000001000001000110010000110000111111"
"2001110110001000001000001000110011000110000110000"
"2000000110011000001000001000110001000110000100000"
"2001100110011000001000001000110011000110000111111"
"2000110010001000001000001000111011000110000110000"
"2011111110001000000000001000111011000010000110001"
"2001100010001000001000001000110001000010000010011"
"2001100110011000001000011000110011000110000111111"
"2011110110001000001000001000110011000010000011000"
"2001110110001000001000011001110010000110000111111"
"2001000110011000001000011000110011000110000111111"
"2011110110011000000000011001110010000010000011111"
"2001110111111100001000001000011011000100000111111"
"2001110110001000001000001000110010000110000110010"
"2000110111011000001000001000110011000110000111111"
"2011110100001000001000011001110010000100000111111"
"2000010010001000000000000000010001000010000110000"
"2000100010011000001000001000110011000010000011111"
"2011111110001000001000001001110011000010000011111"
"2001100111111010001000001000110011000010000010010"
"2011110110001000001000001001110011000010000011111"
"2011110010011000011000011000110010000010000011111"
"2001110110011000001000001000110011000010000010011"
"2001110110011000001000010001110010000110000111111"
"2001110110011000001000001001110010000010000011111"
"2001110010011000001000001000110010000010000011111"
"2000000011001000001000001000110001000010000011111"
"2001110000001000000000001000110010000000000110000"
"2000100110001000001000001000110001000010000011111"
"2011110110011000001000001000110011000110000111111"
"2001110010001000001000001000110011000110000111111"
"2001110110001000001000001000110010000110000111111"
"2001100100001000001000001000110010000100000110010"
"2011111100001000001000001000110011000100000111111"
"2001110110001000001000001000111011000110000111111"
"2001110110001000001000001000111011000110000110000"
"2001110110011000001000001000110010000010000011111"
"2000000010011000001000001000110010000010000110000"
"2001100110000000000000010001110011000010000011011"
"2000000111011000001000001000111011000100000110000"
"2001110110001000001000001001110011000110000110000"
"2001110110001000001000001000110011000010000010000"
"2011111110001000001000001001110011000110000111111"
"2001000011111000001000001000110011000010000011011"
"2000100111011000001000001000110011000110000111111"
"2001110110011000001000001001110011000110000111111"
"2000000110011000001000001000110011000110000110011"
"2000000011111010001000001000110001000010000010000"
"2000100010001000001000001000110011000000000110111"
"2000000111011000001000001000010011000010000011111"
"2001100110001000001000001000110011000100000111111"
"2000000110001000001000001000110011000110000110000"
"2000100110010000000000001000110010000110000111111"
"2001110011001010001000001000010001100010000010000"
"2001110010001010001000001000111001100011000011100"
"2001100010001110001000001000010001100010000010000"
"2001000011000100100100100100010100010010010011001"
"2000000011000010100100100100100000010010010010001"
"2001100000001000001000001000110010000010000000000"
"2000000011001011101000101100101100001010011010001"
"2000000011000011100000100100100100010000011010001"
"2001000011000010100100100000000000010010011011001"
"2000000011100100100100100100000100010010010010001"
"2000000011000000100100100100000100010010011010001"
"2000000011000000100100100100000100010010011000001"
"2001000011000010100000000000000000010010010011001"
"2000001011001011100100100100010100000000001010000"
"2000000011001011000100100100010100000000001010000"
"2000000000001011000010000100010100000000001010000"
"2000000000010000001000010000100001000010000100001"
"2000110011001010001000001000010000100011000010000"
"2000110010001010001000001000010000100001000010000"
"2000100010001110000000001000010001100010000110001"
"2001110010011000001000011000110001100010000111111"
"2001110011011010001000001000010001100010000111111"
"2001110010001000001000001000010001000010000111111"
"2000110011001010001000001000010001100010000010000"
"2000110010001000001000001000010001100010000111111"
"2000110011001010001000001000010001000010000011111"
"2001110010001010001000001000110001100010000110000"
"2001110100001000000000001000110001000010000010000"
"2000000011001110001000001000110001000010000111111"
"2001110010011000001000001000011000110011000011000"
"2001110010011010001000011000010000100001000010000"
"2000100010011010001000001000110001100011000011111"
"2000100011011010001000001000011000110011000011111"
"2000100011111010001000001000011001100010000111111"
"2000100011011010001000001000110001100010000111111"
"2000100011110010001000001000110001100010000011111"
"2000000011011010001000001000011011100010000111111"
"2000100011011110011000011000110001100011000010111"
"2000000011011010001000001000010000100011000011111"
"2001100011011100001000001000010001100011000111111"
"2001110010011010001000001000010000110011000011000"
"2000000011111110001000001000011000100011000011111"
"2000100011110110001000001000010000100011000011111"
"2000100011111010001000001000011000100011000011000"
"2001100011111100001000001000010000100011000111111"
"2000100011111010001000001000010001100011000011111"
"2000100010011010011000011000110001100010000010100"
"2000000001111011011000011000110001110011100011111"
"2000110011111011011000011000110001100011100111111"
"2001100010011000001000011011100010000110000111110"
"2001110010001000001000001011110010000110000111110"
"2001110110001000001000001011110110000110000111111"
"2001100010011000001000001001110010000110000110000"
"2001110110011000001000001011110110000110000110000"
"2001100010011000001000011011100010000110000110010"
"2001100010001000001000001001110010000110000110000"
"2000100011011110001000011000110001100011000011111"
"2001110011111111011000011000110000110001100011111"
"2000100011111011011000011000111000110001100011111"
"2001100011111110011000011000110000110001100011111"
"2000100011111111011000011000111000110001100011111"
"2000100011111110011000010000110000110001100011111"
"2000100011111011011000011000110001110011100111111"
"2001100011011110011000011000110000110001100011111"
"2000000011001011101011110100111111011011011011001"
"2000000011001111101110111100111110011010001010001"
"2000000011111010001000011000010001100011000111111"
"2010000000000101000000000010000000011000011000001"
"2010000111001101000100100010010010011000001000000"
"2000000110000101000100100010100000011000001000001"
"2000001110000101000000100010100010011000011000001"
"2000010100000101000000100010010010011000001000001"
"2000000010000111000101000000100010011010001000000"
"2000000110001101000000100010100000011000011000001"
"2000000000000000000011000101000000100010011010001"
"2000000110000101000000000010100000010000001000000"
"2000000010000011000001000000100000011010001000001"
"2000000000000010000101000100100000010010011000001"
"2000110000001011001000001000100000000010000001110"
"2000110000001011001000001000110010000010001101010"
"2000110000001011001000011000100001000010001001110"
"2000100000001001001000001000110000000010001001010"
"2001110011011110011000011000110001100011000110011"
"2000110000001011001000001000110010000010001001110"
"2000110001001001001000011000100000000010001001010"
"2000110000001011001000001000100010000000001011111"
"2000110001001001001000001000010001000010000001110"
"2000100000001011001000011000100010000010001001010"
"2000000000001011001000001000100000000010001101010"
"2000100011011010001000001000010000100011000011111"
"2001000011011110011000011000110001100011000111011"
"2001110010001110001000001000011001100011000111111"
"2000100001111010001000001000010000100001000011111"
"2000100001111110001000001000011000100011000011111"
"2000100011111010001000001000010000100011000011111"
"2000100011011010001000001000010001100011000111111"
"2000100001111010001000001000011001100001000011111"
"2000100010001010001000001000010000100001000111111"
"2001100010011110001000011000110011100010000111111"
"2000000011000000100100100100010000010010001010001"
"2000000011000011100000100100010000010010001010001"
"2000000011000011100000100100010000010010011010001"
"2000000011000010100100100100010100010010001010000"
"2000000011000010100000100000010000010010001010001"
"2000000011110011010000010000110001100011111110011"
"2000110011111001011000010000110000110011110011111"
"2000000011000011100100100100010100011010001010000"
"2000010011111001001000011000110001100011111011111"
"2000110011111000001000011100010001110011111111011"
"2000100011110011111000011000111001100011111111111"
"2011100011110001010000011100110001100011111111111"
"2000000011000011100100110100010100011010001010001"
"2000000011110001011000010100010000110011111011111"
"2011100011110000110000100001100001110011111010001"
"2000011011111000011000011000010000110111110111111"
"2000110001111100001100011000110001110011111011011"
"2000100111110010010000110000100011100011111011111"
"2000000000001010001011001100111010010010011000011"
"2000000011000011100100100100010100010010001010001"
"2010000101000100100100100000000010010000000000001"
"2000000011000010100100110100010010011010001011001"
"2000000011000010100100100100010000010010001010001"
"2000000011000010100100100100010100011010001010001"
"2000000011000010100000100100010000010010011010001"
"2000000011000010100100100100010100010010001010001"
"2001100010010100010000010000100001000010000010000"
"2001100010110010010000010000100001000011000010000"
"2001100010011110011000011000110001100011000010000"
"2000100011001110001000011000110001100011000111111"
"2011100010011100011000010000110001000010000111111"
"2000000010010010000000010000100001000010000111111"
"2000100011011010001000011000010000100011000110000"
"2001100010011100011000010000100001000010000111111"
"2001100010001010001000010000110001000010000011111"
"2001100010011010011000011000110001100011000011111"
"2001110010001010001000001000110001100010000111111"
"2001100010011010011000011000110001100001000010000"
"2001100110011110001000011000110001000010000111111"
"2000100011011110001000011000110001100011000011000"
"2001100010011110001000010000010001100010000010011"
"2000000010010110001000011000110001000001000111111"
"2001100010011110011000011000100001000010000010011"
"2001100011011010011000011000100001000010000011010"
"2001100010010100000000010000100001100010000111111"
"2001000011110110011000011000110001100010000111000"
"2000110010011110001000011000110000100011000111011"
"2001000011110100011000010000110001000011000110000"
"2001100011110110010000010000110001000011000111011"
"2001100011111010001000011000110001000010000111111"
"2001100110010110010000010000110001000011000111111"
"2001000010010010001000010000100001000010000011111"
"2000100011111110001000001000110000100001000010000"
"2000100011011010001000011000110000100011000110000"
"2000100010010110001000011000100001100001000011111"
"2000100010001010001000010000100001100010000111111"
"2001100010010010001000010000110001100010000011111"
"2001100010010010001000010000110001100011000011111"
"2000000011001110000000010000110001100011000111111"
"2001100010011110011000011000110001100011000111111"
"2000000011110110011000011000110001100000000010000"
"2001100011110110001000010000110001100010000010111"
"2001000011111010001000001000010001100001000010000"
"2000000011011010001000001000110000100011000011111"
"2000110010011110010000010000100001000010000111111"
"2000100010001110001000010000100001100011000011111"
"2001100011011010001000011000110001100010000110000"
"2000110010011010001000011000110001100001000011000"
"2000100010011010011000010000110001000011000010000"
"2000000000100001001000001000100011000000000011010"
"2000010000011000100000011000100000000010000100100"
"2000100011111000011000011000110001100011000011111"
"2001100111111000001000011000110001100011000011111"
"2001100111011010001000010000110001100011000111111"
"2000100111111010011000011000110001100011000011111"
"2001110111111000011000011000110001100011000011111"
"2000000011000011100100100100010100011010001000001"
"2000100011111010001000011000110001100011000011111"
"2001100011111000001000010000110001100001000011111"
"2001100011111000001000011000110001100001000011111"
"2001100011111010001000011000110001100011000011111"
"2000000111111000001000011000110001100011000111111"
"2000100011111110001000011000110001100011000011111"
"2001100011110000010000110001100001000011110000000"
"2001100011111000001000011000110001100011000111111"
"2000000011000011100100100100010000000010001010001"
"2000000011000011100100000100010000000000001010001"
"2001000011000011100010000000010000001010001000000"
"2000000011000011100100000100010000000010001010000"
"2000000011000011100100010100010100000000001000000"
"2000000011000110100100010100010100001010001010000"
"2000000011000010100100010100010000000010001010000"
"2001000011100011100100010100010000000010001000000"
"2001000011100011100100010100010000000010001010000"
"2000000011000011100100000100010100000000001010001"
"2000000000000011100010100100100100010010011011001"
"2000000011100010100100100100000100010011001011001"
"2001000011100010100100100100010000010010011001001"
"2000111001111010001000001000010000100001001011111"
"2000010001111010001000011000010000100001000011111"
"2000010001111010001000001000010000100001000011111"
"2010000011000100100100100000000010010010010000001"
"2000000000111000001000011000100001000011000111110"
"2000010101111101001000001000010000100001000011111"
"2000010001111010001000001000010000000001000011111"
"2000100011011010001000001000010001000010000111111"
"2001110011011010001000001000011001100010000011111"
"2000100011001010001000001000011001100010000111111"
"2000110011011010001000001000011001100010000010000"
"2000100011011110001000001000010001000110000111111"
"2000010000001000001000010000100000000001000011110"
"2001100010001010001000001000011001100010000111111"
"2000100000001010000000001000000000000010000100000"
"2000100010001100000000001000010001000000000100000"
"2000100010001000000000000000010001000000000100000"
"2000100010001100000000000000010001000100000100000"
"2000100000001000000000001000010001000010000000000"
"2000100010000100000000001000010001000100000100000"
"2000100010000100000000000000010001000100000100000"
"2000100010001100000000001000100001000110000100000"
"2000010100101001001000010000100000000001000010000"
"2000100010001000000000001000010001000010000100000"
"2001100010001000001000001000010001000000000100000"
"2000100000001100000000001000000001000100000100000"
"2000100010001000000000001000000001000010000100000"
"2000000011000010100100100100100100010010011010001"
"2000000001000011000000100000100000010010011010001"
"2000000011001010101100101100001110011010011010001"
"2000000011000011000100100100100100010010011000000"
"2000000000101001001000010000100001000011000010000"
"2000000011001010101100101100001100011010011010001"
"2000000011000001000000101100101000011010011010001"
"2000000011000011000100100100100100100010011010001"
"2000000011011010001000001000110001100010000111111"
"2000100011011010001000001000110001100010000011111"
"2000000011001010100100011100010100011010001010001"
"2000000011001010101100100100011100011010001010001"
"2000000011001111101100101100111100011010011010001"
"2001000011101010101010101110011010011010001010001"
"2001000011101011101010101110011010011010001010001"
"2000010000101000001000011000100000100001000011110"
"2000001011001111101100101100011100011110001010001"
"2000000011000011100000100100010010011010001000001"
"2001110011011010011000011000010001100001000011101"
"2001100110010000001000010000100001000000000111110"
"2000110011011100011000011000010000100001000111111"
"2000000111010110001000011000010000100001000011101"
"2000000011010110001000011000100001000001000110111"
"2000100011010000001000010000100000100001000011111"
"2000000011010110001000010000010000100001000011000"
"2000000000101000001000010000100000000001000011100"
"2000100011011010001000001000110000100001000111111"
"2000100011001000001000001000110000100001000011111"
"2001110011001010001000001000110000100001000111111"
"2000100011110000001000010000010000100001000011011"
"2000000110010000010000010000100001000010000111111"
"2001110011001110001000001000010000100001000111111"
"2000100011001010001000001000010000100001000011000"
"2000000011001010101100100100011110011010001010001"
"2001000011100010101100110100010100011010001010001"
"2000010000101000001000011000110000100001000011110"
"2000000011100010100000000100010010011010001010001"
"2001000011100010100000100000010010011010001010001"
"2000000011001010101000101000010010011010001000000"
"2000110011011010001000001000011001100010000010011"
"2000100010011110001000001000010001100010000111111"
"2000110010001010001000001000010001000010000011111"
"2000000011011010001000001000010001000010000011111"
"2000110011011010001000001000010001000010000011111"
"2000100011001010000000001000011001100010000011111"
"2000010000001001001000010000100000000001000010000"
"2000100010001010001000001000011001100010000011111"
"2000100010001010001000001000010001100010000111111"
"2000000011011010001000001000010001100010000111101"
"2000100110011000001000001000110001100010000111111"
"2000100011111110001000001000011000100001000111001"
"2000100010011000001000001000010000100010000010000"
"2001110010001000001000001000010001100011000011111"
"2011110010011000001000011000010001100011000011000"
"2000100010001110001000001000010000100011000111111"
"2000001000100000000000001000010000100001000011110"
"2001100110011000001000011000100001000010000111111"
"2001100011111000001000010000110000100001000011000"
"2000000011011110001000001000010000100001000011000"
"2001110011011010001000011000110000100011000111111"
"2000000011110010001000011000110000100001000011111"
"2001110011011010001000011000110001100011000111110"
"2000000011011110001000011000010000100001000011111"
"2000000011111010001000001000110000100001000011111"
"2000100001111001001000011000010000100001000111111"
"2001000010011000001000011000110001100001000011101"
"2001100010001110001000011000110010000110001101111"
"2000100011011010001000001000010000100001000111100"
"2000100011011010001000011000010001100011000011111"
"2010000101000000000010100010000000010110001100000"
"2110001010011010101100001000010000000001000010000"
"2011000011100010100000010000010000011010001000000"
"2000000011000010100100010000010010000010001000001"
"2001000011100000100100110100010000010010001010001"
"2001000011100010100000100100010000010010001010001"
"2001110011011010001000001000010001100011000111111"
"2000100011011010001000001000010001100011000111011"
"2000000011000100100100100000000010010010010000001"
"2000100010001110001000001001100010000010001101111"
"2000000011001010001000001000010000100011000010000"
"2000100010011010001000011000110001100011000010000"
"2000100011011010001000001000010000100011000010000"
"2000110011001010001000001000010001100011000011111"
"2001110010001110001000011001100010000111001101111"
"2000000011011010001000001000010001100011000010001"
"2000100011011010001000001000010001100011000011011"
"2001110010011010001000011000110001100011000010000"
"2000000011001011101000101100111000011010011010001"
"2000000011000011100000100000010000010011001011001"
"2000000011000011100100100100010000010011001001001"
"2000000011000011100000100100110000010011001011001"
"2011000011000011100100100100010010010010001010001"
"2000000011000011100100100100010000011011001011001"
"2001110110001110001000011000110010000100000101111"
"2000000011000011100100100100010000010010001000001"
"2001000011000010101010101110011010011010001010001"
"2000100011111110001000011001110011000010000011111"
"2000100010001000001000001000110001100010000110011"
"2000100011111110001000011000110011000010000011111"
"2000000011110110011000011001110011000010000111111"
"2000100011110110011000011001110011000010000111111"
"2000000011111110011000011000110011000011000111111"
"2001100011111110001000011001110011000010000011111"
"2000000011001000101100100100010100011010001010001"
"2001100011111110011000011001110011000010000011111"
"2000100011110010011000011000110011000011000011111"
"2001000011000010100100010100010100000000001000001"
"2000000011000010100000100100010100001000001000000"
"2000000011000010100100010100010100010000001000001"
"2000000011100010100100010100010100000000001000000"
"2011000011001010101000101000011010011010001010001"
"2000000011000010100100010100010100001000001000001"
"2000000011000000100100010100010100010000001010001"
"2000000011100010100100010100010100001000001000000"
"2000000011100000100100010100010100000000001000000"
"2000000011001000101000101100011010011010001000001"
"2001000011100010100100010100010100000000001000000"
"2000000010000000100100100100100100100000100010000"
"2011000000000100100100100100100100100000100010000"
"2000000011000000100100100100100100100000100010011"
"2001000010000000100100100100100100100000100010010"
"2001000010000100100100100100100100100100100010000"
"2001000010000100100100100100100100100000100010000"
"2000000011000010101100101100011010011010001010000"
"2001000010000000100100100100100100100000100010000"
"2011000010000000100100100100100100100000100010000"
"2011000010000100100100100100100100100000100010011"
"2000000011000100100100100100100100100000100010010"
"2000000010000100100100100100100100100000100010010"
"2011000010000000100100100100100100100000100010011"
"2000000011000100100100100100100100100000100010000"
"2000000000000100100100100100100100100000100010010"
"2011000011001010101100101100011110011010001010001"
"2010000011000000100100100100100100100000100010011"
"2001000011000000100100100100100100100000100010011"
"2011000010000100100100100100100100100000000010010"
"2011000011000100100100100100100100100000100010000"
"2010000011000100100100100000000010010011010000001"
"2000000011101010101000101100011110011010001010001"
"2001000010000000100100100100100100100000100010011"
"2011000000000100100100100100100100100100100010000"
"2011000010000100100100100100100100100000100010000"
"2011000010000000100100100100100000100000000010011"
"2010000000000100100100100100100100100100100010000"
"2001000010000100100100100100100100100100100010011"
"2001000000000100100100100100100100100000100010011"
"2001000000000100100100100100100100100100100010000"
"2001100010010000011000010000010000100001000011111"
"2011000000000100100100100100100100100100100010011"
"2001000011000000100000100100100000100000100010010"
"2000000000000000100100100100100100100100100000000"
"2011000000000100100100100100100100100000100010010"
"2000110011011010001000011000011001100011000111000"
"2000100010000000001000000000000001100010000000000"
"2001110010001010000000001000010000000010000100111"
"2000000011000010001000000000010000100010000011100"
"2001100010110100000000010000010000100001000011111"
"2000100010001000001000001000010000100000000000010"
"2001100010001000000000001000010000100010000100000"
"2000110010001100001000001000010000000010000100000"
"2001100010001100001000001000010000000010000000000"
"2000000010000000001000000000010001100010000000000"
"2001000010001010001000001000010000000010000100010"
"2000000010001000001000001000010000000010000010000"
"2000010010001000001000001000010000100010000000000"
"2001000011001010001000001000010000100010000000000"
"2001000011110100010000010000000000000001000011110"
"2000000000010000001000001000010000000010000100000"
"2000100010001000001000001000110000100010000000110"
"2000000010011000001000001000010000000010000100000"
"2000110010001000001000001000110000100010000000000"
"2000000000001000001000001000010000100000000010000"
"2000100010000010001000000000000001100010000011111"
"2000000010001010001000001000010000000011000000000"
"2000011000101000000000001001110010000111001000110"
"2000011000101001001000111001110011000111010000110"
"2000000100111001001000001001111011100011001000110"
"2001000111110100010000010000010000100001000111111"
"2000000000111001001000001000110011110111000001110"
"2000001000101101001000001000110011000011001101110"
"2000011000101001001000001000010011100011000101110"
"2000011000101001001000001001110011000111010101110"
"2000110001111011001010001000111011110111110111110"
"2000110001111000001000001001111011000110010001100"
"2100011000111001000000001001111011100111000101110"
"2000010000101001000000001001110011000111001001111"
"2100011100111100000000001001111011000111001000110"
"2000010001111001001000001001111011000111000001110"
"2001100011110100001000001000010000100001000011111"
"2000010001111001001000001000110011000111000001110"
"2000010001111101001100011000110011100111100110111"
"2000011000101001000000001000110011000111000001111"
"2100011000111001001000011001110011100111000101110"
"2000011000111001001000001001110011000111000100110"
"2000010000111000001000001001110011000111000000111"
"2000011001101001001001001001110011100011000111110"
"2000001000101001000000001001110011000011000100110"
"2000010000111001001000001011110010000110110011100"
"2001000111110100010000010000010001000001000111111"
"2000010100111001001000111001111011000111000001100"
"2100011100111100000100001001110011000111010001100"
"2000011100101000001000011001110011000011000001110"
"2000011000001000001000011001110011000111000101110"
"2000001000101000001000001000110001000011000111110"
"2000001000111101000000001000111001110011000101110"
"2000010001001001000000001000011010000111000101010"
"2001000011110100001000001000010000100001000011111"
"2001000011110000011000010000000000100001000011111"
"2001100010011000001000001000010000100001000010001"
)+string("3000000011010100101100101000100010000000000000001"
"3000010011011011111100100100000010001010011010011"
"3000011011111001000100000100000000000010011010001"
"3000011011111101100100000100000100000010000010001"
"3000010011111000000000000100000000000011011011011"
"3000010011111011101001000000000000000010000010011"
"3001110010001110001000010000001000001010001010011"
"3001100010001010001000010000010000001010001010000"
"3000000000011000100011100001000100001100000000000"
"3000001010011001010001100100100000000010001000001"
"3000001000011000100011000100000000001000000000010"
"3000001000011010010101000000100000001010000001000"
"3000011011011011101100100100100110000010001010001"
"3000000000011000100101100001000000001010011000010"
"3001001000011110000101100101001100001000000011000"
"3000001000110000011000001000001010001010001011110"
"3001100010010000001000110000011000001010001010011"
"3010001000001000010000011000001000000010001001111"
"3001100010010010001000010001011000001110001010011"
"3001100011110110001000111000111000001010001011111"
"3001100011111010001000111000111010001110011011111"
"3001100011110110011000111000111000001110011011111"
"3000000011011011111000100000100010000010001010001"
"3001110011111010001000111000111000001011011011110"
"3001000011000011100000100000110000110000001000000"
"3000010011111011101000000000000000001010001010001"
"3000000010111011101000000100000000001010001010001"
"3000010011111011101000001100100000001010001010011"
"3000000010111011101000001100001000001010001010001"
"3001111000010000110000110000001000001010001011010"
"3001111000011000010000111000001000001010001011111"
"3011011001001001000001000000000000100000000011011"
"3000010011111011101100100100000000000010001010001"
"3001110000010000110000110000001000001000001010010"
"3000100000010000100000110000001000001000001010010"
"3011110000110001100000110000011000001100001010000"
"3000010011111010101100100100000000001010001010011"
"3000000011111010011000011000110000001110001011111"
"3001100011011010001000011000110000001110001011011"
"3000010011011011101000100100100000000010001010001"
"3000010011111011101100100100100100001010001010001"
"3000010011111011101100000100000100001010001010001"
"3000000010101101011000100000010000010000010110100"
"3000001000101101001000110000001000011000010110100"
"3001000000011000110000110000011000001000011111110"
"3000000011111000001000011000111000001110001011111"
"3001100011110000001000011000110000001010001011111"
"3001000011110010011000011000111000001010001011110"
"3001000011111000011000011001110000001010001011111"
"3001100011111000001000111001110000001110011011110"
"3000100011110000001000011000110000001010001011111"
"3001100011111000001000111000111000001010001011111"
"3001100011111000001000111000111000001110001011110"
"3011011011111000100100100100100100000000000000001"
"3011111111111100100100100100100100000000001000001"
"3011011011111100100100100100100100100100001000001"
"3011011011111100100100100100100100100100000000000"
"3000010010011101100100000100000100000000000010000"
"3000000010011001100100100100100000100010000010000"
"3000000010011010101100100100100000100010001011001"
"3000010011011001100100100000100000000010000010000"
"3000010011111011111111101110101110001010001010001"
"3100011100000000010101001100000000000000000000001"
"3100011110000100110101000100000000000000000000001"
"3000011110000000110101001100000000000000000000001"
"3100001110000110110101001100000000000000000000001"
"3100000110000110110101001100000000000000000000001"
"3000000111110010011000111011101000001000001011111"
"3001000011110000011011111000011000001000011011110"
"3000010001101001001000001000110000000100010010110"
"3000010001001000001000001000110000000100010010100"
"3000010001001000001000010000110000000100010110100"
"3000010001001001001000011000010000000100010110110"
"3001100110001000001000001001111000001000001011011"
"3001110110001000001000001001111000001000001111011"
"3001100110001000001000001001110000001000001010110"
"3011110100011000001000001001110000001000001110011"
"3001000100011000001000001001110000001000001100010"
"3001100110001000001000001001111000001000000011111"
"3001110010001000001000001001111000001000001010001"
"3000100010001000001000001001110000001000001010010"
"3011110110001000001000001001111000001000001110011"
"3011110110011000001000001001110000001000001110011"
"3001100110011000001000001001111000001000001110011"
"3011110100011000001000001001111000001000001110011"
"3001110010001000001000001001111000001000001110001"
"3001110010001000001000001001111000001000001010011"
"3011110110001000001000001001111000001000001111111"
"3011110110001000001000001001110000001000001111011"
"3000000011011000001000001001111000001000001110011"
"3001110011001000001000001001111000001000001011011"
"3000000010001000001000001000111000001100001011011"
"3001110010001010001000011000011000001010001011111"
"3001110010011110001000001000111000001100001011111"
"3001110010001000001000011000011000001010001011111"
"3000100010001110001000001000111000001100001010011"
"3000000010011110001000001000110000001100001010011"
"3010000011010100101100101000100010000000000000001"
"3000011011011011111100100100100010001010001010001"
"3000100011011010001000001000111000001010001011111"
"3001110010011010001000011000111000001010001011111"
"3000011001101000001000011001110000011110010111100"
"3000000001001001001000010000110000010010010011100"
"3000100110011000011000011001110000011000011011011"
"3001000010011000001000001001111000001000001011011"
"3000000110001000000000001000011000000000000010001"
"3000000010001000001000001000010000001000000010001"
"3001100110001000001000001000011000001000000010001"
"3000000010000000001000001000011000001000000010001"
"3000000110001000001000001000011000001000000010011"
"3000100010001000000000001000011000001000000010001"
"3111000000011000110000100000100000011000001001111"
"3000000010100000000000100000100000001000001000111"
"3001000111111000010000100000100000011000001001110"
"3111100000100000100001100000011000001000001001110"
"3001100111110000010000100000111000001000001111110"
"3011110100001000001000011001111000001100001011111"
"3001100110001000001000001001111000000010000011111"
"3000000110011000001000001001111000001000001011111"
"3000100010001010001000001000111000001100001011011"
"3000010011111011101000101100000100001000001010011"
"3000011011111011100000100100100100000000001010001"
"3000010011111001100000100100100100000010001010011"
"3000010011111000101100100100000100000010001010011"
"3000010011011010101000100100100000000010001010001"
"3000010011111001101100000100000000000010001000001"
"3000000010011011101000100100000100000000001010011"
"3000010011111100101100000100000100000010001000001"
"3000010011111000100100000100000100000010001010011"
"3000000011111001101000100100000100000010001010011"
"3000110010001000001000001000011000001000000010001"
"3000010011111001101100000100000100001000001010011"
"3000000010111011101100100100000100000000000000000"
"3000100010011000001000011000111000001010001010011"
"3000110011001000001000001000110000001000001011001"
"3000110010001000001000011000011000001110001011111"
"3000100010001000001000011000010000001000001011111"
"3001110011011000001000011000110000001000001010011"
"3000100010001000001000011000111000001000001011011"
"3001110010001000001000000000011000001000001011111"
"3001110010011000001000111000010000001000001110011"
"3000000010001000000000001000011000000000000010001"
"3000000010011000001000001001110000001000001010001"
"3001100010011000001000011000110000001100001011011"
"3000100001010000001000000000110000001000001011110"
"3001100111011000001000010000110000001100001111110"
"3000100010001000001000011000011000001000001111011"
"3000000011111010001000111000011000001110001011110"
"3011100110110100011000110011110000001000001111110"
"3000100011110100001000111000011000001100001011111"
"3000000010011010011000010000111000001110001011111"
"3000000110011000000000001000011000000000000011011"
"3001100011011110001000011000011000001110001011111"
"3001110010001000001000001000111000001000001011011"
"3001100010011000001000011000110000001000001011010"
"3001100110011000001000011000110000011000001011110"
"3011100110011000001000011000110000011000001110011"
"3011110110011000001000001000111000001000001011111"
"3001100010010000001000001000110000001000001010010"
"3000100011111010011000111001110000011110011011111"
"3001110011111111011000011001110000011110011111111"
"3001110011011000001000111000111000001110001011111"
"3000100110001000001000001000110000001000000010011"
"3001000011111011011000011001110000011110011011111"
"3000000011111111111111111110101110001010001000000"
"3001010011111011111011101100101110101010001010001"
"3011010011111011111010101000101010101010001000001"
"3001011011111011111100100100100110101010001010001"
"3000010010110001001001001101000001000000000010000"
"3000010000111001001011000011000101000000000010000"
"3000110000111000101011001101000001001001000010000"
"3000010000011000100011000011000000000100000100000"
"3000100110010000001000001000010000001000000011011"
"3000010000111000100000000011000000000100000000000"
"3000010000111011101011001001000000000000000010000"
"3000000000110000110011001011001101001001000000000"
"3010010110111101001101000101000001000000000001000"
"3000000000110011000111001001001001000000000000000"
"3000010000110001100011001111000100000100000100000"
"3000010000111001101011000011000001000101000000000"
"3000010000111000101111000101000101000001000010000"
"3000010000110011101011001001000001000000000000000"
"3000100001110001000011001001000001000010000010000"
"3000100000001000000000001000010000001000000010001"
"3000110000110011001011001111000010000010000010000"
"3000010000110011101011001001000000000000001010000"
"3000100000001011001000010000010000011010010010010"
"3000110010001010001000011000010000001110001100010"
"3000110000001010001000010000010000001010000000010"
"3000110000001011001000010000010000001110001010010"
"3010000011010100101000101000100010000000000000001"
"3000000011001000000000000000010000001000000110001"
"3000100000001001001000010000000000001110001010010"
"3001100011111010001000010001111000001110001011110"
"3000010010011011101000100100100000100010001010001"
"3000110011110110000000110001110000001010001011110"
"3000100111110000011000110000111000001000001011111"
"3000100011111010001000011001110000011010001011110"
"3000100011110000010000110001111000010000011011110"
"3000100011110010011000111000111000001011011011111"
"3000000010011010001000011001110000001100001011011"
"3000010011110010011001110000011000001011011011110"
"3000100011001010001000011000111000001110001011111"
"3000100011010110001000010000110000001100001011010"
"3000000011001000001000010000110000001010001011010"
"3001100010011000001000010000110000001110001010011"
"3001110010001110001000011000110000001110001011111"
"3001100010011010001000011000110000001110001011111"
"3001110011011010001000011000110000001110001011011"
"3000100011001010001000011000010000001010001001111"
"3001110010010010001000010000110000001110001011110"
"3000000011110110011000010001110000001110001011110"
"3001110011011010001000011000110000001110001011111"
"3000100011011000001000011000110000001110001011111"
"3001100111001100001000011000110000001110001011111"
"3000110010010010001000010000110000001110001010001"
"3001100011011100001000011000011000001110001011111"
"3000100011011010001000011000110000001110001010011"
"3000100011011010001000011000011000001110001011110"
"3000100001111010001000011000011000001010001011111"
"3001111010001100001000011000110000001110001011111"
"3001100010001000001000011000110000001100001011111"
"3001100010011010001000111001110000001100001011011"
"3000000001001000001000011000011000001010001011111"
"3000100011011000001000011000111000001110001011111"
"3010001001000110000001100010001000100000001000000"
"3000000001000111000011100010001010001000001000000"
"3011000011111000011001111000111000011000011011111"
"3010000011110000011001110001111000001000011011110"
"3011000111111000011001111001111000011000011011111"
"3000010010111011111000100100100000100000100000000"
"3000010010111011111000100100000000000000000000000"
"3000010010011011111000100100100100000000000010000"
"3000010000111011001011001001000000001000001000001"
"3000010010011011111101100100000100000000000000001"
"3000010011011011111100100100100100000000000000000"
"3000010010111011101101100100000100000000000000000"
"3000110100111111001111000101000100001100001000001"
"3000010010111011111101100100000100000000000000000"
"3000010011111011101101100100000100000000000000000"
"3000010011111011101100101100101100001010001010011"
"3000010011111010101100100100000100000010001010011"
"3000010011111011101000100100100100000010001010011"
"3011000000001000000001000000010000000000000100010"
"3011000000001000000001000000010000000000000110010"
"3011110000000000100001000000010000001000001100010"
"3000011000001000000001000000010000000000000100010"
"3011111000001000100001000000010000000000000100001"
"3011101000001000000001000000010000001000000100010"
"3010000000001000000001000000000000000000000100001"
"3011111000000000000001000000010000001000001000010"
"3000010000111110001111000101000100000100001000001"
"3011110000000000100000000000010000001000001100010"
"3011111000010000100001000000010000001000001000010"
"3000110000000000100001000000010000000000000000000"
"3011100000010000100001000000010000001000001000010"
"3011111000000000100001000000010000000000000000000"
"3000000011010010101100101100101010101010001000000"
"3000010011111011101010100100100110001010001010001"
"3111100000010000000001110000011000001000001111110"
"3000000100111100001100110000110000001000010010110"
"3000000000010000001000001001111000001000001011110"
"3000010001001011001000011000010010011110010110110"
"3011100000011000001000000000010000001000001011110"
"3011100000010000001000001011110000001000001011111"
"3000011011111011101100100100100010001010001010001"
"3000010000011011101110100100100100101010011010011"
"3000011011111011101100100100100100000010001010000"
"3000010000011011101101001100000100001010001010001"
"3000010000101000001000010000010000010000010010100"
"3000010011111011101000100100100000001010001010001"
"3000011011011011101100100100100100000010001010001"
"3000010011111011101100001100001000001010001010001"
"3000011011111011101010100010100010000010001000001"
"3000010011011011101010100100100010001010001010001"
"3000000011111011101110100100100100100110001010001"
"3000010001111011111110100110100110100010001010001"
"3000100011001010001000001000011000001110001011111"
"3010000011010100111000101000100010000000000000001"
"3000010100001100001100110000001000001000011010110"
"3000000010011010001000001000111000001100001011011"
"3000100010001010001000001000111000001000001011011"
"3000100011011010001000001000111000001010001010011"
"3001110000010000100000110000001000001000001011111"
"3001000001111000110000110000001000001000001011111"
"3001010000110000110001100000011000001000001111110"
"3001110010011010001000001000110000001110001010011"
"3000010011011011111100100100100000000010001010001"
"3000011000100000001000010001010000010000010010100"
"3011010011110000110001110000011000001010001011110"
"3010110111111000110001110000011000001010001011111"
"3011100011111000110001110000011000001110001011111"
"3000010010011011101100100100100100100000000000000"
"3000010011111011101100100100000100000000000000000"
"3000010011101001101100100100100100100100100000000"
"3000010011011111101100100100100100100100100000000"
"3000010011011011101100100100100100100000000000000"
"3000010011011011101100100100100100100000100000000"
"3000010010011011101100100100000100000000000000000"
"3000010000101000001000110000001000001100010011110"
"3001010011001100100100100100100100000000000010001"
"3000010011001000100100100100100100000000000010001"
"3011010000001000100100100100100100000000000010001"
"3001010010001000100100100100100100000000000010001"
"3011010010001000100100100100100100000100000010001"
"3011010000001100100100100100100100000100000010001"
"3011010010001000100000100100100000000000000010001"
"3011010010001000100000100000100000000000000010001"
"3011010010001000100100100100100100000000000010001"
"3001100010011010011000011000011000001010001011011"
"3000010011001000100100100100100100000100000010001"
"3000000000001100100100100100100100000000000000001"
"3011010010001000100000100100100100000000000010001"
"3001000111110000011000010000111000011000011011110"
"3001000011111010011000011001111000001110001011011"
"3000000011011000100100100100100100100010001010011"
"3000010010011011101010101100100110101010001010001"
"3000000011000010000000001001110000001100000011001"
"3000001010100100001000110000010000010001110000000"
"3000010100111001001000111001110000010001110111000"
"3000001010011100001000111000110000000011110100000"
"3110010001001000001000110001110000010001100111000"
"3000001010111100001100111000110000010011110110000"
"3000011010111100011100110001110000010001110110000"
"3001001111011010011100110000010000110011110110000"
"3110001110111010101100111001110000010011110110000"
"3000010011111001101100100100100100001010001010001"
"3100001100000101100101100000101011111000100011000"
"3100000101110100010001100000000000110011101000000"
"3110011100101100001001110000110000010111110100000"
"3000100000100000111011100001100001100110000000011"
"3100011000101000011000111000011000110011100100000"
"3000011010001100001000110001110000000001110100000"
"3000111100100000001000110001110000010001110010100"
"3000011000101100001000110000010000010011100100000"
"3000001000100000000100111000110000110011100100000"
"3000010011111010101000101010101010001010001010001"
"3000001100111000001000110000010000010011110100000"
"3000011000101000011001110000010000010011100100000"
"3000010100101000001001111000010000010011110100000"
"3000000100111000001000110001010000010011110100000"
"3000001000000010001110111000110000000011110100000"
"3001001000010010001100111000000000010011110100000"
"3000001010001000001000010000101000000001110000000"
"3000000000111011101010100100100110001010001010001"
"3001100011111000011000010001111000001000001011110"
"3001100010111000010000000000111000001000001011110"
"3010000111010101111100101100100010000000000000001"
"3001100010111000011000010000011000001000001111110"
"3001100010111000001000010001111000001000001111010"
"3001100010011000011000010000111000001000001111110"
"3000100011111000011000010000111000001000001111110"
"3001100010010000010000000001110000011000001010010"
"3001100010111000011000010000110000001000001111110"
"3001100011111000010000010000111000001000001111110"
"3000100011111000001000000001111000001000001110011"
"3001110010011000001000010001111000001000001110010"
"3000000010111000011000010001111000001000001110010"
"3001100110011000011000010000111000001000001110011"
"3000000010010000010000010000111000001000001010010"
"3001100010010000010000010000111000001000001110010"
"3000000001001001001000010000110000010110010011100"
"3000010001001001001000010000110000000110010011110"
"3000010001101001001000001000110000001110011010110"
"3000110001011000001000010000110000010100010010100"
"3000010001001000001000010000110000010110010010100"
"3001110000010000010000110000001000001000001010011"
"3011111000011000010000110000001000001100001011011"
"3011111000010000010000110000001000001100001010011"
"3001110000011000010000110000001000001000001010011"
"3000110001001011001000010000110000010110010011100"
"3000010001001001001000010000110000010110010011110"
"3000000001111010001000010001110000011100010111100"
"3000100011111000111000110001111000011000011111010"
"3000110011111000011000010000111000011000001011010"
"3000000010110011111111100100100000001000001000001"
"3000110010110011111111100100100000001000001000001"
"3000010011111011111111100010100000100010001000001"
"3000000010111011111111100100100000001000001000001"
"3000010010110011111011100010100000100010001000001"
"3000000010110011111111100110100000001010001010001"
"3000010010111111111111100100100000001000001010001"
"3011111001110000110001110000001000001010011011110"
"3000010011111011111100100100100000000010001010001"
"3001111000011000110000111000001000001110011011110"
"3001110001111000110000111000001000001110011011111"
"3001000000011000010000111000001000001010011011111"
"3001110001111000010001111000001000001110001010111"
"3001111000011000010000110000001000001110001011111"
"3001110000110000110000110000010000001110001011110"
"3011000011111100111100100100100010000010001000001"
"3010000011011100111100101100100110000010001000001"
"3010011011101101000100000100000100000010000010011"
)+string("4000000000010000110001010010010011111000010000010"
"4000000111010011110010011001010000110000110000010"
"4100000011010011110010111001010000110000110000010"
"4100000111010011110011011001010000110000110000010"
"4000000111010011111010111001010000110000110000110"
"4100010111010011110010111001010000110000110000010"
"4100010111010011111001011001010000110000110000010"
"4100010111010011111011011001010000110000110000010"
"4000000000010000010111111010010001010000010000010"
"4000010000110000110001010010010000010000010000110"
"4000000001110000111000111000100000100011100011100"
"4100000010010011111000011001010000010000110000010"
"4000000000000001111001111000100110100011100001100"
"4000000000000000111000010100010000010001010000110"
"4000001000111000000000000000010000010110000011100"
"4000010000010000110001010000010010010000010000010"
"4000010000010000110001010000010011011000010000010"
"4000010000010000110001010010010010010000010000010"
"4000010000010000110001010010010000010000010000010"
"4000010000110000110001010010010011111000010000010"
"4000010000010000110001010010010111111000010000010"
"4000000011010010111000011001010000110000010000010"
"4000010000010000110001010010010000010111011000010"
"4000010000111000111001111011011111111000111000011"
"4000000000010111111010010001010001010000110000010"
"4000000000011000111001111011011111111011111000010"
"4000000000011000111001011011011011111011111000011"
"4000000000110001110001110010010111111000010000110"
"4000000000110000110001110010010010110000010000010"
"4000000000110001110001110001010011111000010000000"
"4000010000110000110001010010010111111000110000010"
"4000010000110001110011110010110111111000110000110"
"4000010000110001100010010110010111111000000000110"
"4000000000110001110011110010110110110000110000110"
"4000010000110000110001110010110111111000110000110"
"4000010000110000110001110000010111111000010000010"
"4000000000100000100001100010100110110000100000100"
"4000100000110000100000100010100011111000110000110"
"4000000000010000110001110001010111111000011000010"
"4000000000100000100001100001000010000011111000010"
"4000000111111111111010010001010001110000110000010"
"4000110000110001110001110010110010110000110000110"
"4000010000110000110001110010010111111000011000011"
"4000010000010000110001110011010111111000010000010"
"4000010000110001110001010011010011111000010000010"
"4000010000110000110000110010010110010000010000010"
"4000010000010111111111111010010001010000110000010"
"4000010000010111111010010011010001010000110000010"
"4000010000010111111111111011010001010000110000010"
"4000010000010111111010010010010001010000110000010"
"4000010000110001110001010010010011111000010000010"
"4000010000110001110001010011010110010000010000010"
"4000010000010111111110010011010001010000110000010"
"4000010000110000110001010010010011111011011000010"
"4000010000110000110001111011010101101001111000010"
"4000000011111111111010010001010001010000110000010"
"4000010000110000110001110011010111111001110000010"
"4000000000010111111010010000010001010000110000010"
"4000010000010111111010011000010001010000110000010"
"4000000000010000110001010010010010010000010000010"
"4000010000010111111010011011010001010000110000010"
"4000000000010000110001010011010011111000010000010"
"4000000000010000110001010001010011111000010000010"
"4000100000100001100001000011010011011111111000011"
"4000000000100001100001000001000011011011111000011"
"4000010000110001110001010010010110010111111000010"
"4000010000110000100001000001001011011011111000011"
"4000100000100001100001000011011011011111111000011"
"4000000000100000100001000001001011011011111000001"
"4000000000100001100001000011000010011111111000011"
"4000100000100000100001000010000011011111111000001"
"4000010000110001010010010010010111111000010000010"
"4000000000110001010010010010010111111000010000010"
"4000010000110001110001010010010110011111111000010"
"4000000000110001010011010010010111111000010000010"
"4000010000110001010001010010010111111000010000010"
"4010010111111111111010000001000001100000100000000"
"4000000010110111111011010011000001100000100000000"
"4000010111111111010011010001110000110000100000000"
"4000000111111111010011000001000001100000100000000"
"4000000000000111111110000000000001100000100000010"
"4000000000000111111110000010000001000000110000010"
"4000010011111011111011111001010001110000110000110"
"4000010000010000010001010010010010010000010000010"
"4000000000011000110001011011011010010111111000011"
"4100000011010010100001010001001000000000100000000"
"4000000011010001011000010000010000010000010000000"
"4000000010010011110000010001010000010000110000010"
"4100000010010010110000010001011000010100110000010"
"4000000010010011110001011001011000010000110000010"
"4110010011010010110000010001010000110000110000010"
"4100000010000011100000010001001000100000100000000"
"4000000010010011100000010001001000010000110000010"
"4000010000011000101001001010001111111000001000001"
"4000001000011000111001001011001111111000001000001"
"4000000000010000110001010011010010010111111000010"
"4000010000111001101001001010001111111000001000001"
"4000001000011000101001001011001011111010001000001"
"4000001000111001101011001111111011111000011000011"
"4000001000111001101011001111111011111000001000001"
"4000001000011000011000101001001011111000011000011"
"4000011000011000101001001011001011111000001000001"
"4000000000111000111001001010001111111000001000001"
"4000000000111001101011001111111001111000001000011"
"4000011000111001101011001111111000011000011000011"
"4000000000011000100001000010000000001001010100000"
"4000001000011000111001001110001111111000011000011"
"4000000000011000111001001011001111111000001000001"
"4000000000111001011011011111111000011000001000001"
"4000000000011000111001001011011011111000001000001"
"4000000000011000111001001111111000011000001000001"
"4000001000011000111001001011111011111000001000001"
"4000010000011000111001011010011111111000001000001"
"4000010000111001111011001111111011111000011000011"
"4000000000011000101001001011001011111000001000011"
"4000011000111001101011001011111011111000001000001"
"4000010000111000111001011011010111111000010000011"
"4000001000011000111001101011001111111000001000001"
"4000011000011000101001001011001111111000001000001"
"4000001000011001101011001011001011111000001000001"
"4000001000111001101011001111001011111000001000001"
"4000010000111001111011011010011111111000001000001"
"4000001000011001111011011011011111111000011000011"
"4000000000111001101011001111111001101000001000001"
"4000000000111001101011001011011111111000001000001"
"4000010000110001110011010010011111111000010000010"
"4000000000110000110011010011011111111000010000010"
"4000010000110001110001010010010110011000011000010"
"4000010000010111111010010011010001110000110000010"
"4000000000010111111111111010000011010001110000100"
"4000000000011011111110010011010001010000110000110"
"4000001000011000101001010000010111010000010000010"
"4000000000011000101001010000010100010000110000010"
"4000000000011000001000010000010010010000010000100"
"4000001000011000011000001000010010010000010000010"
"4000000000001000011000001000000011111000010000010"
"4000001000011000101001010000010010010111110000010"
"4000001000011000101000001000010010010000010000010"
"4000010000110001110001010010010110011000010000010"
"4000000100001000011000001000000011111000010000010"
"4000010000110001110001010010010111111000010000010"
"4000010000110000110001010010010111111000010000010"
"4000010000110000110001110010011111111000010000010"
"4000010000110000110001010010010010010000010000010"
"4000010000010000010000010010010010010000011000010"
"4000010000010000010001010010010011111000010000010"
"4000010000110000110001101011010110010000011000010"
"4000010000110000110001010010010010111000010000010"
"4000001000011000111001111001010011111000010000110"
"4000000000011000111001011011010111111000110000110"
"4000001000011000101001110001010011111000010000110"
"4000000000011000111000101001010010010011111000110"
"4000000000011000011001111001010010011111111000110"
"4000001000011000111001011010010011111000110000100"
"4000000000011000111001111001010011111000010000110"
"4000001000011000111001111010010111111000010000100"
"4000010000111000111001010011010111111000001000001"
"4000010000110001110001110011011111111000010000011"
"4000010000110001110011110011010011111000010000010"
"4000010000010000110001110011010011111000010000010"
"4000010000010001110001111011011011111000011000010"
"4000000000110001110011110010010111111000010000010"
"4000010000110001110001110011011111111000010000010"
"4000000000010000100001000010000000001011111000001"
"4000000000010011111111111011010001010001110000110"
"4000001000011000101001001010001010001010001000001"
"4000000000010000100001000000000010001000001000001"
"4000010001010011010011010111111000011000011000001"
"4000100001000011010110011010011000010000011000011"
"4000010000110000001011001111001000001000001000001"
"4000000000011000101001001011111000001000001000001"
"4000010000110001010011011111111000010000010000010"
"4000000000011000111001111011001011111000001000001"
"4000100001110001010011010111111000011000011000011"
"4000000000111001111011011111011000011000010000010"
"4000000001010011010010010011111000010000010000010"
"4000010001010000110010100101111000100000100000100"
"4000100000110001111011011011011000001000001000001"
"4001100011100010010010010111111010011000001000001"
"4000001000011000101001001010011000010000010000000"
"4000010000111000001011001111111000001000001000001"
"4000000000111001101000001010011000010000010000010"
"4000000000111001101001001110011111011000011000001"
"4100000111010010110000011001010000010000010000010"
"4000010000110001110011010110010011111000010000010"
"4000010000111001001011111000001000001000001000001"
"4000000001000010001011111010001000001000001000001"
"4000001000101001001010001111111000001000001000001"
"4000100000101011001111111000011000011000010000010"
"4000000000111001001010001111111000010000010000100"
"4000101001001010001111111011111000010000011000010"
"4000000001000000001010001110011000001000001000001"
"4000010000110001010000010000010110010011111000010"
"4000010000110000110001010010010111111000010000011"
"4000000000110000110001010010010111111000010000010"
"4000000000100001100011000010000011111000000000010"
"4000010000110000110001010010010010110011010000010"
"4000010000110000110001010000010111111000011000010"
"4000010000010000110001010011010010011000010000010"
"4000010000110000110001010010010010011000010000010"
"4000000000110000110001010010010111011000010000010"
"4000000000010000010111111011010001010000110000010"
"4000000000010000110001010010010011111000011000010"
"4000010000010000110001010000010110011000010000010"
"4000000000110000110001010010010110010000011000010"
"4000010000110000110001010010010010010000011000010"
"4000010000110000110001010010010011111010010000010"
"4000010000110000110001010010010010010111111000010"
"4000010000010011111010010010010001010000110000010"
"4000010000010111111110010010010001010000110000110"
"4000010000010111111110010000010001010000110000010"
"4000010000010111111010010000010001010000110000010"
"4000000100011100001101110001010111111000100000100"
"4000010000010111111010111010010001010000110000110"
"4000000000000111111010010010000001010000110000110"
"4000010000010111111010010010010001010000110000110"
"4000110000110001010011010010010100010000011000010"
"4000010000110000000001010010010010010011111000010"
"4000010000110001110000010010010110010000010000010"
"4000010000010000110001010000010010010011111000010"
"4000000000110000010001010010010110011000011000010"
"4000010000110001010001010010010110110000010000010"
"4000000000001000001000000001010010000000111000100"
"4000010000110001010001010010010110010000010000010"
"4000010000010000110001010010010010010111111000010"
"4000010000110000010001010010010010010011111000010"
"4000010000110000010001010010010110010000010000010"
"4000000000010000110001000010000111111000011000000"
"4000010000010000110001010010010111111111111000010"
"4000000000001000011000100001010010000000111000100"
"4000010000110001110010010110010111111100110000010"
"4000010000110000110001010010010011111011111000010"
"4000000000010000110001010010010111111000111000010"
"4000010000110000110001010010010111111111111000010"
"4000010000110000110001110010010110010011111000010"
"4000000000001000011000100001010010010111111000100"
"4000010000010000110001011010010111111011111000011"
"4000010000010000110001010010010111111001010000010"
"4000000000010000110001010011010011111011111000010"
"4000010000111000111001011010001010011000011000001"
"4000000000011000111001000011010011110000100000100"
"4000010000011000111001001011001010011011111000001"
"4000000000011000111001011011011110011000011000011"
"4000000000110000110001010011010110011111111000011"
"4000011000111000111001001010001111111000011000001"
"4000010000111001111011011011011111111000011000011"
"4000010000110001110011010010010111111000010000010"
"4000010000110001110001010010010010111000010000010"
"4000111001111001111011111010111111111000111000111"
"4000000000001000011000101001000010010111111000100"
"4000010000111001111001011010010111111000011000011"
"4000010000111000111001111010011111111000011000111"
"4000010000110001110001010010010111111000011000010"
"4000000000110001110001011010011111111000011000011"
"4000010000010111111111111011110001110000110000010"
"4000010111011111111111111011010000110000110000010"
"4000010111111111111111111011010001110000110000010"
"4000000000001000011000100001010010010011111000100"
"4000010111111111111011010011010001110000110000110"
"4000010111011111111111111011010001110000110000110"
"4000010000010111111111111011010011110000110000000"
"4000010110011111111111111011010001110000110000010"
"4000000111000011110000001001000000100000011010011"
"4000010000010000000011100010011000000001100000110"
"4000000000000111110010011000010000110000110000010"
"4000000000001000011000010000000010010000110000110"
"4000000111010010110000011001010000010000110000010"
"4000000000001000001000100001010011011000111000100"
"4000010000010000110001010010010110010000010000010"
"4000000000010000110001010010010110010000111000010"
"4000010000011000111001011001010010011000111000010"
"4000010000010000111001011010010011111000011000010"
"4000010000010001110001010010010111111001011000010"
"4000010000010000110001010010010110010111111000010"
"4000000000110000110001010010010010010011111000010"
"4000000000001000011000101001010010010011111000100"
"4000000000110000110001010010010010010000111000010"
"4000000001101001111011110111111001111000100001110"
"4001100111100111100111111011111001100001100000100"
"4000001001001011011010010111110001100001110001100"
"4100001101011011110011110011111011111001100011000"
"4000000000001011001011011111110011110000110000110"
"4000000000011010010100010111111000111000100000110"
"4000000000010011011010010000010000010000010000010"
"4001111110110001100000111010011011100001100000100"
"4001101110110011100000110010111010100001100000100"
"4000000110000011000001100000111010101001100000100"
"4010000011100011111000101000100001000001100001100"
"4000000110000011100001111010000011100001100000100"
"4000000110000011111000111000100001100001100000100"
"4000000000010111111010010010010001010000110000010"
"4000000000001000011000101001010010010011110000100"
"4000010000110001110001110010010010010000110000010"
"4000010000110000110001010010010111111010111000010"
"4000010000110001110001010010010010010011111000010"
"4000010000111000111001010010010110010000110000010"
"4000010000110000110001010010010010010011111000010"
"4000000000110001110001010010010010110000111000110"
"4000110000110001110001010010010111111000110000110"
"4000010000110000110001010011010010011000010000010"
"4000010000110000010001010010010011111000010000110"
"4000000000110000110001010010010010010000011000010"
"4000010000110000110001010010010110010111111000010"
"4000000000010000110001010001010010010011111000010"
"4000000000110000110001110001010010010111111000010"
"4000010000110001110001010011010010011011111000010"
"4000010000110001110001010010010011111011111000011"
"4000010000110001110001010010010110010000010000010"
"4000000000010001110001010010010110011000011000011"
"4000000000110001110001010010010010010111111000010"
"4000010000110001110001010010010110010000111000010"
"4000010000110000110001110010110010110000110000110"
"4000000000110000110001011011010010011011111000010"
"4000000000010000110001010000010010010111111000010"
"4000000000110000110001010010010111111011111000010"
"4000010000110000110001010011010010010011110000010"
"4000000000011000111001011011001011011011011000011"
"4000000000110000110001010011010010010011111000010"
"4000000000010000110011010010010110010000011000010"
"4000110000110000110001110010010111110011111000110"
"4000000000110001110001010010010010011000011000010"
"4000010000110000110001010010010010011111111000010"
"4000010000110000110001110010010010110000110000110"
"4000010000110001110001010010010010011011011000010"
"4000000000110000110001010010010110011010111000010"
"4000000000110001110011010010010011110000010000010"
"4000010000110000110001010010010010010011011000010"
"4000010000011001110001010010010110011000011000010"
"4000010000110000110001010011010010010000010000010"
"4000010000110000110001110000010010010111111000010"
"4000010000110000110001010010010011011000011000010"
"4000000000110000110001110010010110010011111000010"
"4000000000010000110001010010010011110000010000110"
"4000000001100001000011000010010010010111111000010"
"4001000001000001000011000010010010010111111000010"
"4000000001100011000011000010010010010011111000010"
"4000100001100001000011000010010011111011111000010"
"4001000001000011000011000010010110110111111000010"
"4001000001000011000010000010010110110011111000110"
"4000000001000010000010000010100010100111111000100"
"4000000001000001000011000010010010110011111000110"
"4001000001000011000011000010010010010111111000110"
"4000010000010111111011010001010000110000010000010"
"4000000000000000000111111010010001000000100000000"
"4000000000000000000111111010100001100000100000100"
"4000000000000000000111111010000001000001100000100"
"4000000000000000000011111010000001000000100000000"
"4000010000010111111011111010010001010000110000010"
"4000000000010111111010010011010001010000110000010"
"4000001000011000111001011000010010011111111000010"
"4000010000110000010001010010010100010000010000010"
"4000010000110000110001010010010111111011010000010"
"4000010000010000110001010010010011111011010000010"
"4100000110010011110010011001010000010000110000010"
"4000010000010011111010010001010001010000110000010"
"4000000000100001000001000010010010011011111000010"
"4000001000001000011000010001010011110000100001100"
"4000100000100001000001000010010010010000010000010"
"4000010000010011111011010011010001010000110000110"
"4000010000010000110001010010010010010010010000010"
"4000010000110000110001010010010100010011010000010"
"4000000000010000110001010011010010010000111000010"
"4000110000110000110001110010110010110111111000110"
"4000000000110000110001010010010010111000010000010"
"4000010000010000110001110011010110010000010000010"
"4100000110001000011000110001010000000111110001000"
"4100000110000011000001110001010000110100010100010"
"4000010000110000110001010010010110010000010000010"
"4000000000010000110001010010010110011000010000010"
"4000010000010000010001010010010111111000010000010"
"4000010000010000110001011010010010011000011000010"
"4000010000010111111110011011010001010000110000010"
"4000000000010000110001010000010011111000011000010"
"4000000000110001110011010010010111111000010000010"
"4001000001000001010010010010010011111000010000010"
"4000010000110001110011010010010011111000010000010"
"4000000000110001110001010010010111111000010000010"
"4000010000110001110011010110010111111000010000010"
"4000010000111001111001011010011111111000011000011"
"4000010000010111111100010010010001010000110000010"
"4000010000010111111100010010010001010001110000110"
"4000010000010111111111111000010001010000110000010"
"4000000000000111111010000000000001000000100000000"
"4000000000000011111000000010000001000000100000000"
"4000000000010011111010010000010001010000110000010"
"4000000000110000010001010010010111111000010000010"
"4000000000010111111000010000010001010000110000010"
"4000000000010111111010000000000001010000100000000"
"4000000000000011111000000010000001000000000000000"
"4000010000010011111000010000010001010000110000110"
"4000010000110000110001010010010111111000111000010"
"4000001000011000111001011010011010011011111000011"
"4000001000011000111001010000010011110111110000100"
"4000000000001000110001110011111111100001000001000"
"4000000000011000111001110111111000100001000001000"
"4000000000011001111011011111110000100001100001000"
"4000000100011000111001110011110101100001100001000"
"4000001000011000111001111111111001100001100001000"
"4000001000011001110011111111110001100011000010000"
"4000000000011000111001011011111001100001100001000"
"4000001000011000111001011010010110111111111000100"
"4000001000011001111011010111111001100001000011000"
"4000001000011001111000010011111111110001000010000"
"4000000000001000111001010011110111100011000011000"
"4000000000010000110011010011111011100011000010000"
"4000010000110001110011111111111001000001000011000"
"4000001000110011010011111011111011000011000110000"
"4110000110010001110011111111111011000010000010000"
"4000000000011001110010111011110001100011000010000"
"4000000111010011110010111001010000010000110000010"
"4000001100011000111001000010010010110000110000110"
"4000001100011000110011111111111011100001000010000"
"4000000000001100111001111011111011110001100000000"
"4000000000001000111010010111111001100001000011000"
"4000000100011000111011011111111011100001000010000"
"4000010000110001100000100111111001000010000110000"
"4000001000011000111000001001011011111011111000110"
"4000001000011000111000001001010011111010110000110"
"4000001000011000111001001000000010010000110000010"
"4000001000001000111000101000010010010000110000010"
"4000001000011000111001010010010110010000110000100"
"4000001100011000011001111001011010010111111000010"
"4000001000011000111000111001010010011000010000110"
"4000001000011000111001011010011110011111111000010"
"4000001000011000111001010011010010110000110000100"
"4000001000011000111001011010010010010000010000110"
"4000001000011000111001011011011111111010010000010"
"4000001000011000111001011000010010010000110000110"
"4000010011111111111011111000010001010000110000010"
"4000000000110001110001110010110111111000110000110"
"4000010000110000110001010011010111111000010000010"
"4000010000110000110001010000110011111000110000110"
"4000010000110001110001110010010011111000010000010"
"4000010000110001110001110010110111111000010000110"
"4000010000110000110001110001110111111000110000110"
"4000000000110001110001110010110011111000110000110"
"4000010000110001110001110010110111111000110000110"
"4100000011010010111000011001010000110000010000010"
"4000000000110000110001110010010111111000010000010"
"4000000000110000110001010000010111111000110000010"
"4000010000110001110001110010010111111000010000010"
"4000010000110000110001110011010011111000010000010"
"4000010000110000110001010010110111111000110000110"
"4000010000110001110001110011010111111000010000110"
"4000001000011000101000010000010000010010111000000"
"4000001000011000011000000010010100000000010000100"
"4100000111110011110000011001010001110000110000010"
)+string("5100010100110100110101001100000011000001000000001"
"5000000000111000111000000000000000000011101011101"
"5000000100111101101101000101000101000111001001100"
"5000000100111001101001000001000001001011101001101"
"5000100100111101001101001101001101001111001011001"
"5000110100111100111101001101000100001111001011101"
"5100110100111101001101001101000101001111001001101"
"5000000100111101111001001001000001001011001011001"
"5000110100111101001101001101000101000111001011101"
"5000010100111100111100001100000100001111101001101"
"5000000100111101111101001101000101001111101001101"
"5000000100111101001101001101001101001111001001000"
"5011111011111010000011110010011000001110011011110"
"5010001010000010000011110100001000001100001010010"
"5001001010000010000010110010001000001110001010001"
"5010011010000010000011110010001000001110001010010"
"5000000010000010000011110000001000001110001011010"
"5000010100111100001101000101000101000111001001001"
"5000010100111100001101000101000100000111001011001"
"5000010100111100001001000001000001000011101001101"
"5000010100111101101101000101000101000111001001001"
"5000010100111100001101000100000100000111101001101"
"5000011010000010000011111000001000001000001010011"
"5000011010000010000011110000001000001000001010011"
"5000111100111100000100000101000101000101001011001"
"5000010100111100000100000100000101000101000111001"
"5000010100111100101100000100000100000111101011001"
"5000000000010000011100000100000100000101000011001"
"5000111100111100000100000100000101000101001011001"
"5000010000111000100000000100000000000000000011000"
"5000011000111100100100100100100100100011111000000"
"5100000100000101010101001111000001000000000000001"
"5100000100100101000101001101000001000000000000001"
"5100000100000100011101001101000011000000000000001"
"5000000000000000110000001010000011000000000000001"
"5000000000000000000000100000001000000000000110001"
"5011110011110010000111110000011000001000001011111"
"5000010011110010000011110000011000001010011011110"
"5000000011110010000011100010111000001000001011111"
"5000010011110010000011110000011000011010011011011"
"5000100011100010000011100010011000001000001011111"
"5001010011110010000111110000011000001000011011110"
"5011100011100110000011110000011000001000011011110"
"5000100011100110000111110010011000001000011011110"
"5000111001000001000001110010011000001110010010110"
"5000000001000001000011110010001000001000010010100"
"5000001011001010000010000010011000000000000110001"
"5000111001000001000001110010010000000100010110100"
"5000111001000001000011110000001000001110010011100"
"5001111001000000000011110000010000010100010110100"
"5000111001000001000001110010001000001100010011100"
"5001111110000110000111110000001000001000001010001"
"5000110010000010000011110000001000001010001011111"
"5000110110000110000110000000011000001000001010001"
"5000010010000010000011100000001000001000001010001"
"5000010010000010000011110000001000001000001010001"
"5011111010000010000111110000001000001000001011011"
"5011100010000010000011110000001000001110001011011"
"5000010110000110000111110000001000001000001111111"
"5000100110000110000011110000001000001000001011111"
"5001110110000110000111100000011000001000001010001"
"5011110010000010000111110000001000001100001011111"
"5000111011111010000011110010001000001000001011111"
"5001011011011010000010110010001000001100001010011"
"5001111011111010000011110010001000001000001111011"
"5001110001100010000011100010100000100110100011000"
"5000111001111001000011110011011000011110010011110"
"5001111001111001000001110010011000011110010011100"
"5000000011011110000111110000011000001110001111110"
"5000000001111001000001110010010000010110010011100"
"5001111001000001000011110010010000010110010111100"
"5001111001111001000011110010010000010110010011100"
"5001111001111001000001110011010000010110010111110"
"5000001001111001000001110010011000010110010011100"
"5001111001111001000001110011011000010110010011110"
"5001011001111001000001110010011000011110010011100"
"5001111001111001000011110010011000010110010011100"
"5000000001111001000011110010010000010110010011100"
"5001111001111001000001110010010000010110010011100"
"5000000000111000100001110000010000010000010010100"
"5000000001111001000001110011011000010110010011100"
"5000000001111001000001110011010000010110010111100"
"5011111011110110000111100000001000001000001011111"
"5000000100000100000010000000001000001000001010011"
"5001110100000100000111100000001000000000001110110"
"5011110011100010000111110000001000001110001011011"
"5000010111100000000010000001100000001000001001011"
"5111100110000110000110000111110000001000001011110"
"5011111100010100000100000010000000000000000111100"
"5000111110000100000010000001111000000000000011111"
"5010000010000000000010000001110000000000001001110"
"5000010011000010000000000000001000001000001001110"
"5001110110000100000100000011111000001000001001110"
"5011100110000000000011110000001000001000011001000"
"5000000011000100000100000101000010001000001001011"
"5000000100000100110101001101001010000001100000001"
"5000011011110010000011111000001000001010001011011"
"5000000011110100000010000010000000001000000000001"
"5000000100000100000010000010010000001000001010011"
"5001111100000100000100000001110000001000000000001"
"5000001010000010000001100000000000010000010110100"
"5000001010000000000010000011110000001000001010010"
"5011110110000010000010000001110000001000001001011"
"5001100110000110000011110001001000001000001001110"
"5000000010000100000010000000001000001000010001000"
"5001111001000011000001110000001000001000001111100"
"5011110011110010000110011000001000001110001011111"
"5000110001000011000011110000001000001000001110010"
"5011111100000100000111111000001000001010001011111"
"5011111100000100000111110000001000001100001010011"
"5011111110000010000011111000000000000000000011011"
"5011110110000110000011100000001000001000001011111"
"5011111110000100000111110000001000001000001011111"
"5000000110000110000111110000001000001000001011011"
"5001111010000010000011111000001000001000001001111"
"5000000010000010000011111000001000001010001001011"
"5111110100000100000111110000001000001110001011110"
"5011111011100010000110010000001000001110001011110"
"5111111110000100000111110000001000001000001011001"
"5111111100000100000111110000011000001000001111011"
"5111111100000100000111110000001000001100001111011"
"5001010011110000000010000000001000001000001011111"
"5001110110000100000011110000001000001010001011110"
"5111111100000100000111110000001000001100001011111"
"5011111111110100000110000000001000001000001011111"
"5000010110000110000111110000001000001010001011111"
"5111111110000100000111111000001000001000001011111"
"5111111100000100000111110000001000001100001111111"
"5011111110000010000011110000001000001000001010011"
"5011111110000100000111110000001000001000001111111"
"5011111110000100000111110000001000001010001111111"
"5011111010000000000011110000001000001000001011111"
"5000000110000110000111110000001000001000001011111"
"5000000010000010000011110000001000001000001011111"
"5111111100000100000111110000011000011000001011111"
"5000001110000010000011111000001000001000001001111"
"5111111100000100000111110000001000001000001010001"
"5001111001110000000011110000011000001000001111110"
"5000011110000100000111000000001000001000001011111"
"5001011110000100000111111000001000001010001011111"
"5011111100000100000111110000011000001000001011011"
"5000011100000100000111110000001000001000001011111"
"5000100111110100000110000000001000001000001111011"
"5011111010000010000011110000001000001100001011110"
"5000110000111000001000000001000100001011001011011"
"5000110100111101001101000101000101001111001011101"
"5000010100111000001000000000000000000011101000000"
"5000111001000011000011110000011000001000001010010"
"5000000000111000001101000001000001000111001011011"
"5000110100111101001101000101000101000111001011001"
"5000110100111101001101000101000101001111001011011"
"5000010100111101001101000101000101000111001011011"
"5010111010000010000011110010001000001000001011110"
"5011111010000010000011111010001000001000001010011"
"5010111011110010000011110010001000001000001011111"
"5011111010000000000011111010001000000000001010010"
"5011111010000010000111111110001000001100001110011"
"5011110010000010000011110010001000001000001010001"
"5011111010000010000011110000001000001100001011010"
"5011110010000010000011110010001000001010001011010"
"5000010010000010000111110110001000001000001110011"
"5011110010000010000011110010001000001000001010011"
"5001111001111011000011111011011000011110011011110"
"5000000110000110000111100000011000001000001010011"
"5001111001110011000011110000011000011000011111110"
"5001111011111011000011110010111000011000011111110"
"5011101011111011000011110000011000011000011011111"
"5011111011000011000011110000011000011000011111110"
"5011100111110010000011100010011000001000001011110"
"5000010011111011000011100000111000011000011111111"
"5000010000111100101000001000000000000011000011100"
"5000010000110000101000000000000000000111000011000"
"5000110000110100001101000101000000000001000010000"
"5000000001100010000011010100001000001110001100010"
"5000111001100010000011010000001000001110001000010"
"5000111001100000000011010000001000001110001000010"
"5011000111111001000011100000110010011011011000110"
"5011100010000010000011110010011000011110011011110"
"5010000011111010000011000011001000001110001011110"
"5011111010000010000010000010001000001110001011011"
"5000111001111011000011110000011000001000001111010"
"5011111010000010000011110000001000001110001001110"
"5011111010100000000011000110011000001100001011111"
"5011111011100010000010000010001000001100001010011"
"5011111011001110000111110010001000001100001110011"
"5000000011111010000011110010001000001110001011110"
"5011110010000010000011100011011000001110001010010"
"5001110010000010000011110010001000001110001001110"
"5000110010001010000011110010001000001110001011111"
"5010110111111110000011000011111000001000001011111"
"5000010111111110000111000111111000001000011111111"
"5100000100110100111101001101000111000001100000101"
"5000111001000011000011110000011000001000001111110"
"5011110011110110000010000001111000001000001011110"
"5011110111110110000110000011111000011000001011111"
"5011110111110110000110000011111000001000011011111"
"5000000111110110000111000011111000001000011111111"
"5011111110000110000110000000111000001000001111111"
"5011110111110010000010000001111000001000011011110"
"5011110111110110000110000011111000001000001011111"
"5001100011111110001000011000110001100011000011111"
"5001110111111110000111100011111000001000001011111"
"5000011001110010000011110000011000001000001111110"
"5011111010000010000111000011111000011000011111110"
"5001110011110010000011000011111000001000001011111"
"5000010111111110000011000011111000001000011011111"
"5000100011110010000011000011111000001000011011111"
"5011110111110100000110000010111000001000011011111"
"5011110011111010000111000011111000001000001011111"
"5011100011110010000011000001110000011000011011111"
"5000010100111100111100100100100100100011000000001"
"5000010100111100101000000000000000000011101001101"
"5000000001111001000001110000010000010000010010100"
"5000111001111011000011110000011000001000001010010"
"5010111011111010000011110010001000001110001011111"
"5011111010000010000011110110001000001100001010011"
"5010001011000010000011110110001000001100001011111"
"5011111010000010000011110010001000001110001011111"
"5011111100000100000100000000010000000000000000001"
"5011110100000100000100000000000000001000001110010"
"5011111100000100000100000000010000000000000100000"
"5011111100000000000000000000010000000000000000000"
"5011111100000100000100000000010000001000000110010"
"5000011001000001000001110010010000010110010011100"
"5011111100000100000100000000010000001000001100010"
"5011000100000100000100000000010000001000001100010"
"5000001100000100000100000011110000001000001000010"
"5010011100000100000100000000010000000000000000001"
"5000010000111000101100000100000100000010101011101"
"5011110110000010000010000011111000001000001011111"
"5011111010000010000110000010001000001000001011111"
"5011111010000010000011110000001000001000001011011"
"5000110010000010000011110000001000001100001011110"
"5001110011000010000011110000001000001000001011111"
"5011010010000010000011110000001000001000001011110"
"5001000101100100010100000110101000100000000000000"
"5000000000111000101001000001000101001111001011111"
"5000110100111101001001000001000011001011101001101"
"5000010100111101101101001101000101001011001001101"
"5000110100111100001100000100000100000011001001101"
"5000010100111100101101000101000100000111101001101"
"5000010100111100101100000100000100000111101001101"
"5000111001000001000011110010011000011100010011100"
"5000000010000010000011110010001000001010001011011"
"5000000010000010000011110010001000001110001011011"
"5010001010000010000011110000001000001010001011110"
"5011111011000010000011111010001000001010001011011"
"5011111010000010000011110000001000001110001010011"
"5000000010000010000011110000001000001110001010011"
"5011000011111010000111110010011000001110011011110"
"5001010011110010000011110010011000001010001011110"
"5011110011111010000011110010011000001010001011111"
"5000010100111100101100000100000100000111000001000"
"5000010100101100001100001100000100000110000000001"
"5000110100001101000101000101000101000101000101001"
"5100110100001101000101000101000101000101001101001"
"5000000000001101000101000001000101000001000011001"
"5000110100001100000101000101000101000101000101001"
"5000110100001101000101000101000101000101000111001"
"5100110100001101000101000101000101000101000101001"
"5000010100001101000101000101000101000101000111001"
"5000010100001101000101000101000101000101000101001"
"5000000100001101000100000100000100000100000110001"
"5000001001000001000001110010010000010110010010100"
"5001111011110010000011110010001000001110001011111"
"5000000111110110000011110010011000001000011011111"
"5011110010000010000011110000011000001110011011110"
"5001111011100010000011111000001000001110001011110"
"5000011010000010000011110010000000000010000011000"
"5000000010000010000011110000001000000000001000011"
"5000000000111001000001110110010000110011000100000"
"5000000100011100100101110100010000100011100000000"
"5000001010010100111101001000011001110010000110000"
"5100011100100001110011010000110000110011100100000"
"5001110010000010000011110000001000001100001010011"
"5000001100011100111001111001001000110011100110000"
"5000000000100001111011001000001000011011110100000"
"5000000000110100100001110010010001100011000100000"
"5000000000011000100001110010010000100011100100000"
"5000000100010100110001110011000000110011100110000"
"5000011100100100100101110011010000100001100011000"
"5000000000110101110001110000010001100011000100000"
"5000000100010100111001101000011000110011000000000"
"5000001100010100100001010000010000100010000000000"
"5100000100110101111001001001000011000000101000001"
"5000111001111001000011110010010000010100010111100"
"5000001100010100111101000000001000011011110000000"
"5001111001111001000011110010011000011110011110110"
"5101111001000001000001110010010000010110010011100"
"5000011001101001000011110010011000011110010011100"
"5000111001111001000011110010011000011110010011100"
"5001110010000010000111111000001000001110001011110"
"5001110010000010000111110000001000001110001011110"
"5001110010000010000011110000001000001110001011110"
"5000000011110011000011110000001000001010001011011"
"5001110010000111000011110000001000001100001011111"
"5100000100000100110100001100000011000001100000001"
"5001110010000010000111111000001000001110011011110"
"5000000011110010000011111000001000001110001011110"
"5001110011000010000011110000001000001010001011111"
"5001110011110010000011110000001000001110001011110"
"5100000100111101111101001101000111000001111000011"
"5000010100111101001101000101000101000101011011011"
"5000010100111101001101000101000101000101000011011"
"5000010000111100101001000001000001000000001011001"
"5000010000111101000001000101000101000000001011101"
"5100000100110100110101001101001011000000100000001"
"5000010000111100001100000101000101000100011011011"
"5000010000111001000101000100000100000101000011011"
"5000000000000000001100011100100100100111001000001"
"5000000000000100000000011100000011100000001000010"
"5000000000000000001000001000010000000100101011101"
"5011111010000010000011110010001000001010001011110"
"5001110011110011000011111011011000001010011011110"
"5001110001111011000011110011011000001110011011111"
"5001111011110011000011110010001000001110011011110"
"5001111011111011000011110010001000001110001011111"
"5001110011110011000011110011011000001110001011111"
"5001000011111011000011110010011000001110011011110"
"5000010011000000000100000011000000000000000011000"
"5000110100111101101101001101000101001111101011101"
"5000110100111101101101001101000101001011001001000"
"5001111010000010000010011000001000001010001010010"
"5000010100111100111000000100000100000111101001001"
"5001111011000011000011110000001000001010001011010"
"5001111010000010000011011000001000001110001011111"
"5001110010000010000010010000001000001100001011110"
"5000000011110010000011110000001000001010001010011"
"5000110011110010000011111000001000001010001010011"
"5011110010000110000111011000001000001100001110011"
"5001110011010011000011110000001000001110001011010"
"5001110011000011000011111000001000001110001011111"
"5001110010000010000111110000001000001110001010010"
"5000010100111100111100000100000000000011101001101"
"5000110100111100101101000101000101001111101011101"
"5000110100111101111101001101000101000111001111111"
"5000010100111101101101000001000001000111101001101"
"5000010100111100111101001101000101001111001111101"
"5100110100111101001101000101000101001111101000000"
"5000110100111101111101000101000101000111001000101"
"5100110101111101001101000101000101001111101111101"
)+string("6010000100110100111100000100000010000001100000111"
"6010000010110100111100001000000010000001100000111"
"6000110011001010000111100010001010001110001011010"
"6000100010011010000111110000001100001010001011110"
"6000110011001010000111111010001010001010001001111"
"6000100001111010000011100110001110001010001001110"
"6000110001001011000011010011001010001011001001100"
"6000110011001010000111110110001110001010001011011"
"6000110010011010000011110110001110001110001011111"
"6000110011011010000010000000001010001110001011010"
"6000110010001010000010100110001110001010001011011"
"6010000010110000110000001010000010100001100000111"
"6000100011011010000011110010001010001010001001010"
"6000110001011010000011110010001010001110001011011"
"6001110011011110000011110110010110001110001011110"
"6000100001111110000111111010001110001010001011011"
"6000000001111010000111110010001010000010001011001"
"6001110010001010000111110110001110001110001011011"
"6000110001011010000011110110001010001010001001110"
"6000000001111010000011110010011010001010001001010"
"6000100010001110000110100110001110001110001011110"
"6000100011011110000111110110001110001110001011111"
"6010000010000100110100001000000010000001100000111"
"6000110010001010000011110010001010001010001011111"
"6000110011011010000111111110001010001010001001111"
"6000110011001011000011110111001011001011001001111"
"6000010010111010101101001101000010101011111001110"
"6000010010111010111101000100000010001011111001110"
"6010110010111010101101000100000010101011111011110"
"6000010010111010101100000000000010101011111001110"
"6000000010111010001101000101000010001011111001110"
"6000000010111010111100001100000000001011111011111"
"6010000110110100101100001000000010000001100000111"
"6000110010001010000010110110001010001010001011010"
"6001110010001010001101110110001100001010001010000"
"6000000010001010000001110110001110001010001011110"
"6000010000000000101000100000100100001010111001110"
"6000000010111010101100000101000100000010101011111"
"6000010010101100001101000100000100001011111001110"
"6000010010111110101101000100000100000011111001110"
"6000010010111010101100000000000000000011111001110"
"6000010101100001000011101010001100001000010011110"
"6000001010100101000110101011001010001010010010100"
"6010000110110100110100001100000010000010100000111"
"6000100011011010000010000011011110001010001001001"
"6000000011011010000010000111111110001010001011011"
"6000000001011010000000000011111010001010001011011"
"6000000001011010000010000011011010001010001001011"
"6000100010001010000010000011111010001010001011011"
"6000010010111010111100000100000100000011111011111"
"6010000010000100110000001000001010000001100000111"
"6000100001000001000011110010011010001110001011111"
"6001000001000001000011110010001010001010001001110"
"6000000001000011000011100110011110011110001011111"
"6001000011000011000111110110011110001110011011110"
"6000100001100011000011110110111110001110001001110"
"6000000001100001000011110010011110001010001011111"
"6000000001000011000011000111111110011110001011111"
"6000010000100001100001100011111010001110001011111"
"6010010010111010111100001100000010000011111001111"
"6000000001000011000010000011111111001110001011111"
"6000100001100001000011000011111110011110011011111"
"6000100001100011000010110111111110001011011011111"
"6001000011000010000010000011110110011010001011111"
"6000100001100011000011100111111110001010001011111"
"6000000011000010000010100011111110001010001011111"
"6000000000100001000010000011110110011110001011111"
"6001000001000010000010000111110110011010001011111"
"6001000001000011000011110010111110001010011011110"
"6000100011110010000111110011011010001010011011111"
"6000100011110010000010100011111110001010001001110"
"6000000011010010000111110111011110001010011011110"
"6000100011110010000110110111111110001110001011110"
"6000010001001001000010110010001100001100011010010"
"6000010001001010000010110010001010001100010010110"
"6000010001001001000010110010001000001100010011110"
"6000010001101010000011110110001100001100010111110"
"6000111001001001000010110010011010001100010010110"
"6000010001001001000010110010011010001100010010110"
"6000100010001110000010000111111010001110001010011"
"6000100010011010000111110110001000001000001011110"
"6001100011001010000011110110001010001010001011111"
"6000100011011010000110110110001110001010001011011"
"6000110011001010000011110110001010001010001011111"
"6001100010010100000110010100010010011000001000001"
"6000000011001010000111110110001110000010000011011"
"6000100011011010000101110110001000001010001011010"
"6000100011011010000000110110001000001010001011011"
"6000110011001010000011110110001010000010001011011"
"6000100010001100000101100100001000001000000010001"
"6000000011011010000000110111001010001010001011111"
"6001110011111010000101110110001000001010001011111"
"6000011001111001000010100010011110011110010011100"
"6000010000111001001011000011110010011110010011110"
"6000010001101001001010000011011110011110010011110"
"6000011001111001001011110011011010001110010011110"
"6000010001101011000010110011010010010110010011110"
"6000011001111001001011110011011110001010011011110"
"6000010001001001000011110010011110010110010011100"
"6000010001101011001010110011110010010110010011100"
"6010000010110100110100101000000010100001100001111"
"6000000010001000000011110000001000000000000010001"
"6000011001101001000011110010011010001010010011100"
"6000010001111001000010110011010010010110010011110"
"6000010001101001001000000011110010010110010011110"
"6000011001111001001010000011110110011110010011110"
"6000010011001000000000100000001000000000000011011"
"6000000010001010000010000010001010000010000001001"
"6010000010000100000100000100011100101011001001111"
"6010000010000000000100000100010010101001011001110"
"6000110011011010000011110010001010001010001011111"
"6001000000000010000110000110000111111011001001111"
"6001000011000110000110000100000100111110011011110"
"6000000011000010000010000100000111111110001011111"
"6100110001100001000010000110010110101011001001110"
"6000000001000010000000000000000000111010101001111"
"6001000001000010000010000100000101111011001011001"
"6000000100000100000100010100101000101011111001110"
"6010000010000010000000000000011010101011110001100"
"6000100001100001000010000010010110111110001010110"
"6011000010000010000110000110000011000011110001111"
"6000000010111010111000000100000010001011111011111"
"6000000110011110000100000111110110001110001011111"
"6011110110011100000100000111111100001110001111111"
"6011110110011100000100000110111100011100001011111"
"6001110010011110000010000011111010001010001011001"
"6011111010001010000010000010001010001010001011111"
"6000100011111110001110000011111010001110001011111"
"6000110011001010000010110110001110000010001011001"
"6001110011001010000111110110001110001010001011010"
"6000100011001010000111110110001110001010001011111"
"6000100001011010000010100110001010001010001010010"
"6000010010111010101000100100000010101011111001111"
"6010010010111000000100000000000010001010001011111"
"6000010010111000001100000100000000001011111001110"
"6010100010111100001100000100000100000010101011110"
"6000001000000011011001001001001011111000001010001"
"6000010010111000001000000100000100000010111011110"
"6000110011000010000101110110001000001010001011110"
"6000110001000010000100110111001110001110001011011"
"6000111001000010000010000010001010001010001010011"
"6000000001000010000010000010011010001010001010001"
"6000110001000010000010000010001110001010001001011"
"6000010010111010101100000100000000000011101011111"
"6000110011000010000010110110001110001010000011010"
"6000100001001010000110100110001100001110001011011"
"6000110001000010000010000010001010001010001001010"
"6000100011000010000000100010001000001010001011001"
"6001110001000010000110000110011110001100001011011"
"6000110011110010000000000110010100001000001011010"
"6000110001000010000111110110001110001010001011110"
"6001110001000010000000110110001100001010001011110"
"6000111001000010000011110110001010001010001001010"
"6000110011000010000000100111011110001010001011001"
"6000111001000010000110100111001110001010001011001"
"6000010011001010000010110111001100001010001011011"
"6000000011011110001110100110001110001110001011011"
"6000100011001010001011110111001010001010001001111"
"6001110011011010000010110110001000001010001010110"
"6000110011011010001110110111011110001010001011111"
"6001110011011110001111110110001110001110001011110"
"6000100011111010001110100111011110001010001011011"
"6000110011110010001010100110011010001010001011110"
"6000010000111001000011111010001100001100010011100"
"6000100011011010001111110110011110001010001011111"
"6000000011111010001110110111011110001010001011111"
"6000110001111011001011110111111110011010011011110"
"6001110010000110000110000110001110001110001011111"
"6001100010000010000110000110001110001110001011111"
"6001110010000110000101100110011100001100001011111"
"6000100001100001100011110111111110011110011011110"
"6000110001110011100011110111111111011111011011111"
"6000100001110011100011110111011110011110011011111"
"6000100001100011000011000111111110011110011011111"
"6000000000111001100001110010001010011110010011100"
"6000110001100011000011110011011110011010011011110"
"6000110001100001100011110011011110011010011011111"
"6000000001100011100011100110011110011110011111111"
"6001100011111011011011110011111111011111011011111"
"6000110001100011100011110010011110011010011011111"
"6000011000111100111110111111101011101001111000010"
"6000010000111100111110101111001011111001111000111"
"6000000000011100000100000000000010000001011000110"
"6000010000011100001000000010000010010001011000111"
"6000000000011000101100100000100010100011001001111"
"6000000000101001000001110010010010010000010010100"
"6000010001001001001010100010001010001110011010010"
"6000010001001010000010100010001110001110001010010"
"6000010001001001001010000010000010001010001010010"
"6000010001001011001010100010011110001100011010010"
"6000010001001010001011110010010100001100011010010"
"6000010001001011000010110011001010001110001010010"
"6000110011111010000011110111011110001010001011111"
"6000010000101001000011110010010010000100010110110"
"6000110011001010001101110110001110001010001011111"
"6000010010111010101100000100000010001011111001110"
"6000010010111010101000000100000000001011111011111"
"6000000010110010101100000000000000001011111001110"
"6000010010111000100100000100000100001010111011111"
"6001000001000011000010000110111111101110001011111"
"6000000001111011000010000110111111001110001011111"
"6000010000110001100010010110111111001011011001110"
"6010000110110100101100001000000010100001101000111"
"6000100011011010011011110110001110001010001011010"
"6000010001110011000010000010111111011111001011111"
"6000100000111000101000101000001110011011011001111"
"6000010010111010101100000100000000001010101011110"
"6000010010111010101000000000000000000010101011111"
"6000110010111000001101100100001110101010101001110"
"6000010010111010101100000100000010001010101011111"
"6000010010111010101100000100000000001010101001110"
"6001110011011010000110110110001110001010001001111"
"6001100011111010000111110110001110001010001001111"
"6000011001100010000011100110001100001110001011111"
"6001110001011010000010110010001010001010001001011"
"6000110011010010000110100110001110001010001011010"
"6000100011011010000111111110011110001010001011111"
"6000000011111010001010000110001010001010001011011"
"6000000001001010000011110110001010001010001001111"
"6000100011001010000010100010001010001010001011110"
"6001110011011010000111110110011010001010001011011"
"6000000001111011000011110111001110001010001011111"
"6000100001010010001110000110001110001110001010001"
"6001110010001010000111110110001010001010001011110"
"6000001001110011000111110100011100001110001011010"
"6001110011001010000011110110011110001010001001110"
"6000000011011010000111110110001110001010001001111"
"6001110010011010000011110110011010001010001011010"
"6000000001111010000110000110001110001010001011011"
"6001110001001010000011110010001010001010001001110"
"6000000011110010001010100110010110011010001011111"
"6000000001011010000110000110001110001010001011011"
"6000110011011010000010000110001010001010001011011"
"6000100011111010001110100110011110001010001010011"
"6000110011001011000011110010001010001011001001110"
"6000011000100001000011110010011110001010001011010"
"6000110011011010000010110110001010001010001011110"
"6000100011010010001110000110010110001010001011110"
"6000100001011010000010100110001010001010001001011"
"6000110001011010000000110110001010001010001010001"
"6000110011111010000010000110001110001010001011001"
"6001110001011010000110100110001110001010001001011"
"6001110010011010000011110010001010001010001001110"
"6001110011011010000010100110011010001010001011111"
"6000000001111010000010000110011010001010011001111"
"6000110011011010000111110110011010001010011011111"
"6000110001011010000010110110001010001010001001111"
"6000110011001010000011110110001010001010001001110"
"6000100011011010000111110110001110001010001001110"
"6001110010011010000111110110011110001010011011110"
"6000110011011010000111110110001010001010001001111"
"6000110011111110000111100110001110001110001011110"
"6001110010001010000011110010001010001010001001011"
"6000110001111010000010000111001010001010001011111"
"6000100011011010000011110111001010001010001001111"
"6000010001110011000110000110011110001110001011111"
"6000000011110011000110000111111100001110001011111"
"6000000001100011000111000011111010001010011011111"
"6000100001100011000111100110111100011110011011111"
"6000010001110010000111000111111100001110011011111"
"6000111001110011000011110011111010001010001011111"
"6000100001100011000111000011110000001010011011110"
"6100010100111100101000100010100010100001101001110"
"6000010000011000100000100010100010100001101000110"
"6000010000111000101000100010000010100001111000110"
"6000001000100001000011110110001110001010001001001"
"6000010000011000100000100010100010100001111000110"
"6000010100111000100000100010100010101001111000110"
"6000000100111100101000100010100010100001111001110"
"6000010010111010001001001101000000001011111001110"
"6000010010111000101100000100000000100010101011111"
"6000001000110001000011110010010110011110010010110"
"6000100010001010000110100110001010001010001011011"
"6000001000100011000010000110001110001010001011011"
"6000100011011010000110110110001010001010001001111"
"6010010010111001101000000000000000000010101011111"
"6000010010111010101100000100000010001011101001110"
"6000011000100011000011100010011010001010001011111"
"6000010010111010101100000100000010000011101011111"
"6000000011011010001010000010001010001010001011011"
"6000100011111010000011100010011010001010001011111"
"6001110010000010000011100010011000001000001011110"
"6000001000100011000010110110001110001010001011001"
"6000100001111011000011110011001010001011001001110"
"6000100011111010000011100011011010001010001001110"
"6000000000011000100001000001010010010010000011000"
"6000011000101001000001100011010100000100100111001"
"6000000000101001000001110010010000010100101011001"
"6000011000101000000001000001010010010000101100000"
"6010110100001000100000110010001001001000100000011"
"6010010110001000000010100010001001100000110000010"
"6000001000100000000001100011010010010000100001000"
"6000010010111000000101000001000000000010101011111"
"6000000010110100011000001010000010000001101000011"
"6000011001100011000010100110011110001110001011010"
"6000010000111001001001000001000001000011101111111"
"6000010010111010101000000100000000000010111001110"
"6000010010111010101100000000000010101011111011111"
"6000010010111010101101001101000100001010111011111"
"6000010010111010101100000101000000000010111011111"
"6000010010111110101100000100000010101011111001110"
"6000010010111010101100000100000000100010101011111"
"6001100010001010000111110110001110001010001010011"
"6000010001101010000011110010011110001110010011100"
"6001110010001010000111110110001110001010001011011"
"6000100010001010000111110110001110001010001010011"
"6000000010011010001100110110001110001010001010011"
"6000110011111010000110000111110010001011001011111"
"6000100001111010000110000111111110001011001001111"
"6000010100111100000000000000000010000001101001111"
"6000010000111100101001000001000010000010101001111"
"6000010000001100100100100000100010100000100001111"
"6000010000001100100100100100100010100010100001101"
"6000010000001100100100100100100010100010100000101"
"6000011000001100100100100000100010100010100001101"
"6000011000001100100100100100100000000010000001111"
"6000011000001000100100100000100010100010100001101"
"6000010000101100100100100000100010100010100001101"
"6000010000001000100100100100100000100010100001101"
"6000010000001000100100100000100010100000100000101"
"6000010000001100100100100100100000100010100001101"
"6000100011011010000011110010011010001010001011011"
"6001100010011010000111110110011110001010011011110"
"6001100011011010000111110110011110001010011011110"
"6000100010001010000100110110001100001100001010001"
"6000100010001000000100110110001100000010001001011"
"6000000000011000110001000010100000110100110011100"
"6000000000010000000000001010001101100011100010100"
"6000001000011000100001000010110010110010110011100"
"6000000000011001100011000011110100110110100011100"
"6000000000011000110001100110111100011110011011110"
"6000010001001000000011110010010000010000010011100"
"6000001000010001100011110011110110010110110011100"
"6000000000011000100001000011110100010110110011100"
"6000000001100110101010110011111001101000101000010"
"6100000100001000110001100011110111110111110011100"
"6000000000000110000100011001000011100100100011100"
"6000000000011001100011100011110100010100010011100"
"6000001000011000100011100011110000110111110011100"
"6000000010001100110001100011000011101111101011001"
"6000001000010100100001000000110000110100010011100"
"6000000001111010000011110010010100011100010011100"
"6010000100010100100001000011100011110000100011000"
"6000000000010000100001100010110110110100110111100"
"6000000000011000110001100010110110110111110011100"
"6000001000010000100001000010100101110110010011100"
"6000000000011000110001000011100011110110110011100"
"6000000000001000100001000011100100000110000011100"
"6000000000001000100001000010010100110100010011100"
"6000000000010000100001000011100100110100100011100"
"6000010001111001000010110010011010011100010010110"
"6000110001111001000010110010011110011100010010110"
"6000100010001010000011110010001100001010001011011"
"6000011001101010000011110110011110011110011111110"
"6010000010110100111100001000000010000011100000111"
"6000000001111001001010110010011110001110010011110"
"6000111001001011000011110010011010010110010011100"
"6000110001111011001010110010011110011110010011110"
"6000110001001001001010110010011110011110010011110"
"6000111001001010000011110110001110001100011111100"
"6000110011111010000011110110001110001010001011111"
"6000100011011010000011110110001010001010001001111"
"6000100001011011000111110011001010001011001001110"
"6000000001110010000011110011011010001010001011010"
"6000100001111010000011110011011010001010001011111"
"6000100011011010000111110010011110001110001011010"
"6000110011111010001011110011011010001011011001110"
"6000000001111011000010100011011010001010001011011"
"6000100001111011000011110011001010001010001001011"
"6000100001010010000011110010011010001010001001111"
"6000100011011010000111110111011110001010011011111"
"6000110011111010000011100111011110001010001011111"
"6000110001000010000011110010001010001010001001110"
"6000010001011011000011110011001010001011001001111"
"6000110011010010000111110110010110001110001001110"
"6000100001010010000111110111010010001010011001110"
"6000100001011010000011110111001110001010001001110"
"6000110011011010000011110010001010001010001001011"
"6000110011001010000111110111001110001110001001110"
"6000110011111011000011110111001010001011001001111"
"6000100001110010000011110011011010001010001001110"
"6000000011111010000111110111011010001011001001111"
"6000000001110010000011110010010010001010001001110"
"6010000000110100001100001010000010000001101000111"
"6000100001001001000010111010010110001010010010100"
"6000000010110110111101101101001011100011111001111"
"6000000010110110111101001111000010000001111001111"
"6000000010110110111101001101000011000011111001111"
"6000010010111010101100000101000000000011111001110"
"6000010001001010001110000110001110001010001001010"
"6000010100011100001010100010100001101001111000110"
"6100001100011100101000101000101010001011100011110"
"6000100010001010000011110010001010001010001011011"
"6010000010110100111000001000000010100001100000111"
"6000100011111011001011110111111010001010001001110"
"6000100011110010011011110111111110001010011011110"
"6000010100011000011010010001010000001000100000011"
"6000000000110001000010000100000000001010101001100"
"6000010001011010000010000010001010001110001001011"
"6000100011011010000010100110001110001010001010010"
"6001110010001010000011110010001010001010001001110"
)+string("7100000100000110000101000100100100110100011000001"
"7100000110000100000101000100100100110100001000000"
"7111111000011000010000010000100000100001100001000"
"7000101000001000001000110000100000100001000011000"
"7011111000011000010000110000100001100001000011000"
"7011111000001000001000010000110000100001100001000"
"7000000110000110000101000100100100011100001100000"
"7000000010000010000001000000110000011100001100000"
"7010000110000101000101100100110100011100000100000"
"7000000010000011000001000000110100011100001000000"
"7000000000000110000101000100110100011100001000000"
"7000000010000010000001000000110100011100001100000"
"7100000110000110000101000100100100010100011000001"
"7000000110000010000001100000110000011100000100000"
"7100000110000110000101100100110100011100001100000"
"7001100111111010000010000010000011000010000010000"
"7000000110000110000101000000100000011000001100000"
"7000000010000111000001000000110000001000000100000"
"7011111000001000000000010000000000100000100001000"
"7111111000001000001000010000110000100001100001000"
"7010001000001000001000010000010000100001100001000"
"7000000010000010000001000000110100111100000100000"
"7011111000011000010000110000100001100001000001000"
"7100000100000111100111100101110100111100011100001"
"7000000010000011100011110000111000011000001000000"
"7000000010000111100111100101110000011100011100001"
"7000000111000011000011110000111000011000001000000"
"7000000010000011000000110000011000001000000110000"
"7000000111100011111011111001000001000001000001000"
"7001000011111000001000010000100001100001000001000"
"7001000111101100001000010000100001000001000011000"
"7001000010111100001000010000100001000001000000000"
"7001000011101100001000010000100001000001000001000"
"7000000010111100001000010000100000000001000001000"
"7000000110000111000101100100110100011100000100000"
"7100000110000111000101100100111100011100000100000"
"7000000010000010000001100000111000011000000100000"
"7000000010000011000001000000110000011000001100000"
"7000000010000010000001000001111100011100000100000"
"7000000010000010000001000000110100011000001100000"
"7100000110000111000101000100110100011100000100000"
"7001110010011000001000011000010000010000100000100"
"7011111000001000011000010000010000110000100000100"
"7001111000001000001000010000110000100001100001000"
"7011110000011000011000011000110000100000100001000"
"7001111000001000001000011000010000000000100000100"
"7001110111111000001000010000010000010000110000100"
"7100000110000110000101000100111100111100000100000"
"7111111000011000011000010000110000100001100001000"
"7001110000001000011000011000010000110000100001000"
"7111111000001000011000010000010000100000100001100"
"7111111000001000001000010000110000110000100000100"
"7001110000011000010000010000110000100000100001100"
"7000010000011000010000110000110000100001100001000"
"7011111000011000011000011000010000110000100001100"
"7011111000001000001000011000010000110000100001100"
"7011000000011000011000010000110000100000100001100"
"7111111000001000001000010000110000110000100001100"
"7011000000011000010000010000010000100000100001100"
"7111111000001000011000010000010000110001100001100"
"7011110111111000011000011000011000110000110000110"
"7011110011111000011000010000010000110000110000100"
"7100000010000011000001100000111100001000000000000"
"7000000110000110000101000100110100011100001100000"
"7000000110000111000101100100010100001000000100000"
"7000000000001000010000000000100000000001000010000"
"7010000000001000001000000000010000110000100000000"
"7000000110000111000111100100111110011110000000000"
"7100000110000110000101000101111100111100000100000"
"7000000010000011000001000000110000011100001000000"
"7000000010000111000000100100110100011100000100000"
"7000000110000010000001100000110000011000001000000"
"7000000110000111000101100100111100011100001100000"
"7111111000001000010000000000100000000001000001000"
"7000000011111000001000010000110000100001000001000"
"7000100011111000010000110000100000100001000011000"
"7000000000011000011000010000110001100001100001000"
"7111111000011000011000110000100000100001000001000"
"7011111000001000010000010000110001100001000001000"
"7010000000001000001000010000000000100000000001000"
"7000010000010100000101000100100000010000001000001"
"7001000110110100000101000100100000010000011000000"
"7000010110001100000101000100100000010000001000001"
"7100000110000111000101100100110100011100001100000"
"7111111111111000010000110000100001100001000001000"
"7111111000011000010000110000100001000001000001000"
"7000000000001000001000010000010000100000100000100"
"7000000000001000001000010000010000110000100001100"
"7110000111000111100100110100011100000100000100000"
"7000010000000000001000010000010000100001000001000"
"7000000110000111100101110100011100001100000100000"
"7100000110000101000100100100010100001100001100000"
"7100000110000111000101100100010100001100000100000"
"7000000110000101000100100100010100001100000100000"
"7000000110000100000101000100100100010100011000001"
"7000110000001000010000000000100001000000000010000"
"7000000010000101000100100100010100001100000100000"
"7100000110000101000100100100010100001100000100000"
"7000000010000001000000100000010000001000000100000"
"7011111000011000010000100001100001000001000001000"
"7000000010000001000100100100010100001100000100000"
"7000110000001000010000010000100000100000100000000"
"7011110000001000010000000000100000000000000000000"
"7000000001111000000000010000100001000010000110000"
"7010000011100100001000110001000000000010000010000"
"7000100000000000010000001011111100000001000010000"
"7000000001111010011000010000100001000010000010000"
"7000001000111101011010010000100001000010001110011"
"7000000001111000010110110000100001000010000010000"
"7001001001111010111000010000100001000110000010000"
"7011000000111000010000110000100001100001000001000"
"7000000001111010011100110001100011000011000010000"
"7000100010111001001100010000101001001011000110000"
"7000100000111100000000010000100001000010000110000"
"7100000001111010011100010000100001000010000110000"
"7000000001111010001000010000100001000010000010000"
"7000001011111010110100110001100011000011000011000"
"7000000001011010111000110000101001001010000010000"
"7000000001111011011100010000100001000111000110000"
"7000001101111101011100110000100001100011000011000"
"7000000001111010011000010000100011000011000111000"
"7001000001111010001010010000110000100001000001000"
"7001000001000000000000000001111010110001000110000"
"7000000001111000010000000001000011000110000010000"
"7000001000011000010001110001100011100111000011000"
"7000000101111100001000010000100001000010000110000"
"7000000001001011111100010000100001000010000010000"
"7000000001111010011000000000100001000010000110000"
"7001000011111000010000000001001011001010001010001"
"7000000001111010111000010000100001000010000110000"
"7000000000000010111000010000100000000010011110000"
"7000000001001010010100010000100001100001001010001"
"7000000011011010011100010000100000100001000001000"
"7001000001111010001010010000010000000000100000000"
"7011111111111000011000010000010000010000100001100"
"7000000011111000001000010000010000100000100001000"
"7001101000001000010000100001100001000010000010000"
"7100001110000010000101000100100000100100011000001"
"7011111000001000010000100001000001000010000010000"
"7001111011111000010000100001100001000011000110000"
"7000000000001000010000100000100001000010000010000"
"7011111000001000001000010000010000100000100001000"
"7000111011111000010000100001100001000011000110000"
"7011111011111000010000100001000001000010000010000"
"7011111011111000010000100001100001000010000010000"
"7000011110000010000001000000100000110100011000001"
"7000000000001000011000010000110000100001100001000"
"7011111000001000011000010000110000100000100001000"
"7011111000011000010000010000100000100001100001000"
"7100000110000110000101000101100100110100011100001"
"7100000110000110000111000101100100110100011100001"
"7100010110000110000111000101100100110100011100000"
"7100000110000110000101000101100100110100111100011"
"7100010110000100000101000101100100010100001000001"
"7110000011000010000010100100011100000010000010000"
"7000000000000010000001111000100000000000000000001"
"7111011111111000010000010000100001100001000001000"
"7000000010000001000000000000000000001000000100000"
"7000001000001000001000000000010000000000100000100"
"7000000010000011000011100000110010011000000010000"
"7100000110000110000101000100110100011100000100000"
"7011110000001000001000010000110000100001100001000"
"7011110000001000001000010000100000100000100001000"
"7100010110000110000101000100100100100100011000001"
"7111111000001000001000010000110000110001100001100"
"7111111000011000010000110000100001100001000011000"
"7111100010011000010000010000100000100001100001000"
"7011111000001000001000011000010000100001100001100"
"7100000010000011111010001010010010010010010010100"
"7011111111111000010000010000110000100001100001000"
"7011111111111000010000010000110000100000100001000"
"7000000110000111000101100100110100011100001100000"
"7000000010000011000001100000110000011100001000000"
"7100000110000010000101000100100000100100011000001"
"7000000011111000011000110001100001100001000011000"
"7011111011111000011000110000100000100000100000100"
"7000000011111000011000011000110000100000100001100"
"7001000010111000010000010000110000100000110000110"
"7011111011111000111000110000100001100001100001100"
"7001001000001000010000100000000001000010000010000"
"7001111000001000010000100000000001000010000010000"
"7100000110000110000100000100100100110100010000001"
"7111111010001000010000110000100000100001000001000"
"7000000000001000010000000000100000000001000001000"
"7001111000011000011000110000100001000011000011000"
"7011111011111000011000110000100001000011000011000"
"7001111011111000011000110000100001100011000011000"
"7000000011111000010000110000100001100011000011000"
"7001111011111000011000010000100001100001000011000"
"7011111011111000010000110000100001000001000010000"
"7000011011111000010000100000100001000011000010000"
"7011110000011000011000110001110001000001000011000"
"7100000110000110000101000100100100110100010000001"
"7100000000111000001000010000110001100011000110000"
"7110000011111000001000010000100000100001000010000"
"7111000001111000001000001000001000010000100011000"
"7011100000001000001000001000010000000000100001100"
"7011111010011000010000010000000000100000100001000"
"7011111000011000010000010000110000100001000001000"
"7011111000001000010000110000100001100001000001000"
)+string("8000100010011010001010010010011100001110001010011"
"8010000011000101111100001000000011000001100000011"
"8000001011111100100100100000100000100000100011111"
"8011011011111000100000100000100000101011111010010"
"8011011011111100100100100100100100100011111011011"
"8010000011000000110000101000100011100000100000011"
"8010000101000101111100001100000011000000100000011"
"8000000001000001110001001000000011100000100000010"
"8010000110000101110100001100000011100000100000011"
"8000100011111010011011111011111010001110011011110"
"8000100011110100010011110011111010001010001011110"
"8010000111110100101100001000000011100000101000011"
"8000100001110010011011110011001010001110001011110"
"8000000011110110010110110011111010001010001011110"
"8000110011111010001011111011111110001110001011110"
"8000100011110110010011110011011110001110001011111"
"8001000011111010011011110011111110001110001011111"
"8000100011110010010011110011011010001010001011110"
"8001110011111110001011111011011110001110001011111"
"8001100011010110001011110011011010001010001011111"
"8001100010110010010011111001011010001011001001111"
"8000100011110010010111110011111110001110011011110"
"8010000111010100101100001100000011100000100000011"
"8001000011110010010011110011110011011011001011111"
"8000100011110110010011110011111010001110001111111"
"8000100011110010011011110011011110001110011011111"
"8000000011110110010011110011011010001010001011110"
"8000100011111011011010001011110010011010001011111"
"8000010001001001001001110010010010011110010010100"
"8000000001001001001001001001110010001100010110010"
"8001110010001110001110001011111110001110001011011"
"8001110010011010001010001011111010011010001011011"
"8010000011010101111100101000100011100000011000011"
"8001110010001010001010001011111010001010001010011"
"8001110010001100001010001011111010001110001010011"
"8000110010011110001010001011111010001010001011011"
"8001110010001000001010001011111010001100001011111"
"8001110010001110001010001011111110001110001010001"
"8001110110001110001110001011111110001110001110001"
"8011110010001100001110001011111110001110001010001"
"8000100010001110001010011011111100001100001011011"
"8010000101110100101100100000000011100000010000011"
"8000000011111010001010001011110100001100000011111"
"8001110010001010001000010010001100001010001011111"
"8000100011001010001011001010001110001110001011111"
"8000100011011010001011011011111110001010001011111"
"8001100010011010001010011011010100001110001011011"
"8001100010011010001010011011011010001010001011111"
"8000100011011010001011010011111010001010001011111"
"8001100010011010001010001011011110001100001010011"
"8001110010001010001011010011110010001010001011111"
"8001110011011010001010001011111110001110001011011"
"8000011011011011101100100100100111101011111000011"
"8000000011011010001010011011111010001110001011111"
"8000000001111001001001011001110010011110011111110"
"8000000000111001001001010001100010010110010011100"
"8000000000001011110010100011000011000000000001000"
"8000010000111001001001011001110010001110010011110"
"8000000110011110011110011011111110011110011011111"
"8000000010001010001010001011110010001010001011111"
"8001100010010000000000000011111000001000001010001"
"8000100010001010001010001001110011001010000001001"
"8000010011011011101100100100000011100011111000011"
"8001110010011100000100000011111000001100000010001"
"8000110001001100001000001000010001100000010001010"
"8001100110011100001100011011010001100001011001111"
"8001000010110110001010001001011000110000010000111"
"8001100010110000010011100001100011110110001010001"
"8010000110110100001110001000011000110000110000110"
"8001100011010100100000110011110001110001001001110"
"8000000011100110110000110001100001100001000101011"
"8000100011010110001110001001110001110001111011111"
"8000100000011110001110011010110001100011010010001"
"8000000010011100001010001010010000001000000011001"
"8111000110011110011011111001110001100010010001110"
"8000010010001010001001110000100001100010000011110"
"8000001110001010001010111001100011100010100011100"
"8001110010011010001010001011111010001010001011111"
"8001110110011110001110001011110110001110001011110"
"8001110111001110001110001011111110000110000011011"
"8001110011011110001110001011111110001010001011111"
"8011110110001100001100001111111110001110001111111"
"8000110010010110001010001001110010001010001001110"
"8001100010001000001010001011110000001010001010001"
"8001110110011110001010011011111010001010001011111"
"8001110110001100001110001011110100001110001011111"
"8001110010001010001011111011011000001100001011111"
"8000000000011011100100100100000100000010101011011"
"8000110011111011101100001100000100101001101011111"
"8000010010111011101000101100000100001000101011111"
"8000010011111011101100100100000100000011101011011"
"8000000011111011101100100100000100000001101011111"
"8000010011111000100100000100100000100011101011011"
"8000010011111000101000100100100100100111101011111"
"8001110011011010011011110011111110001110001011111"
"8000010011111001101000000100000100000011111010111"
"8000010011111000100100000100000000100011111000010"
"8000010011011010101100100100100100100011101010011"
"8000010011111101100100000100000100000011111001111"
"8000010011111001101000000100000000000011101011111"
"8000010011111010101000100100000000100011101011011"
"8000000011011010100100100100100100100011111001011"
"8000010011111010101100000100000100100011101011111"
"8000010011111011101100000100000100000011101011011"
"8000000011111011101100100100000100100011101011111"
"8001100010011010001010010010011110001110001011110"
"8000010011111011101000100000100001100011101010011"
"8000010011111000100100000000100000100011101000011"
"8000000011111011101100000100000000000011101011111"
"8000010011101000101000000100000000001011101011111"
"8000011011111001101000100100000100100011101011111"
"8000010011111001100000000000000000000011111001111"
"8000010011111011101000000000000000000011101010010"
"8000010011111000101100100100100000100011111011111"
"8000100010011000001011010011110100001100001011011"
"8001110010001010001001010001110010001110000010001"
"8000010011111011101101000100000101100011101011111"
"8000100010001010001011011001110010001110001011001"
"8001100010000010001011010001110010001100001010010"
"8000100010011010001010011001110010001100001011111"
"8000100010001010001010001001110010001100000011011"
"8000100011010010001011010011110010001110001011110"
"8000100010010010001011010011110110001110001011010"
"8001110010011010001011110011111110001110001011111"
"8000100010011010001011111011111110001110001011111"
"8000100011111010001011011011111010001000001011111"
"8001110011011010001011111010001010001110001011110"
"8000010011111011101101100100100101100011111010011"
"8000110011111011011011111011110010011110011011110"
"8001110011111011011011111011111010011011011011110"
"8001110110011110001011111110011110001110001011110"
"8001110010011110001011111010011110001110001011110"
"8001100010011110001011111010011110001010001011111"
"8000100111111111011011011011111110011111001111111"
"8001110111111111011011011011111011011011011011111"
"8001110011011010011011111011111110011010011011111"
"8001100111111110011011111011111110011110011011111"
"8001110011111011011011011011111010011010011011111"
"8000100011111011011011011001110011011111011011111"
"8011111011111011111111111110101110101011111011111"
"8000010011111011111111111100101111111011111011011"
"8010010011110101101000000000000011100001100000011"
"8000000011111011101010100000100010100011100001111"
"8000100001001001001001011000001010001110001010010"
"8000110001001001001001010011001110001110001010010"
"8000010011111011101100101100000101101011101011111"
"8000110001001011001001110010001010001100001010010"
"8000110001001010001001111010001110001110001110010"
"8000110001001001001001110000001010001110001010010"
"8000100011111010001011011010011110001110001011111"
"8001100011011010001011111011111010001010001011111"
"8000010000011011101000100100100000100011101011111"
"8000010011111011101000100100100000100011101011111"
"8000010011111011101000100100100100100011101010111"
"8000100001111011011011110001000010100111100011100"
"8000010000101001001001111001001010001100011010110"
"8000010011111010011010110001100011111010011011110"
"8000010011111011101100100100000000101011101011111"
"8000110011001010001010001001110010001110001011011"
"8001110011011010001010000011110010001110001011011"
"8000100011111010001011011001110010001010001011111"
"8000100010010110001010010011110010001000001010011"
"8001100010010110011010010011110010001110001011010"
"8001100010001110011010010011110100001100001011110"
"8000100010011010001010010001110110001010001011110"
"8000001000101001001001011001001010001100001010110"
"8000000010010010011010010001110010001010001011111"
"8000100011011010001010011011110110001110001011111"
"8000100010011010011010011001110010011100011011110"
"8001100010010000010010010011110110001110001010010"
"8000100011011110001010001001110010001010001011001"
"8000100011110010001010011011110010001110001011110"
"8001100010001010001010010011110110001100001011110"
"8001100010011010001010010011110110001110001011110"
"8001100011011010001010011011110100001100001011111"
"8000100010011010011010011001100010001010001010011"
"8011111010011011011011111110001110001110001010011"
"8000100011111010001010011001110010001110001011110"
"8001000010110010011010010011110110001110001011011"
"8000100010000010001001010001010010001010001001110"
"8000100010010110001010001001110010001110001010001"
"8001100011110110011010011001110010011100001011111"
"8000000010011010001010010010110100001100001011111"
"8000000011010010001010010011110010001100001011110"
"8001100010010110011010010011110010001010001011110"
"8000100011011110011010010010011100011010011011110"
"8000000011110010010010010011110010011000001011110"
"8000000011011010001011011010011110001010001011111"
"8000100011111010001110001001110010001010001011111"
"8011100110110110011110010011110100011100001011110"
"8000100011011010001011011011110010001010001011011"
"8001000010010010001010011001110110001110001010011"
"8001100011010010001010010011110010001110001011110"
"8001000011011110001010011001110110001110001011111"
"8001100010010110001010010011010110001110001010010"
"8000110011111110001011011011111010001110001111111"
"8000100011111010001011111011111010001110011011111"
"8001100011111110001011111011111110001110011011111"
"8001000011011010011011011010011110001110001010011"
"8010011011111100100100100100100101100011011010011"
"8000010011011000100000100000100001100011011001011"
"8000000011011000101100100100100101100011001010011"
"8000010011111011101000100000000000100011101011111"
"8001110011011010001010001011110110001110001011011"
"8001100010011010001010010010011000001010001011110"
"8010110011111111101101001100001111101011111000110"
"8000100010001010001011010001010010001010001010011"
"8000000010011010001010001011011110001110001011011"
"8001100010011010001010000010011010001110001011011"
"8001110010011010001011010010011100001100001010011"
"8000000010011010011011110010011110001110001011011"
"8000100010011010001011011010011010001010001011111"
"8000000010011010001010001011111010001010001011011"
"8000000010011010001010011011110010001110001011011"
"8000100010001010001011011011011110001110001011011"
"8000100010011010001010000011110010001100001010011"
"8000100010011010001010010011011010001010001011010"
"8000000010011110001010011011011100001100001011111"
"8000010011111001101101101101001101101011101010011"
"8000100011011010001001010011010010001110001011111"
"8001110010001010001011110011011010001110001011111"
"8000100011111010001010011011111010001110001011111"
"8000111010001101000000111000001000100000100000011"
"8000001000100000001000110001010010010000000111100"
"8000011000100000001000110001010010010000010110100"
"8000000000100000100000111001110010010110010100100"
"8000010011011011101010100100100010100011101011011"
"8000010011111011101100100100100100100011111011011"
"8000010011111011101000101100101110101011111011011"
"8001110011011010001010001001110010001110001011011"
"8001100010001010001010001011110010001110001011011"
"8000100010011010001010000001110010001110001010011"
"8000011011011011101100100100100000100011101011011"
"8000010011011011101000100000100000100011011011011"
"8000100011111010001011001011111010001010001011111"
"8001110011111110001011111011111010001110001001111"
"8000100011111010011011011011111110011010011011110"
"8000010011011010101100100100100000100011101001011"
"8000000011111011101100001100001101101011101010011"
"8000010011011000100100100100100000100011101011011"
"8001010011011000100100100100100100100000100011011"
"8011011000000100100100100100100100100100100011001"
"8001011010001100100100100100100100100000100010001"
"8011010000001100100100100100100100100000100011011"
"8001100011011010011011110011111110001110001011111"
"8000000011111010011011011011111110001110001011011"
"8001100011011010011011110011011110001010011011111"
"8001100011111110011011111011011110001110011011111"
"8000100011111010011011011011111110011110011011111"
"8000000010001010001011010001100010011010001010011"
"8000000010000000000010000010010000000100000011111"
"8001000010000000001010000001110100001100001010001"
"8000100010001010001010000011010110001100000010001"
"8000000011011000100100100100100100100000100011011"
"8000000010000010001010000001110000001000000011011"
"8001100010001000000010001001110000000100000010011"
"8001100010001000000010011001011100000100000010001"
"8000000010001010001000000010000000000000000010010"
"8000000000011000100001001011110110100111100011000"
"8000001000111001001001001011110010100100100011100"
"8001100011111010001010011011110010001010001011110"
"8000001000111100100000111001100010100000100111000"
"8110000000111001001001001011110111100110110011100"
"8000001100111100101001110011100011100011100011100"
"8000000000111001100001101011110111100110100111100"
"8000001000011000100000001001110011010110100011100"
"8000000000111100100000101001110010100010100011100"
"8000000000111000101000101001100011110110110011100"
"8000000000011010101101010001100011100111100011001"
"8000001100111101101001110001100010100111100111000"
"8000001100111101101001111001110011100111100011100"
"8000010001111001001001010001100011100110100011100"
"8000001000111010101100010000100010101100101111000"
"8000101000111000110101000011011001011101000010000"
"8000000000001000000000010000100001101011101011001"
"8000001010101100001000110000100010101110101011001"
"8000001000100101000001000000100001100000100101000"
"8000001001001001111000011010000010000100000100000"
"8000011000101001001001111011110010000010100011000"
"8101000000011000100001011011100010100100100011000"
"8000000100111101100001001011111111110011110011100"
"8001110010001010001011010001110010011110001011011"
"8000100010001110001011010001110010001110001011011"
"8000110001001001001001010001010010011110010011110"
"8000100011011110001011011011011100001110001010001"
"8001100010001010001011110010011100001100001011011"
"8001100010001010001001110010011010001000001010011"
"8000010001001010001001011011011110011110011111110"
"8000010001001011001011011011110010011110010111110"
"8000111001001011001001110011011010011100010111110"
"8000100010011010001011110011011110001010001011111"
"8001100011010010000011010011010110001010001011010"
"8000000011111110011011110011011010011110001011011"
"8000100011111010011001111011011010001110001011011"
"8000000011011010001011111011011010001010001011011"
"8000100010001010001011011010001110001110001011011"
"8001110010011010011011110011011010001010001011111"
"8000110011001010001001110011001010001010001001111"
"8000110011011010001011111001011110001010011001011"
"8000100010011010001011111010011010001110001011111"
"8000100011011110011011110010010010011010001011011"
"8001110011011010011011110011011010001010001011111"
"8000000011111010001001110011011010001010001011111"
"8000100011001010011001110011011010001010001011111"
"8001110010001010001001110011011110001010001001111"
"8000100011011010001001110011011010001110001011111"
"8001110010001010001010010010001110001110001011111"
"8000100011011010001011011011011010001010001011011"
"8000100011011010001011011011011010001110001011011"
"8001110011001010001001110011011110001110001011011"
"8000000011111010001011011011011010001010001011111"
"8000110011011010001001110011111110001110001011111"
"8000110011011010001011110011011010001010001011111"
"8000000011011110001110011110010111110000110000110"
"8000000011010101111100101100100011100011101000111"
"8011000011111100101100100100100011100011111000111"
"8010000111010100111100001000000011100000100000011"
"8000000011111101111100101100100011100011111000111"
"8000100011111010011011111011111110011010001011110"
"8000000010010110001010010001110000110001001001010"
"8000110010001010001001110011001110001110001011001"
"8001100011010010000011010010010100001110001010011"
"8000110011001010001001100011001010001010001011001"
"8001110010011010000011110010001100001100001011110"
"8000100010001010001011010010010110001110001010011"
"8001110011011010001001010010001010001010001011011"
"8001110010001010001001110010001000001010001011111"
"8010000011010101111100001000000011000011100000011"
"8000100001010010001011010010001110001010001011011"
"8001110000000010000001110010001010000010000001110"
"8001110011001010000001110010000010001010001011000"
"8001110010001010000011110011011110001110001011110"
"8000100010000010001001010001110010001110001010011"
"8001110010001010001001111001001110001110001010001"
"8001110010001010001001110000001110001110001110001"
"8001010010001010001001100000000000001010001001111"
"8001110010001010001001110010001010001110001001111"
"8010000111010100111100000000000011100000111000011"
"8000000001001011001001000001011010001110001011001"
"8001110010001010001011110010001110001110001011110"
"8000000001011010011001011001011010001110001001011"
"8001110011011010001011001011001110001010001011001"
"8000010011111010101000100000100000101011111011011"
"8000001010100101001100111011001010001010001010110"
"8000100011110010001011110011011110001010001011110"
"8001100011110110010011110011111110001110001011110"
"8010000011000001000011011011101100100101000011011"
)+string("9011110100110100101100001010000010100001101000001"
"9001100010001100001010000011001000000000000011011"
"9000100001001010001010001010011000101010010010100"
"9000000011110011111000101100000100000010101011101"
"9000100011110010101000101100000100000010101011101"
"9000100011110110111110111011110000100001000001100"
"9001100010010110001110001010001001011000001011110"
"9001110011010010001010001010011000101000001011010"
"9011110010010100011110011011111000011100010011110"
"9000000001101010001000011000010011100000100001000"
"9000000011110110011110011010011001111000011011110"
"9001100011010110001100001010011001111000011011110"
"9001100010010100011100001010011000001110011011110"
"9001100011010000001010001010011001101000001010010"
"9000100011011110001110001010011000001010011011110"
"9001100010001110001110001011011001101010011011011"
"9001110011011110001110001011011000001000001011110"
"9001100010001110001110001011011000001110001011110"
"9011100010011110011100011010011000011100011011100"
"9000010001101010001010011100010011110000100000100"
"9001000011100010110000101100101000100011100011000"
"9001100011110010101000000000000000100010101011001"
"9001100011110010101100001100000000001010101011101"
"9000100010001100000000001001010000010000000000100"
"9001110110001110001110001011001000010000000001100"
"9000100011011110001110001011001000110000110011000"
"9000000110011100001110001011111000001010010011110"
"9000000111010100001100001110011000011000001110010"
"9001100110110100001100001110011000011000010111110"
"9001110011011110001110001011011001001110001011011"
"9000100010001110001110001010011000011000010011000"
"9001100010001100001100001010011001001000001011110"
"9001100010001010001010001010011000001000001010011"
"9001100011110010011010011011111000101000011011110"
"9001000011110010111000101100000000100000100011100"
"9001100011110010101100001100000000000010100011100"
"9001100011110010110100001100000000000010100011100"
"9001100011110010110100001100001000001010100011100"
"9001100011110010001100001100000000000010100011100"
"9001000011110010111000001100000000100000100011100"
"9000110001001010001010001011011000010000010011100"
"9001110010100000101100101100100100100000100011000"
"9000000010110000101100100100100100100000100001000"
"9011100010110100100100101100100100100000000011000"
"9011100010110000101100101100100000100000000011000"
"9001100010010000001100001000001000001000001010010"
"9001110010001100001100001010001000001000001010010"
"9001110010001100001100000010000001100010001010010"
"9001100010010000001000001010001000001010001000010"
"9000000010000100001100000010001001100000001010010"
"9000110001011011001010001011010001010000010110100"
"9001100010001100000100000010001000001100001010001"
"9000000010010100001100000010000000000000001010010"
"9000100010000000001100000010000000000010001010010"
"9000000000010000001000000010000001100000001010010"
"9000000010000000001100001010011000001100001010010"
"9000011101111101001000001001111000010000100111000"
"9000011000111101001100001001101001010000100011000"
"9000011100101101001101001001101000010000100011000"
"9000010101111001001010001001101000010001100111000"
"9000111001111011001010001011111001110000100111000"
"9000010001111011001010001011111001110111100111000"
"9000010001111011001011101001011000110001100111000"
"9000011000111101001001001001111001110001100111000"
"9000010000111101101001011001111001110001100110000"
"9000110101111111001010001011111001110001100110000"
"9000010000111101001101001001110000110001100110000"
"9000010000101001001001001001101000010000100011100"
"9000100001110010010010110011100000100001100110001"
"9000110001011010001010001011111001010000010011100"
"9000010010111111101101001001110001110001100111000"
"9000010001101000001000001001111000110001100011000"
"9000100010011100001100001010001001101000001010010"
"9000100011001000001010001010001000101000001010010"
"9000000011100100111100001000000010100001101000001"
"9000000001111010001010001011011001011110010111110"
"9000110001001011001010001011111001011000010111110"
"9000010001001010001010001011011000001110010111110"
"9000110011011010001010001001011000001010011001110"
"9000110011011110001010001011011000001010001011111"
"9000100011111010001010001011011000111010011011110"
"9011100011110000111100001100010010100011100011101"
"9001100011110110111100001100000010100011101001001"
"9011100011110010111100001100000010100011101001001"
"9011100011110010111100001100001010000011101001101"
"9011000011100000110100101000000010000011100001001"
"9001100011110100111100001100000010100011101001001"
"9001100011110000101100000100000100000000101011001"
"9001000010011110001011111010011011110001000000100"
"9000000011110110011110001011111001111010011011110"
"9001100011111110001110001011111001111110011011110"
"9000100011111110001110001011111001111010001011111"
"9001100011110010001110001011111001111010011011110"
"9000000010001000110001100000100000000001000001000"
"9000000011111010101000000000000000000010101011101"
"9011000011100000110000001000000010000011100000001"
"9000100010011110001110001010011000111010011011110"
"9000100010010110001010001011011000111010010011010"
"9000100011011010001110001010001000001000001011111"
"9000110011010010001110001011001000001000001011010"
"9000110001011010001010001001111000001000001001111"
"9001110011110011111100101100000100101011101011101"
"9000000010100011001101001100011000100000010010100"
"9000100011011110001110001011011000101110000000010"
"9000100011011010001010001010111000001000001000010"
"9000100011110010001010001011111000001010001011110"
"9011000011100100111100101000000010100001101000001"
"9000100011111110001010001011111000001000001010010"
"9010000100100100110100101000000010100001100000001"
"9010000001100100110100001000000010000001100000001"
"9001110011111110001110001011111000001000011011110"
"9001000011110110011011011001111000011000011000110"
"9001000011110110011011111001001000011000010000100"
"9000100011111010001110011011111000011000010000100"
"9001100011111010011111111011111000011000110001110"
"9001100011111110001011011001111000111001110011100"
"9000100011111010011110011011111000011000110001100"
"9001100011111011111000001100000000100011101011101"
"9000100011111010001110001011111000001000001011110"
"9000110001001010001010001011011001000000010110100"
"9000010001001010001010001010011000000100010110100"
"9000010001001000001010001010011001101000010110110"
"9000010001111001001000001001011001101000010011100"
"9001100110001110001110011001111000001000001111011"
"9000100010001000001000000001001000000000000010011"
"9000100010000000001000001010001000101000001010010"
"9000000010000100001100001010001000101000001010010"
"9001110011111011101100000100000000100011101011101"
"9000100010001000000100000010000000100000000010011"
"9001110010011000000100000010000001111000000011111"
"9000100010001110111000011000011000001000001000001"
"9000110011010100010111110000010000011000011000011"
"9001100110000100110000001000001000001000010000010"
"9001000010000100110011110000010000010000001000001"
"9000000110011100001010001011111000001000001011010"
"9011110110011100001110001011111000001000001110001"
"9001110010001010001010001000001000001010001011111"
"9001110110011100001100001011111000001000001011111"
"9001110110011100001100001001111000001100001111111"
"9001110110001110001010001001111000001000001011011"
"9001110110001100001110001001111000001100001111111"
"9000000011110011111000000100000000000011101011101"
"9000100011110010101000001100001000001010101011001"
"9001100011110000101100101100000100000000101011101"
"9000110011001010001010001011011000001000001000011"
"9001110010001110001110001001111000001000001011110"
"9001110010000100001100001010011000001000000000010"
"9001000010001110001110001011011000001000001011110"
"9000100010011000000000000011000000000000000011011"
"9000100011011110001010001011111000001110001011110"
"9001100111111110011110011111111000111001110011100"
"9001100011111110011011011011111000111000110001100"
"9000100011110011001101000101000000000011000011000"
"9000110001001011000010000010000010000010000001000"
)+string("a000001000001000010000100000100001000011111010000"
"a000100000100001110001010011010011111011111110001"
"a001100010010100001000001010001100001100001010101"
"a000100000010000001000000000000011111000000000000"
"a000000001000000000000000000000011111100000100000"
"a000100000110001010001010011011011111010001010001"
"a000000000001000110011010110010011110000111000001"
"a000000000011000110011010110010011110000110000001"
"a000100011110110010000110011010110010110010011111"
"a001000011110110010000010011110110010110010011111"
"a000000011110010010000010011110110010110010011111"
"a000001000001000011000100000000001111010000010001"
"a001100011110010010000010011110110010110010011111"
"a001110011110010011000011011111010010110010011111"
"a100000000111001001001001010010000010100110011110"
"a000010000101001001010010010010100010100110011111"
"a100010100111001001001001010010000010100100011111"
"a000001000110000000001001010010010010000100111111"
"a100011000101001001010011010010100100100100011111"
"a000011000101001001000010010010000010000100011011"
"a000010000111001001010001010010000010100100011101"
"a100011000101001001010001010010000010100110011111"
"a000100000100001010001010000010011110010001010001"
"a000010000101001001010000010010000010100110011111"
"a000000011111111101100101100100010101010111000011"
"a000000011111011111100101100101010101010111010111"
"a001110010001010001000001011101010001010001011111"
"a001000011110010010000010100010010010010011011111"
"a000000001111010011100001100001010001011011001111"
"a000000011111010011100011100011010011011111011111"
"a001110011111010001110001010001010001011111001111"
"a000100001100001100001010010010011110010001110001"
"a000100011111010001010001110001010001011011011111"
"a000000011111010001010001110001110000011111011111"
"a001110011111010001010000110010010000011111001111"
"a001100011110010011100001110001110001011111011111"
"a000000011111010001010001010001110001011011001111"
"a000100011111010001010001010001010001010011011111"
"a000100000100000100001110001110001110010011010011"
"a001100011111010011110001110000010001011111001111"
"a000111001111001001000111001111010011110011011111"
"a000010001111011001000011011111110011110110111111"
"a000000001111001001000011001111010011110011111111"
"a000010001111001001000011011111010011110111111111"
"a001100001100001100011110010010011110010011100001"
"a000000000001000111001110011010001010000010000000"
"a000100000100001110001010001010011110011001110001"
"a000000000100001010001010000010011111010001100001"
"a001110011111000001000011011001110001110011011101"
"a000000011111010001000111011111110001110001011111"
"a000100111111000001001111011001110001100011111111"
"a000100011111110001000011011101110001110001111111"
"a000110011011000001000111011101010001110001011111"
"a000001001111011000010000010000001000000110000011"
"a000001000111011110110000011000001100000110000011"
"a000001000111011100110000011000000100000010000001"
"a000000000010000010000111001010001011000001010001"
"a000000000010000110000110001010000011010001110001"
"a000000000010000111000111001011001001010001010001"
"a000000000010000110000110001010001010010011000011"
"a000000000010000110000111000011001001001001010001"
"a000000000010000010000110000111001011001011000001"
"a000000000010000011000111000111001011001011010001"
"a000000000011000110011010110010011010000110000001"
"a000000011011010011000011011011110011110011011101"
"a011000011111000011000001011111110001110001011111"
"a001000011111010001100001110001100001010011001110"
"a000000011111111111100101100100100100010101010011"
"a011111011111010001100101100100100100010101000011"
"a000000011111011111100101100101000101010111010011"
"a001110011011000001000111011001110001110011011111"
"a001110011111010001000011011101110001110011011111"
"a001110011111010001000111011101010001110011011111"
"a001110011011010001000111011001010001110011011101"
"a001100011111010001000111011101110001100011011111"
"a001100011011000001000011011111100001100001011111"
"a001000011111110001000111011001100001100011111111"
"a001111011111111001100001100101110101010111000010"
"a000001011111101001100001100100110101010111000011"
"a000000011111011111100101100101010101010111000011"
"a000000011111011111100101100100010100010111000011"
"a000000011111011001100001100101010101010111000011"
"a000000011111011111100101100100110100010101000011"
"a001100011010000000000001011001100000100011011001"
)+string("b010000110010100011100010111110100001100001111111"
"b000011011111001100000000000000100000011111011111"
"b000010001111011101000100000000100000100000111111"
"b000111011101000000000000100000100000110101011111"
"b000010011111000100000000000000000000100000011111"
"b000011011101101000100000100000100000101111011111"
"b011100010011010001010001011111010001010001011111"
"b011100010011010001010011111110110001110001110011"
"b011100000001100000100001100010100000100000100000"
"b011100000001100000100001100010100001100001100010"
"b000010011011000100100100100100100100100100111111"
"b010000110000111100111111110001010001010001111110"
"b000000110000011110011111010001010001010001111111"
"b011100011111010001011111011111010001010001011111"
"b000001000110011000011110010010011110010001111110"
"b010000010000011110111111010001010001010001011111"
"b010000011110000001011001011111000001010001111111"
"b010000011110000001000001011111010001010001011111"
"b011110010011010001010011110011010001110001111111"
"b010000010000011100010011010001010001010001111111"
"b011100110011010001111111011111110001110001111111"
"b011100011110010001011011011111010001010001011111"
"b000000110011100001111111100001110001010000010000"
"b011000011111010001010011011111010001010001111111"
"b010000010000010100011111010001010001010001011111"
"b111110100010100001110010100001100001100001111111"
"b011100100011100001000001011111000001110001110011"
"b100000100000100100110011100001100001100001110010"
"b010000010000011110010001010001010001010001111111"
"b111100110011110011110011111111110001110001111111"
"b110000110000110000110001110001110001110001111010"
"b000000110000010000010010000001100001000001010011"
"b000000110000110100110010110001110001110001111010"
"b000110000101010000011000001000101000000000011111"
"b000011000011000000011100010100100100011100011111"
"b000000000111000001011000001000101000100000111111"
"b001110001001001001010001011110010000010000010010"
"b000110001001001001011001011110010011010010010010"
"b000010001001001001011001011110011000010010010010"
"b001100001001001001001001011110010011010010010010"
"b011100010011110001000011011110110001100001110011"
"b001110001001001001001001001110011001010011010010"
"b001110001001011001010001010010010011010011010010"
"b011110001001001001011001011010010001010011010010"
"b001110001001001001001001001110010011010011010010"
"b000010001001001001000001010010010001010011010010"
"b000010001001001001011001011110010001010011010010"
"b100000100000100100110011100001100001110001111110"
"b000010011111011101100100100100100100100100111111"
"b000011011111000101100100100100100100100100111111"
"b011100100001100001100001111111100001100001100001"
"b010000110000010000011110010001010001010001110010"
"b010000010000010000011111010001010001010001011111"
"b000000110000110000111110110001110001110001111111"
"b000000010000010000011110010001010001010001011110"
"b000110000111001001001000001000001000111101111111"
"b010000110000110000111110010001010001010001011111"
"b111100100011100001110010110010100001110001111110"
"b000000110000010000011110110001010001110001111011"
"b000000110000110000111110110001110001110001110011"
"b000110000111001001001000001000000001000101111111"
"b000010000111000101001000001000000000111111111111"
"b000010000111000101001000001000000000011111111111"
"b000110000111001101001000001000001001111111111111"
"b000000000000000000001010010001000000010001000001"
"b010000111110110010110110111111110001100001111111"
"b000000110000111110111111010001110001010001111111"
)+string("c000110011111110001010000110000110001011001011111"
"c001100011011010001110000110000010001011011011110"
"c001100010001110001110000110000110000010001011011"
"c000000000001010001100000000000010000011001001110"
"c000000010001010001000000000000010001010001011111"
"c000000000001100000100000100000100000100000010001"
"c000100010001000000100000000000100000000000010001"
"c000000011001010001100000100000100000010001011011"
"c000100010001010001100000100000100001010001011001"
"c000010011011000001100000100000000001010011001110"
"c000100010011110001100000100000100000010001011011"
"c000010011011010001100000100000000000010001011110"
"c000000000000000000010000010000001000001001000111"
"c000000100000000000000000010000011000001101000110"
"c001110011111110001110000110000110001010001001110"
"c000010010011010001100000100000000001010001001110"
"c000000011000110000010001010001001000001100100111"
"c000010010010010001100000000001010001011111001110"
"c000110011011010001110000110000010000010001011011"
"c001100011011010001100000100000100001010001011110"
"c000100011111010001010000110000010001010001001111"
"c000010010011010001100001100001010001011111011110"
"c000010010011010001100000100000010001011111001110"
"c000000010011010011100001100001110001011111011110"
"c000010010011010001100001100001010001011111001110"
"c000000001011010001010000110000010000010001011011"
"c000100011011010001110000110000110000010001011011"
"c000000011011010001100000100000000001010001001110"
"c000000000001010001000000100000000000010001001110"
)+string("D000000000001000001001110000010010000000100101100"
"D000100011110010001000001000000000000000000011111"
"D100000100000100001001110010010000000100100101100"
"D000001000001000001001101010011100001010001010011"
"D111111111111000101001000001000001001000111000111"
"D111000010001010000010000010000010000010000010000"
"D010000010001010000010000010000010000000001000010"
"D010100110010110001010000010000010001010001011110"
"D000000011110010000000001100000100000100000011111"
"D000100011111010001100001100000100000100000111111"
"D000100011111010001000000000000000000000000011111"
)+string("e011111100000100000110000100000100000100000111111"
"e000000100000100000100000100000000000011111011110"
"e011111111111010000011110011100010000010000011111"
"e001010011011011001100001100001100001011001011110"
"e000000011011010001100101100101010101011111001110"
"e000000011001011001100001100000010001011111011110"
"e001000011001011001100001100001010001011111001110"
"e001000011101010101000101000101010101011111001110"
"e001011110000100000100000110000100000100000100000"
"e011111000000000000000000000000000000100000000000"
"e001110011000110000110000111100110000110000111111"
"e000100011110010001111111110000100001010001011111"
)+string("F000000000100001000001100001000000000010000000000"
"F000000100100100100100100100100100100110111111111"
"F000000100100100100100100100100100100100100111111"
"F000000100100100100100100100100100100111111111111"
"F000010001100001000111111001000001000001000001000"
"F000100111111000000000000011111000000000000000000"
"F011111111111110000110000111111110000110000110000"
"F011000100000100000100000100000100000100000100000"
"F010011000000100000100000000000100000100000100000"
"F000000100100100100100100100100100100111111011111"
"F001111111111110000111110111110010000110000110000"
"F000000000100100100000100000100000100011111011111"
"F100000100000100100100000100000100000111111011110"
"F000010001110001110111111000100001100001100001100"
"F000000000000000000000000000100000100000100011111"
"F000000100000000000000000000100000100000100011111"
"F000000000000100000100000000100100000000000011111"
)+string("g001100010001100000100000100000100000100000010000"
"g000100010000000000000000000000000000100000010000"
"g000000010001000000000000100000100000000000010001"
"g000000001111010000100000100000110000010000001111"
"g000100000101010111000100000000000000010001001110"
"g000000001000010000000000000000001111000001001010"
"g001110010001100000100000100001100000000000001000"
"g000100010001000000000000100000000000000000000001"
"g001100010011110001110000110111110001010001011011"
)+string("H000010000100000100001111001001010001010001100000"
"H010000010000010000110000011010010010010010010011"
"H000000010001010001011011011111010001010001010001"
"H000001110001010001111111111111110001010001010001"
"H000000000001000001000001011001010001000001100001"
)+string("I000011000011000111000110001100001100011000110000"
"I001100011110000100000110001100001100001110111111"
"I011110011110001100001100001110001110001100011111"
"I011111000100000100000100000100000100000100000100"
"I000000111000111100111110011111001111000111000001"
"I100000100000100000111111111111100000100000100000"
"I100000100000100000111111111111111111100000100000"
"I100000100000111111111111111111100000100000100000"
"I000000111110000100000100000100000110000100011111"
)+string("J000100111111000001000000000000000000000001000001"
)+string("k010000110000110000010000010010011100110110110011"
"k000001110010010100011100011000011100010110010011"
)+string("L001100001000001000001001001000001000011000011001"
"L000000110000110000110000110000110000110000110000"
"L000000010000000000000000000000000000100000100000"
"L000000100000100000100000100000100000100000100000"
"L000000100000100000000000100000100000100000100000"
)+string("M010001110011110011111011111011111001110101110101"
)+string("N000110011001010001010000010000110001110001010001"
"N000000011111000111001110001100011000011111000000"
"N000000010000001000001000000100000010000010000001"
"N000000000000100000100000100100100000100000100000"
"N000000010000001000100000100100100000100000100001"
"N000000010000000000000000100000000000000010100001"
"N000000000000000000101000100000100000100000100001"
)+string("P011100010011010001010001011111010000010000010000"
"P011100111111110001110001111110110000110000110000"
"P011100110011110001110001111110110000110000110000"
"P000000010011010001010001011111010000110000110000"
"P011000011000011100100100100100111111111111100000"
"P000000011100000100100100100100000100000100011111"
"P001000011100010100100000000000000000000001011111"
"P001000011100000000000000100000100000100000011111"
"P001000011100000100000000100000100000000000011111"
"P000111010000000001000110001100000100000100000010"
)+string("Q001100010010110001110001110001111101010010000111"
)+string("R011110001001011001011001011110010000010010010011"
"R000001000001011111000111000100000100111101111111"
"R001000111111100001100001111110100010100010100001"
"R011100111111000001000001011110000010000010000001"
"R011000100001100000100001100100100000100010100001"
"R000000100001100000000001100000100100100010100001"
"R011100000001000000000000000000000000000010000001"
"R011100100001100000100000100000100100100000100001"
"R000000100000100000100000101100100100100010100000"
"R000000111111010011010011011110010110010010010011"
"R000000011001011111100110100100100100100100111111"
"R011011011111100100100100100100100100100100111111"
"R011011111111100100100100100100100100111111011111"
"R000001011111000100000100100100100100100100011111"
"R000000000001011000010111100100100100100100111111"
)+string("S000010010111000100100100100100100100000001011011"
"S000010010111110101100100100100100101110001011011"
"S000010010011010101000100100000000001011001011011"
"S000010010111110101100100100100100101101001011011"
"S000110010001100000000000000000000000000000001000"
"S000000010001110000001000000010000001000001000010"
"S001100000001100000010000000000000001000001010000"
"S000100010001100000010000000000000000000000010000"
"S000000010000100000010000000010000000000000010000"
"S000100010001100000010000000010000000000000000001"
"S000010010011010101100100100100100000011001011001"
"S000000010011110000011000000110000001000001010011"
"S001110010001110000001000000011000001000001011111"
"S001100010011110000011000000110000011000001010011"
"S000100011111011001011100000111010001110001001111"
"S000000011110010011011100001111000011110011011110"
"S000010010011010101100100100100000100000001011000"
"S001100011111100001011000001110000001000000011011"
"S000010010111010101100101100100000101011001011011"
"S000010010011010101000100000100000100011101011011"
)+string("t000010000100000100000100000100000100000100000100"
"t011100000100000000000000000000000100000000000000"
"t010000001000001000001000001000001100001000001000"
"t010111000000000000000000000000000000000000000000"
"t011111011111000100000100000100000100000100000100"
"t010000011111000100000100000110000110000110000110"
)+string("u011001010001010001010001010000100010100010010100"
)+string("V100000110000101000000110000011101110111000100000"
"V100000100000100001110010110000111100111000110000"
)+string("W000000000000000000000000010000010000011011010001"
)+string("X100000110001011011001110001100010011110001100000"
"X100000110001011011001010001110011010110011100000"
"X100000110001010011001110001100001010110001100000"
"X100000110001011011001010001110011010110001100001"
"X000000110001010011001110001100011010110001000001"
"X000100010001001010000100000100000010000000010001"
"X111011011000001010001100000100001110000010010011"
"X000000010001001010001100000100001010010010010001"
"X000000010001001010001110000100001010011011010001"
"X000000110000011001001111000110001111011001110000"
)+string("Y110000010001001011001010000100000100000100000100"
"Y111001011001001000001110001110000100000100000100"
"Y100000110000001000000111001111111100111000100000"
"Y100000000001010010000000000000000000000000000000"
"Y100000000001010010001000000100000100000000000000"
)+string("Z111111000001000000000000000100000000010000100000"
"Z011000000001000000000000000100001000010000000000"
"Z011000000000000001000010000100001000010000000000"
);

////////////////////////////////////////
// BEGIN MAIN CLASS
////////////////////////////////////////
class PatentLabeling
{
public:

  IplImage * img;
  SVMPredictor * sp;

  inline double similar(string & a, string & b)
  {
    double total = 0;
    double inter = 0;

    for (int i = 0; i < a.length(); ++i)
    {
      if (a[i] == b[i] && a[i] == '1')
        inter++;
      if (a[i] == '1' || b[i] == '1') 
        total++;
    }
    if (total == 0) total = 1;
    return inter / total;
  }

  vector<string> patterns;
  vector<string> anss;
  PatentLabeling()
  {
    img = NULL;
    for (int i = 0; i < pattern.length() / (1 + FEADIM); ++i)
    {
      patterns.push_back(pattern.substr(i * (1 + FEADIM) + 1, FEADIM));
      anss.push_back(pattern.substr(i * (1 + FEADIM), 1));
    }
  }

  string recong(IplImage * img)
  {
    vector<pair<int, double> > feature = extractFeature(img);
    string query(FEADIM, '0');
    for (int i = 0; i < feature.size(); ++i)
      query[feature[i].first - 1] = '0' + feature[i].second;

    string ans(1, ' ');
    string mostsimi;
    double maxsimi = 0;
    for (int i = 0; i < patterns.size(); ++i)
    {
      double nowsimi = similar(query, patterns[i]);
      if (nowsimi > maxsimi)
      {
        maxsimi = nowsimi;
        ans = anss[i];
        mostsimi = patterns[i];
      }
    }

    /**
    output(query);
    cerr << ans << endl;
    output(mostsimi);
    cerr << endl;
    */

    return ans;
  }

  void output(string & s)
  {
    for (int i = 0; i < FEAH; ++i)
    {
      for (int j = 0; j < FEAW; ++j)
        cerr << s[i * FEAW + j];
      cerr << endl;
    }
  }

  vector <string> getFigures(int H, int W, vector <int> image, vector <string> text)
  {
    img = constructImage(H, W, image);
    IplImage * binary = cvCreateImage(cvSize(W, H), 8, 1);
    getBinaryCharImage(img, binary);

    vector<CvRegion> regions;
    vector<vector<int> > coms(H, vector<int>(W));
    floodFill(binary, coms, regions, 8, 100, 1000000);

    for (int i = 0; i < regions.size(); ++i)
    {
      regions[i].valid = false;
      
      if (regions[i].points.size() > 0)
      {
        if (regions[i].rectH > H / 10 || regions[i].rectW > W / 10) regions[i].valid = true;
        if (regions[i].rectH > regions[i].rectW * 5) regions[i].valid = true;
        if (regions[i].rectW > regions[i].rectH * 5) regions[i].valid = true;
      }
      if (regions[i].valid) 
        regions[i].text = "1";
    }

    vector<string> ret;
    for (int i = 0; i < regions.size(); ++i)
    {
      if (regions[i].valid) 
      {
        ret.push_back(regions[i].toString(H, W, false));
        cerr << regions[i] << endl;
      }
    }

    cvReleaseImage(&img);
    cvReleaseImage(&binary);
    return ret;
  }
  
  vector <string> getPartLabels(int H, int W, vector <int> image, vector <string> text)
  {
    img = constructImage(H, W, image);
    IplImage * binary = cvCreateImage(cvSize(W, H), 8, 1);
    getBinaryCharImage(img, binary);

    vector<CvRegion> regions;
    vector<vector<int> > coms(H, vector<int>(W));
    floodFill(binary, coms, regions, 8, 10, 10000);

    for (int i = 0; i < regions.size(); ++i)
    {
      if (regions[i].rectH > H / 10 || regions[i].rectW > W / 10) regions[i].valid = false;
      if (regions[i].rectH > regions[i].rectW * 5) regions[i].valid = false;
      if (regions[i].rectW > regions[i].rectH * 5) regions[i].valid = false;
      if (regions[i].points.size() == 0) regions[i].valid = false;
    }

    for (int i = 0; i < regions.size(); ++i)
    {
      if (regions[i].valid)
      {
        IplImage * sub = getSubImage(img, regions[i].rect);
        regions[i].text = recong(sub);
        cvReleaseImage(&sub);

        if (regions[i].text == "1" && regions[i].rectH <= 25 && regions[i].rectW <= 25)
        regions[i].valid = false;
      }
    }

    // merge hor text
    sort(regions.begin(), regions.end(), comRegionByX);
    for (int i = 0; i < regions.size(); ++i)
      for (int j = i + 1; j < regions.size(); ++j)
      {
        if (regions[i].valid && regions[j].valid)
        {
          int C = min(regions[i].rectW, regions[j].rectW) * 2;
          int ydiff = abs(regions[i].minH -regions[j].minH);
          int xdiff = abs(regions[j].minW - regions[i].maxW);
          xdiff = min(xdiff, abs(regions[i].minW - regions[j].maxW));
          int x1diff = abs(regions[i].minW - regions[j].minW);
          if (x1diff < 5) continue;

          double area1 = regions[i].rectH;
          double area2 = regions[j].rectH;
          if (area1 > area2 * 3 || area2 > area1 * 3) continue;

          if (xdiff < C && ydiff < C)
            regions[i].mergeRegion(regions[j]);
        }
      }

    // merge vertical text
    sort(regions.begin(), regions.end(), comRegionByY);
    for (int i = 0; i < regions.size(); ++i)
      for (int j = i + 1; j < regions.size(); ++j)
      {
        if (regions[i].valid && regions[j].valid)
        {
          int C = min(regions[i].rectW, regions[j].rectW) * 2 / 3;
         
          int ydiff = abs(regions[j].minH - regions[i].maxH);
          ydiff = min(ydiff, abs(regions[i].minH - regions[j].maxH));

          int xdiff = abs(regions[i].minW - regions[j].minW);
          if (xdiff > 5) continue;

          double area1 = regions[i].rectW;
          double area2 = regions[j].rectW;
          if (area1 > area2 * 3 || area2 > area1 * 3) continue;

          if (xdiff < C && ydiff < C) //&& regions[i].text.length() + regions[j].text.length() <= 3)
            regions[i].mergeRegion(regions[j]);
        }
      }


    // filter
    for (int i = 0; i < regions.size(); ++i)
    {
      if (regions[i].text.length() > 4) 
        regions[i].valid = false;
      if (regions[i].points.size() <= 10) 
        regions[i].valid = false;
      if (regions[i].rectH < 2 || regions[i].rectW < 2) 
        regions[i].valid = false;
    }

    vector<string> ret;
    for (int i = 0; i < regions.size(); ++i)
    {
      if (regions[i].valid) 
      {
        ret.push_back(regions[i].toString(H, W, true));
        cerr << regions[i] << endl;
      }
    }

    cvReleaseImage(&img);
    cvReleaseImage(&binary);
    return ret;
  }
};

////////////////////////////////////////////////////
// LOCAL
////////////////////////////////////////////////////

#ifdef LOCAL
#include "Util.h"
#include "CvImageOperationExt.h"

// Extract Feature from training data
void extractTrainingFeature()
{
  ofstream fout("E:\\Coder\\marathon\\patent\\pattern.txt");

  vector<vector<pair<int, double> > > input;
  vector<double> output;
  set<string> app;

  DirFinder finder("E:\\Coder\\marathon\\patent\\training\\chars\\");
  fout << "string pattern = string()";
  int cnt = 0;
  while (finder.hasNext())
  {
    string dir = finder.next();
    FileFinder find(dir);
    string ans = Util::getFileName(dir);

    fout << "+string(";
    while (find.hasNext())
    {
      string file = find.next();
      if (file.find("jpg") == string::npos) continue;
      cnt++;

      IplImage * img = cvLoadImage(file);
      IplImage * gray = cvCreateImage(cvGetSize(img), 8, 1);
      cvCvtColor(img, gray, CV_BGR2GRAY);
      vector<pair<int, double> > nowfea = extractFeature(gray);
      cvReleaseImage(&gray);
      cvReleaseImage(&img);

      string ss(FEADIM, '0');
      for (int i = 0; i < nowfea.size(); ++i)
        ss[nowfea[i].first - 1] = '0' + nowfea[i].second;

      if (app.find(ss) != app.end()) continue;
      app.insert(ss);
      fout << "\"";
      fout << ans[0];
      fout << ss;
      fout << "\"" << endl;

      output.push_back(ans[0]);
      input.push_back(nowfea);
    }
    fout << ")";
  }
  fout << ";" << endl;

  fout.close();
  debug1(cnt);

  SVMPredictor sp(input, output, "E:\\Coder\\marathon\\patent\\model.txt");
  sp.outputModelToCode("E:\\Coder\\marathon\\patent\\model.txt", "E:\\Coder\\marathon\\patent\\modelcode.txt");
}

// Extract Character from training data 
map<char, int> charcnt;
void extractCharacter(IplImage * img, string text)
{
  IplImage * gray = cvCreateImage(cvGetSize(img), 8, 1);
  cvCvtColor(img, gray, CV_BGR2GRAY);
  int h = img->height; 
  int w = img->width;
  getBinaryCharImage(gray, gray);
  
  vector<vector<int> > coms(h, vector<int>(w));
  vector<CvRegion> regions;
  floodFill(gray, coms, regions, 8, 10, 10000);

  vector<CvRegion> validrs;
  if (regions.size() == text.length()) validrs = regions;
  else
  {
    for (int i = 0; i < regions.size(); ++i)
    {
      if (regions[i].rectH > h * 8 / 10) continue;
      if (regions[i].rectW > w * 8 / 10) continue;
      if (regions[i].area < h * w / 25) continue;
      validrs.push_back(regions[i]);
    }
  }

  if (validrs.size() == text.length())
  {
    sort(validrs.begin(), validrs.end(), comRegionByX);
    int WRange = validrs.back().maxW - validrs.front().minW;

    sort(validrs.begin(), validrs.end(), comRegionByY);
    int HRange = validrs.back().maxH - validrs.front().minH;

    if (WRange > HRange)
      sort(validrs.begin(), validrs.end(), comRegionByX);
    else
    {
      sort(validrs.begin(), validrs.end(), comRegionByY);
      reverse(validrs.begin(), validrs.end());
    }

    for (int i = 0; i < validrs.size(); ++i)
    {
      char c = text[i];
      string dir = "C:\\training\\chars\\" + string(1, c);
      Util::mkdir(dir);
      IplImage * subImage = CvExt::getSubImage(img, validrs[i].rect);
      charcnt[c]++;
      string file = dir + "\\" + string(1, c) + "_" + toString(charcnt[c]) + ".jpg";
      cvSaveImage(file.data(), subImage);
      cvReleaseImage(&subImage);
    }
  }

  cvReleaseImage(&gray);
}

// Extract SubImage and corresponding label answer from training data set 
void extractAnswerFromTraining()
{
  string savePath = "C:\\training\\dataT\\";

  FileFinder finder("C:\\training\\images\\");
  while (finder.hasNext())
  {
    string file = finder.next();
    string filename = Util::getFileTrueName(file);
    IplImage * img = cvLoadImage(file.data());

    string figFile = "C:\\training\\figures\\" + filename + ".ans";
    string partFile = "C:\\training\\parts\\" + filename + ".ans";
    ifstream fin1(partFile);

    cout << file << endl;

    int cnt;
    fin1 >> cnt;
    for (int i = 0; i < cnt; ++i)
    {
      int k;
      fin1 >> k;
      int minx = 9999, miny = 9999;
      int maxx = 0, maxy = 0;
      while (k--)
      {
        int x, y;
        fin1 >> x >> y;
        minx = min(minx, x);
        miny = min(miny, y);
        maxx = max(maxx, x);
        maxy = max(maxy, y);
      }

      string text;
      fin1 >> text;

      CvRect rect = cvRect(minx, miny, maxx - minx + 1, maxy - miny + 1);
      IplImage * sub = CvExt::getSubImage(img, rect);
      string savefile = savePath + filename + "!" + text + ".jpg";
      cvSaveImage(savefile.data(), sub);

      extractCharacter(sub, text);

      cvReleaseImage(&sub);
    }
    cvReleaseImage(&img);
  }
}


int main()
{
  //extractAnswerFromTraining();
  //extractTrainingFeature();
  //return 0;


  PatentLabeling pt;

  int H, W;
  cin >> H >> W;

  int iLen;
  cin >> iLen;

  vector<int> image(iLen);
  for (int i = 0; i < iLen; ++i)
    scanf("%d", &image[i]);

  int tLen;
  cin >> tLen;
  vector<string> text(tLen);

  string t;
  getline(cin, t);
  for (int i = 0; i < tLen; ++i)
    getline(cin, text[i]);

  int callType;
  cin >> callType;

  vector<string> ret;
  if (callType == 1)
    ret = pt.getFigures(H, W, image, text);
  else
    ret = pt.getPartLabels(H, W, image, text);

  cout << ret.size() << endl;

  for (int i = 0; i < ret.size(); ++i)
    cout << ret[i] << endl;
}
#endif