#ifndef UNION_SET
#define UNION_SET

#include <iostream>
using namespace std;

//不相交集合类
//按规模合并
//路径压缩
class Relation
{
private:
	//存贮每个元素的等价关系，state[i]>0表示他的父亲节点为state[i] state[i]<0表示他是一个根节点 其绝对值为树的规模
	int *state;
	//等价类的规模
	int size;
	float *adddata;

public:
	//构造函数
	Relation(int x)
	{
		size=x;
		state=new int[x];
		adddata=new float[x];
		clear();
	}

	Relation()
	{
		size=4000000;
		state=new int[size];
		adddata=new float[size];
		clear();
	}
	
	//析构函数
	~Relation()
	{
		delete [] state;
		delete [] adddata;
	}

	//等价关系清零
	void clear()
	{
		for (int i=0;i<size;++i) 
		{
			state[i]=-1;
			adddata[i]=0.0;
		}
	}

	//返回一个元素所在的等价类的代表元的数组下标
	int Find(int position)
	{
		if (state[position]<0)
				return position;
		else 
			//路径压缩
			return state[position]=Find(state[position]);
	}

	// 返回一个等价类的代表元代表的规模
	int GetNumber(int position)
	{ return -state[position]; }

	float GetData(int position)
	{ return adddata[position]; } 

	//将一个新的等价关系添加到类中
	void Union(int p1,int p2,float data=0.0)
	{
		//按规模合并，先获得两个元素的代表元下标，然后获得规模
		int x1=Find(p1);
		int x2=Find(p2);
		int s1=abs(state[x1]);
		int s2=abs(state[x2]);
		
		if (s1<=s2)
		{
			state[x1]=x2;
			state[x2]-=s1;
			adddata[x2]=data;
		}
		else
		{
			state[x2]=x1;
			state[x1]-=s2;
			adddata[x1]=data;
		}
		
	}

	//调试用
	void print()
	{
		for (int i=0;i<size;++i) cout<<i<<" "<<state[i]<<"# ";
		cout<<endl;
	}
};

#endif


