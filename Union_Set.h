#ifndef UNION_SET
#define UNION_SET

#include <iostream>
using namespace std;

//���ཻ������
//����ģ�ϲ�
//·��ѹ��
class Relation
{
private:
	//����ÿ��Ԫ�صĵȼ۹�ϵ��state[i]>0��ʾ���ĸ��׽ڵ�Ϊstate[i] state[i]<0��ʾ����һ�����ڵ� �����ֵΪ���Ĺ�ģ
	int *state;
	//�ȼ���Ĺ�ģ
	int size;
	float *adddata;

public:
	//���캯��
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
	
	//��������
	~Relation()
	{
		delete [] state;
		delete [] adddata;
	}

	//�ȼ۹�ϵ����
	void clear()
	{
		for (int i=0;i<size;++i) 
		{
			state[i]=-1;
			adddata[i]=0.0;
		}
	}

	//����һ��Ԫ�����ڵĵȼ���Ĵ���Ԫ�������±�
	int Find(int position)
	{
		if (state[position]<0)
				return position;
		else 
			//·��ѹ��
			return state[position]=Find(state[position]);
	}

	// ����һ���ȼ���Ĵ���Ԫ����Ĺ�ģ
	int GetNumber(int position)
	{ return -state[position]; }

	float GetData(int position)
	{ return adddata[position]; } 

	//��һ���µĵȼ۹�ϵ��ӵ�����
	void Union(int p1,int p2,float data=0.0)
	{
		//����ģ�ϲ����Ȼ������Ԫ�صĴ���Ԫ�±꣬Ȼ���ù�ģ
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

	//������
	void print()
	{
		for (int i=0;i<size;++i) cout<<i<<" "<<state[i]<<"# ";
		cout<<endl;
	}
};

#endif


