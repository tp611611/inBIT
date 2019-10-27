#pragma once
#include<iostream>
#include"wall.h"
#include"food.h"
using namespace std;
//����
class Snake
{
public:	//���캯��
	enum{UP = 'w',LEFT = 'a',DOWN = 's',RIGHT = 'd' };


	Snake(Wall &Walltemp,Food &Foodtemp);
	//�ڵ�
	struct Point
	{
		//������
		int x;
		int y;
		//ָ����
		Point *next;
	};

	//��ʼ���ڵ�
	void initSnake();


	//��ӽڵ�
	void addPoint(int x,int y);

	//ɾ���ڵ�
	void destroyPoint();

	//ɾ���ڵ�
	void delPoint();

	//���ƶ�������boolֵ��ʾ�ƶ��ĳɹ����
	bool move(char KEY);
	//�����Ѷ�
	//���˯��ʱ��
	int getSleepTime();
	//����ߵĳ���
	int getSnakeLength();
	//��ȡ����
	int getScore();

	//ͷ�ڵ�
	Wall & wall;
	Food & food;
	Point *pHead;
	bool istail;//β��ѭ��
};