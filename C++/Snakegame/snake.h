#pragma once
#include<iostream>
#include"wall.h"
#include"food.h"
using namespace std;
//蛇类
class Snake
{
public:	//构造函数
	enum{UP = 'w',LEFT = 'a',DOWN = 's',RIGHT = 'd' };


	Snake(Wall &Walltemp,Food &Foodtemp);
	//节点
	struct Point
	{
		//数据区
		int x;
		int y;
		//指针域
		Point *next;
	};

	//初始化节点
	void initSnake();


	//添加节点
	void addPoint(int x,int y);

	//删除节点
	void destroyPoint();

	//删除节点
	void delPoint();

	//蛇移动操作，bool值表示移动的成功与否
	bool move(char KEY);
	//设置难度
	//获得睡眠时间
	int getSleepTime();
	//获得蛇的长度
	int getSnakeLength();
	//获取分数
	int getScore();

	//头节点
	Wall & wall;
	Food & food;
	Point *pHead;
	bool istail;//尾巴循环
};