#pragma once
#ifndef WALL_HEAD
#define WALL_HEAD
#include <iostream>
using namespace std;
class Wall
{
public:
	Wall();
	~Wall();
	enum
	{
		ROW = 26,//����
		COL = 26  //����
	};
	//��ʼ��ǽ��
	void initWall();
	//������ǽ��
	void drawWall();
	//��������ֵ�����ض�ά�����е�����
	char getWall(int x, int y);
	//��������ֵ�����ö�ά�����е�����
	void setWall(int x, int y, char c);
private:
	char gameArray[ROW][COL];
};
#endif // !WALL_HEAD

