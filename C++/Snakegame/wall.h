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
		ROW = 26,//行数
		COL = 26  //列数
	};
	//初始化墙壁
	void initWall();
	//画出来墙壁
	void drawWall();
	//根据索引值，返回二维数组中的内容
	char getWall(int x, int y);
	//根据索引值，设置二维数组中的内容
	void setWall(int x, int y, char c);
private:
	char gameArray[ROW][COL];
};
#endif // !WALL_HEAD

