#include "wall.h"
Wall::Wall()
{
}
Wall::~Wall()
{
}
void  Wall::initWall()
{
	for (int i = 0; i < ROW; i++)
	{
		for(int j = 0; j < COL; j++)
		{
			if (i == 0 || j == 0 || i == ROW - 1 || j == COL - 1)
			{
				gameArray[i][j] = '@';
			}
			else
			{
				gameArray[i][j] = ' ';
			}
		}
	}
}

void  Wall::drawWall()
{
	for (int i = 0; i < ROW; i++)
	{
		for (int j = 0; j < COL; j++)
		{
			cout << gameArray[i][j]<<' ';
		}
		if (i == 5)
		{
			cout << "create by cache_tp";
		}
		if (i == 6)
		{
			cout << "a: left";
		}
		if (i == 7)
		{
			cout << "d: right";

		}
		if (i == 8)
		{
			cout << "w: up";

		}
		if (i == 9)
		{
			cout << "s: down";

		}
		if (i == 10)
		{
			cout << "as its length is becoming longer  ";

		}
		if (i == 11)
		{
			cout << "its speed is becoming faster and faster";
		}
		cout << endl;
	}
}

//根据索引值，设置二维数组中的内容

void Wall::setWall(int x, int y, char c)
{
	gameArray[x][y] = c;
}

char Wall::getWall(int x, int y)
{
	return gameArray[x][y];

}



