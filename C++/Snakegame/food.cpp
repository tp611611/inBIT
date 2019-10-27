#include "food.h"
#include "windows.h"
/****** 光标移到指定位置 ********************************/
void gotoxy2(HANDLE hOut2, int x, int y)
{
	COORD pos;
	pos.X = x;             //横坐标
	pos.Y = y;            //纵坐标
	SetConsoleCursorPosition(hOut2, pos);
}
HANDLE hOut2 = GetStdHandle(STD_OUTPUT_HANDLE);//定义显示器句柄变量
Food::Food(Wall &tempWall):wall(tempWall)
{

}

void Food::setFood()
{	
	while (true)
	{
		Food_x = rand() % (Wall::ROW - 2) + 1;
		Food_y = rand() % (Wall::COL - 2) + 1;
		//如果生成的位置是空格的位置
		if (wall.getWall(Food_x, Food_y) == ' ')
		{
			wall.setWall(Food_x, Food_y, '0');
			gotoxy2(hOut2, Food_y * 2, Food_x);
			cout << '0';
			break;
		}
	}


}
