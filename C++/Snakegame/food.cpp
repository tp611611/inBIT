#include "food.h"
#include "windows.h"
/****** ����Ƶ�ָ��λ�� ********************************/
void gotoxy2(HANDLE hOut2, int x, int y)
{
	COORD pos;
	pos.X = x;             //������
	pos.Y = y;            //������
	SetConsoleCursorPosition(hOut2, pos);
}
HANDLE hOut2 = GetStdHandle(STD_OUTPUT_HANDLE);//������ʾ���������
Food::Food(Wall &tempWall):wall(tempWall)
{

}

void Food::setFood()
{	
	while (true)
	{
		Food_x = rand() % (Wall::ROW - 2) + 1;
		Food_y = rand() % (Wall::COL - 2) + 1;
		//������ɵ�λ���ǿո��λ��
		if (wall.getWall(Food_x, Food_y) == ' ')
		{
			wall.setWall(Food_x, Food_y, '0');
			gotoxy2(hOut2, Food_y * 2, Food_x);
			cout << '0';
			break;
		}
	}


}
