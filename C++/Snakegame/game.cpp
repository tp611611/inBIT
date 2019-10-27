#include<iostream>
using namespace std;
#include "wall.h"
#include "snake.h"
#include "food.h"
#include<ctime>
#include<conio.h>
#include<Windows.h>

/****** 光标移到指定位置 ********************************/
void gotoxy(HANDLE hOut, int x, int y)
{
	COORD pos;
	pos.X = x;             //横坐标
	pos.Y = y;            //纵坐标
	SetConsoleCursorPosition(hOut, pos);
}
HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);//定义显示器句柄变量

int main() 
{	
	//添加随机数种子
	srand((unsigned int)time(NULL));
	//添加死亡标志
	int isdead = 1;
	//添加判断是否第一次移动
	char pre = NULL;

	Wall mywall;
	
	mywall.initWall();
	mywall.drawWall();

	Food food(mywall);
	food.setFood();


	Snake snake(mywall,food );
	snake.initSnake();



	//接受用户的输入
	gotoxy(hOut, 20, 5);

	while (isdead)
	{
		char input = _getch();
		//一开始直接不让移动，判断是不是第一次
		if (input == 'a'&&pre == NULL)
		{
			continue;
		}
		do
		{  
			if (input == snake.UP || input == snake.DOWN || input == snake.LEFT || input == snake.RIGHT)
			{	
				//判断上次是否冲突
				if ((input == snake.LEFT && pre == snake.LEFT) ||
					(input == snake.RIGHT && pre == snake.LEFT) ||
					(input == snake.UP && pre == snake.DOWN) ||
					(input == snake.DOWN && pre == snake.UP))
				{
					input = pre;
				}
				else {
					pre = input;
				}

				//移动成功，重新绘制图像
				if (snake.move(input) == true)
				{
					/*system("cls");
					mywall.drawWall();*/
					//cout << "              恭喜你得分：" << snake.getScore() << "分" << endl;
					gotoxy(hOut, (snake.pHead->y) * 2,snake.pHead->x);
					Sleep(snake.getSleepTime());

				}
				else
				{
					isdead = 0;
					break;
				}

			}
			else
			{
				input = pre;  //将本次输入变成上次成功输入值
				continue;
			}

			
		} while (!_kbhit());//当没有键盘输入的时候返回0


	}


	system("pause");

	return EXIT_SUCCESS;
}
