#include<iostream>
using namespace std;
#include "wall.h"
#include "snake.h"
#include "food.h"
#include<ctime>
#include<conio.h>
#include<Windows.h>

/****** ����Ƶ�ָ��λ�� ********************************/
void gotoxy(HANDLE hOut, int x, int y)
{
	COORD pos;
	pos.X = x;             //������
	pos.Y = y;            //������
	SetConsoleCursorPosition(hOut, pos);
}
HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);//������ʾ���������

int main() 
{	
	//������������
	srand((unsigned int)time(NULL));
	//���������־
	int isdead = 1;
	//����ж��Ƿ��һ���ƶ�
	char pre = NULL;

	Wall mywall;
	
	mywall.initWall();
	mywall.drawWall();

	Food food(mywall);
	food.setFood();


	Snake snake(mywall,food );
	snake.initSnake();



	//�����û�������
	gotoxy(hOut, 20, 5);

	while (isdead)
	{
		char input = _getch();
		//һ��ʼֱ�Ӳ����ƶ����ж��ǲ��ǵ�һ��
		if (input == 'a'&&pre == NULL)
		{
			continue;
		}
		do
		{  
			if (input == snake.UP || input == snake.DOWN || input == snake.LEFT || input == snake.RIGHT)
			{	
				//�ж��ϴ��Ƿ��ͻ
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

				//�ƶ��ɹ������»���ͼ��
				if (snake.move(input) == true)
				{
					/*system("cls");
					mywall.drawWall();*/
					//cout << "              ��ϲ��÷֣�" << snake.getScore() << "��" << endl;
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
				input = pre;  //�������������ϴγɹ�����ֵ
				continue;
			}

			
		} while (!_kbhit());//��û�м��������ʱ�򷵻�0


	}


	system("pause");

	return EXIT_SUCCESS;
}
