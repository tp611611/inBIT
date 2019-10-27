#include "snake.h"
#include"windows.h"
/****** ����Ƶ�ָ��λ�� ********************************/
void gotoxy1(HANDLE hOut1, int x, int y)
{
	COORD pos;
	pos.X = x;             //������
	pos.Y = y;            //������
	SetConsoleCursorPosition(hOut1, pos);
}
HANDLE hOut1 = GetStdHandle(STD_OUTPUT_HANDLE);//������ʾ���������

Snake::Snake(Wall & Walltemp, Food &Foodtemp):wall(Walltemp),food(Foodtemp)
{
	pHead = NULL;
	istail = false;
}

void Snake::initSnake()
{
	destroyPoint();
	addPoint(5, 8);
	addPoint(5, 9);
	addPoint(5, 10);
}

void Snake::addPoint(int x,int y)
{
	//��ӽڵ�y
	Point *newPoint = new Point();
	newPoint->x = x;
	newPoint->y = y;
	newPoint->next = NULL;
	if (pHead != NULL)
	{
		wall.setWall(pHead->x, pHead->y, '=');
		gotoxy1(hOut1, pHead->y * 2, pHead->x);
		cout << '=';
	}
	newPoint->next = pHead; //����ͷ��
	pHead = newPoint;
	wall.setWall(pHead->x, pHead->y, '&');
	gotoxy1(hOut1, pHead->y * 2, pHead->x);
	cout << '&';
}

//�������еĽڵ�
void Snake::destroyPoint()
{
	Point *cur = pHead;
	while (pHead != NULL)
	{
		cur = pHead->next;
		delete pHead;
		pHead = cur;
	}
}

void Snake::delPoint()
{
	//�����ڵ����ϲ���ɾ������
	if (pHead == NULL || pHead->next == NULL)
	{
		return;
	}
	Point * pcur = pHead->next;
	Point * pre = pHead;
	while (pcur->next!=NULL)
	{
		pre = pcur;
		pcur = pcur->next;
	}
	wall.setWall(pcur->x, pcur->y, ' ');
	gotoxy1(hOut1, pcur->y * 2, pcur->x);
	cout << ' ';

	delete pcur;
	pcur = NULL;
	pre->next = NULL;

}

bool Snake::move(char KEY)
{	
	int x = pHead->x;
	int y = pHead->y;
	switch (KEY)
	{
	case UP:
		x--;
		break;
	case DOWN:
		x++;
		break;
	case RIGHT:
		y++;
		break;
	case LEFT:
		y--;
		break;
	default:
		break;
	}

	//�����һ����������β�Ͳ�Ӧ������
	Point * pcur = pHead->next;
	Point * pre = pHead;
	while (pcur->next != NULL)
	{
		pre = pcur;
		pcur = pcur->next;
	}
	if (pcur->x == x&&pcur->y==y)
	{
		//��������β�͵�ѭ��
		istail = true;
	}
	else
	{
		if (wall.getWall(x, y) == '&' || wall.getWall(x, y) == '=' || wall.getWall(x, y) == '@')
		{
			addPoint(x, y);
		//	system("cls");
		//	wall.drawWall();
			gotoxy1(hOut1, Wall::COL * 2, Wall::ROW);
		//	cout << "              ��ϲ��÷֣�" << getScore() << "��" << endl;
			cout << "GAME OVER! BYE BYE" << endl;
			return false;
		}
		
	}
	//�ƶ��ɹ������֣�1�ǳԵ���ʳ�2��û�гԵ�ʳ��
	if (wall.getWall(x, y) == '0')
	{
		addPoint(x, y);
		//��������ʳ��
		food.setFood();
		gotoxy1(hOut1, Wall::COL * 2-20, Wall::ROW);
		cout <<  "��ϲ��÷֣�" << getScore() << "��" << endl;


	}
	else
	{
		addPoint(x, y);
		delPoint();
		if (istail) {
			wall.setWall(x, y, '&');
			gotoxy1(hOut1, y * 2, x);
			cout << '&';

		}
	}
	return true;
}

int Snake::getSleepTime()
{   
	int sleepTime = 0;
	int len = getSnakeLength();
	if (len < 5)
	{
		sleepTime = 300;
	}
	else if (4 < len && len < 8)
	{
		sleepTime = 200;
	}
	else if(len < 10&&len>7)
	{
		sleepTime = 100;
	}
	else if(len >9 && len < 13)
	{
		sleepTime = 50;
	}
	else {
		sleepTime = 25;
	}
	return sleepTime;
}

int Snake::getSnakeLength()
{   
	int length = 0;
	Point * pcur = pHead;
	while (pcur!=NULL)
	{
		length++;
		pcur = pcur->next;
	}
	return length;
}

int Snake::getScore()
{   
	int score = 0;
	int len1 = getSnakeLength();
	score = 100 * (len1-3);
	return score;
}
