#pragma once
#include<iostream>
#include"wall.h"
using namespace std;

class Food
{
public:
	Food(Wall &tempWall);
	//…Ë÷√ ≥ŒÔ
	void setFood();

	int Food_x;
	int Food_y;

	Wall & wall;


};