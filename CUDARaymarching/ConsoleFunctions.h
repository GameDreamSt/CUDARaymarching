#pragma once

#include <windows.h>

void SetPointerPos(int x, int y)
{
	HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
	COORD destCoord{};

	destCoord.X = x;
	destCoord.Y = y;
	SetConsoleCursorPosition(hStdout, destCoord);
}

void GetConsoleSize(int* row, int* col)
{
	CONSOLE_SCREEN_BUFFER_INFO csbi;

	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
	*col = csbi.srWindow.Right - csbi.srWindow.Left + 1;
	*row = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
}