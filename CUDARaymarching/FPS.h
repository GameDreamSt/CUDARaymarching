#pragma once

#include <SDL.h>
#include <Vector.h>

#include <vector>
#include <unordered_map>

using namespace std;

class FPS
{
public:
	FPS();
	~FPS(){}

	void DrawFPS(int fps);
private:
	Vector2 position, scale;

	int FPS_Sum = 0;
	int FPS_Counter = 0;
	int FPS_Show = 0;

	SDL_Texture* numberTexture;
	vector<SDL_Rect> numberTextures;
	unordered_map<char, size_t> numberHashmap;

	SDL_Rect GetLetterTexture(char letter);
	void LoadNumbers();
};

