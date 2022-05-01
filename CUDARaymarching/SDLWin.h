#pragma once

#include <SDL.h>
#include <Vector.h>

#define GAME_BASE_RESOLUTION_X 960
#define GAME_BASE_RESOLUTION_Y 540
#define GAME_BASE_RESOLUTION Vector2(GAME_BASE_RESOLUTION_X, GAME_BASE_RESOLUTION_Y)

class SDLWIN
{
public:
	SDLWIN();
	~SDLWIN() {};

	static SDLWIN* singleton;

	SDL_Window* window;
	SDL_Renderer* renderer;
	SDL_Texture* renderTexture;

	void SDLInitialize();
	void SDLQuit();
	void SDLDraw(SDL_Color* pixels);
};