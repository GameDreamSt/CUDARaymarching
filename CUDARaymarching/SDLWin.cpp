
#include <SDLWin.h>

SDLWIN* SDLWIN::singleton = nullptr;

SDLWIN::SDLWIN()
{
	window = nullptr;
	renderer = nullptr;
	renderTexture = nullptr;

	singleton = this;

	SDLInitialize();
}

void SDLWIN::SDLInitialize()
{
	SDL_Init(SDL_INIT_EVERYTHING);
	SDL_CreateWindowAndRenderer(GAME_BASE_RESOLUTION_X, GAME_BASE_RESOLUTION_Y, 0, &window, &renderer);

	renderTexture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB888, SDL_TEXTUREACCESS_STREAMING, GAME_BASE_RESOLUTION_X, GAME_BASE_RESOLUTION_Y);
}

void SDLWIN::SDLQuit()
{
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
}

void SDLWIN::SDLDraw(SDL_Color* pixels)
{
	SDL_UpdateTexture(renderTexture, NULL, pixels, GAME_BASE_RESOLUTION_X * 4);
	SDL_RenderCopy(renderer, renderTexture, NULL, NULL);
}