
#include <FPS.h>

#include <SDLWin.h>

#include <string>

FPS::FPS()
{
	position = GAME_BASE_RESOLUTION / 100.;
	scale = GAME_BASE_RESOLUTION / 200.;

	LoadNumbers();
}

void FPS::DrawFPS(int fps)
{
	FPS_Counter++;

	if (FPS_Counter >= 16)
	{
		FPS_Show = FPS_Sum / FPS_Counter;
		FPS_Counter = 0;
		FPS_Sum = 0;
	}
	else
		FPS_Sum += fps;

	string str = to_string(FPS_Show);
	for (size_t i = 0; i < str.size(); i++)
	{
		SDL_Rect texture = GetLetterTexture(str[i]);
		SDL_Rect dest = texture;

		dest.w *= (int)scale.x;
		dest.h = dest.w;

		dest.x = (int)position.x + dest.w * (int)i;
		dest.y = (int)position.y;

		SDL_RenderCopy(SDLWIN::singleton->renderer, numberTexture, &texture, &dest);
	}
}

SDL_Rect PosSizeToRect(Vector2 pos, Vector2 size)
{
	SDL_Rect rect{};
	rect.x = (int)pos.x;
	rect.y = (int)pos.y;
	rect.w = (int)size.x;
	rect.h = (int)size.y;
	return rect;
}

void FPS::LoadNumbers()
{
	SDL_Surface* Surface = SDL_LoadBMP("numbers.bmp");
	numberTexture = SDL_CreateTextureFromSurface(SDLWIN::singleton->renderer, Surface);
	SDL_FreeSurface(Surface);

	string charSet = "0123456789";

	Vector2 letterPositions[11];

	for (int i = 0; i < 11; i++) // 0123456789
		letterPositions[i] = Vector2(i * 8.f, 0.f);

	Vector2 letterSize = Vector2(8.f, 8.f);

	for (size_t i = 0; i < 11; i++)
	{
		numberTextures.push_back(PosSizeToRect(letterPositions[i], letterSize));
		numberHashmap[charSet[i]] = i;
	}
}

SDL_Rect FPS::GetLetterTexture(char letter)
{
	if (!numberHashmap.count(letter))
		return SDL_Rect{};

	return numberTextures[numberHashmap[letter]];
}