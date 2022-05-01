#pragma once

#include <Vector.h>
#include <SDL.h>

bool RenderImage(SDL_Color* pixels, int rows, int cols, Vector3 camPos, float camRotY, Vector3 lightPos);