
#include <SDL.h>
#include <SDLWin.h>
#include <FPS.h>

#include <Vector.h>

#include <CudaRender.cuh>

#include <Windows.h>
#include <chrono>

using namespace std;

void MainLoop();

int main(int argc, char** argv)
{
	MainLoop();

	return 0;
}

float deltatime = 1.f;
const float maxSpeed = 5.f;
float acceleration = 50.f;
Vector3 speedVec;

const float maxRotSpeed = PI / 1.25f;
float rotAcceleration = PI * 25.f;
float rotVel = 0.f;
void CatchCamInput(Vector3& camPos, float &camRotY)
{
	bool pressed = false;
	if (GetAsyncKeyState('A'))
	{
		speedVec += Vector3::RotateByY(Vector3::left * acceleration * deltatime, camRotY);
		pressed = true;
	}
	else if (GetAsyncKeyState('D'))
	{
		speedVec += Vector3::RotateByY(Vector3::right * acceleration * deltatime, camRotY);
		pressed = true;
	}

	if (GetAsyncKeyState('W'))
	{
		speedVec += Vector3::RotateByY(Vector3::forwards * acceleration * deltatime, camRotY);
		pressed = true;
	}
	else if (GetAsyncKeyState('S'))
	{
		speedVec += Vector3::RotateByY(Vector3::backwards * acceleration * deltatime, camRotY);
		pressed = true;
	}

	if (pressed)
	{
		if (speedVec.Magnitude() > maxSpeed)
			speedVec = speedVec.Normalize() * maxSpeed;
	}
	else
		speedVec = speedVec - speedVec * acceleration * deltatime;

	camPos += speedVec * deltatime;

	pressed = false;

	if (GetAsyncKeyState('Q'))
	{
		rotVel += -rotAcceleration * deltatime;
		pressed = true;
	}
	else if (GetAsyncKeyState('E'))
	{
		rotVel += rotAcceleration * deltatime;
		pressed = true;
	}

	if (pressed)
	{
		if (rotVel > maxRotSpeed)
			rotVel = maxRotSpeed;
		else if (rotVel < -maxRotSpeed)
			rotVel = -maxRotSpeed;
	}
	else
	{
		if (abs(rotVel) < abs(rotVel * rotAcceleration * deltatime))
			rotVel = 0.f;
		else
			rotVel = rotVel - rotVel * rotAcceleration * deltatime;
	}

	camRotY += rotVel * deltatime;
}

float timeX;
void CatchLightInput(Vector3& lightPos)
{
	timeX += deltatime;
	lightPos.x = sin(timeX) * 10.f;
	lightPos.z = cos(timeX) * 10.f + 6.f;
}

void MainLoop()
{
	bool gameQuit = false;
	bool pressedPause = false;
	bool paused = false;

	Vector3 cameraPos = Vector3(0, 1, 0);
	float camRotY = 0.f;
	Vector3 lightPos = Vector3(0, 5, 6);

	SDLWIN* SDLWin = new SDLWIN();
	FPS* fps = new FPS();

	SDL_Color* pixels = new SDL_Color[GAME_BASE_RESOLUTION_X * GAME_BASE_RESOLUTION_Y];

	while (!gameQuit && !GetAsyncKeyState(VK_ESCAPE))
	{
		SDL_Event event;
		if (SDL_PollEvent(&event) && event.type == SDL_QUIT)
			break;

		if (GetAsyncKeyState('P'))
		{
			if (pressedPause != true)
				paused = !paused;

			pressedPause = true;
		}
		else
			pressedPause = false;

		if (!paused)
		{
			CatchCamInput(cameraPos, camRotY);
			CatchLightInput(lightPos);

			if (!paused)
			{
				auto t1 = std::chrono::steady_clock::now();
				RenderImage(pixels, GAME_BASE_RESOLUTION_Y, GAME_BASE_RESOLUTION_X, cameraPos, camRotY, lightPos);
				auto t2 = chrono::steady_clock::now();
				long long micro = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();

				if (micro == 0)
					micro = 1;

				SDL_RenderClear(SDLWin->renderer);
				SDLWin->SDLDraw(pixels);
				fps->DrawFPS((int)(1000000.f / micro));
				SDL_RenderPresent(SDLWin->renderer);

				t2 = chrono::steady_clock::now();
				deltatime = (float)(chrono::duration_cast<chrono::microseconds>(t2 - t1).count() / 1000000.f);
			}
		}
	}

	delete fps;
	delete[] pixels;
	SDLWin->SDLQuit();
	delete SDLWin;
}