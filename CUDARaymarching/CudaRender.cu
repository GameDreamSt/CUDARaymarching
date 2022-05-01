
#include "CudaRender.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <CVector.cuh>
#include <CudaFunctions.cuh>

#include <iostream>
#include <vector>
#include <chrono>
#include <windows.h>

using namespace std;

#define THREAD_COUNT 256

SDL_Color* host_pixels = nullptr;
int SCREEN_ROW, SCREEN_COL;

__device__ float DistanceEstimator(CVector3 point)
{
	//float dPlane = point.y; // dPlane = yCamera - yPlane

	CVector3 mySphere = CVector3(0.f, 1.f, 6.f);
	float sphereRadius = 1.;

	float dSphere = (mySphere - point).Magnitude() - sphereRadius; // dSphere = (SpherePos - CameraPos).Magnitude() - SphereRadius

	return min(point.y, dSphere);
}

#define MAX_RAY_STEPS 100
#define MAX_DISTANCE 25.f
#define MIN_DISTANCE 0.005f

__device__ float RayMarch(CVector3 from, CVector3 direction)
{
	float totalDistance = 0.0f;

	for (int steps = 0; steps < MAX_RAY_STEPS; steps++)
	{
		CVector3 point = from + direction * totalDistance;

		float distance = DistanceEstimator(point);
		totalDistance += distance;

		if (distance < MIN_DISTANCE || distance > MAX_DISTANCE)
			break;
	}

	return totalDistance;
}

__device__ CVector3 GetNormal(CVector3 point)
{
	float distance = DistanceEstimator(point);
	CVector2 e = CVector2(0.01f, 0.f);

	CVector3 n = CVector3(
		distance - DistanceEstimator(point - CVector3(e.x, e.y, e.y)),
		distance - DistanceEstimator(point - CVector3(e.y, e.x, e.y)),
		distance - DistanceEstimator(point - CVector3(e.y, e.y, e.x)));

	return n.Normalize();
}

struct S3
{
	float x, y, z;
};

__device__ float SoftShadowRaymarch(CVector3 from, CVector3 direction, float k)
{
	float res = 1.0;
	float ph = 1e20;

	for (float t = MIN_DISTANCE; t < MAX_DISTANCE; )
	{
		float h = DistanceEstimator(from + direction * t);

		if (h < 0.001f)
			return 0.f;

		float y = h * h / (2.f * ph);
		float d = sqrt(h * h - y * y);
		res = min(res, k * d / max(0.f, t - y));
		ph = h;
		t += h;
	}

	return res;
}

__device__ float GetLight(CVector3 point, S3 lightS3)
{
	CVector3 lightPos = CVector3(lightS3.x, lightS3.y, lightS3.z);
	CVector3 lightRay = (lightPos - point).Normalize();
	CVector3 normal = GetNormal(point);

	float diffuse = lightRay.Scalar(normal);

	if (diffuse > 1.f)
		diffuse = 1.f;
	else if (diffuse < 0.f)
		return 0.f;

	float shadow = SoftShadowRaymarch(point + normal * 0.02f, lightRay, 64.f);

	if (shadow < 0.1f)
		return diffuse * 0.1f;

	return diffuse * shadow;
}
__global__ void Shader(SDL_Color*pixels, int screenX, int screenY, S3 camS3, float camRotY, S3 lightS3)
{
	int index = threadIdx.x + blockIdx.x * THREAD_COUNT;

	if (index > screenX * screenY) // offscreen
		return;

	int screenCoordY = index / screenX;
	int screenCoordX = index - screenX * screenCoordY;

	CVector2 screenCoord = CVector2(screenCoordX, screenCoordY);

	CVector2 iResolution = CVector2(screenX, screenY);

	CVector2 uv = (screenCoord - iResolution * 0.5f) / iResolution.y;

	CVector3 rd = CVector3(uv.x, -uv.y, 1.0f).Normalize();

	rd = CVector3::RotateByY(rd, camRotY);

	CVector3 cam = CVector3(camS3.x, camS3.y, camS3.z);
	float dist = RayMarch(cam, rd);

	CVector3 point = cam + rd * dist;

	pixels[index].r = pixels[index].g = pixels[index].b = (unsigned char)(GetLight(point, lightS3) * 255.);
	pixels[index].a = 255;
}

void ApplyShader(Vector3 camPos, float camRotY, Vector3 lightPos)
{
	int totalPixels = SCREEN_ROW * SCREEN_COL;

	SDL_Color* dev_pixels = 0;
	CudaMalloc(dev_pixels, sizeof(SDL_Color) * totalPixels);
	
	int blockCount = (int)ceil((double)totalPixels / (double)THREAD_COUNT);

	S3 camS3{};
	camS3.x = camPos.x;
	camS3.y = camPos.y;
	camS3.z = camPos.z;

	S3 lightS3{};
	lightS3.x = lightPos.x;
	lightS3.y = lightPos.y;
	lightS3.z = lightPos.z;

	Shader << <blockCount, THREAD_COUNT >> > (dev_pixels, SCREEN_COL, SCREEN_ROW, camS3, camRotY, lightS3);

	CudaDeviceSynchronize();

	CudaCopyFromGPU(host_pixels, dev_pixels, sizeof(SDL_Color) * totalPixels);

	cudaFree(dev_pixels);
}

bool RenderImage(SDL_Color* pixels, int rows, int cols, Vector3 camPos, float camRotY, Vector3 lightPos)
{
	SCREEN_ROW = rows;
	SCREEN_COL = cols;

	if (!CudaSetDevice())
		return false;

	host_pixels = pixels;

	ApplyShader(camPos, camRotY, lightPos);

	return true;
}