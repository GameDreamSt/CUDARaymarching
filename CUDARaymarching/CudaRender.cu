
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

enum HitObject
{
	None = 0,
	Ground = 1,
	Sphere = 2
};

__device__ float DistanceEstimatorHit(CVector3 point, HitObject& hitObj)
{
	CVector3 mySphere = CVector3(0.f, 1.f, 6.f);
	float sphereRadius = 1.;

	float dSphere = (mySphere - point).Magnitude() - sphereRadius;

	if (point.y < dSphere)
	{
		hitObj = HitObject::Ground;
		return point.y;
	}
	else
	{
		hitObj = HitObject::Sphere;
		return dSphere;
	}
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

__device__ float RayMarchHit(CVector3 from, CVector3 direction, HitObject& hitObj)
{
	float totalDistance = 0.0f;

	for (int steps = 0; steps < MAX_RAY_STEPS; steps++)
	{
		CVector3 point = from + direction * totalDistance;

		float distance = DistanceEstimatorHit(point, hitObj);
		totalDistance += distance;

		if (distance < MIN_DISTANCE)
			break;

		if (distance > MAX_DISTANCE)
		{
			hitObj = HitObject::None;
			break;
		}
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

	S3() {}

	__device__ S3(float X, float Y, float Z)
	{
		x = X;
		y = Y;
		z = Z;
	}
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

__device__ float SimpleLight(CVector3 point, S3 lightS3)
{
	CVector3 lightPos = CVector3(lightS3.x, lightS3.y, lightS3.z);
	CVector3 lightRay = (lightPos - point).Normalize();
	CVector3 normal = GetNormal(point);

	float diffuse = lightRay.Scalar(normal);

	if (diffuse > 1.f)
		return 1.f;
	else if (diffuse < 0.f)
		return 0.f;

	return diffuse;
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

#define TILESIZE 2

__device__ S3 GetGroundColor(CVector3& point)
{
	S3 col = S3(0,0,0);

	if (point.x < 0.0f)
		point.x -= TILESIZE;

	if (point.z < 0.0f)
		point.z -= TILESIZE;

	bool red;
	if ((int)(point.x / TILESIZE) % 2 == 0)
		red = (int)(point.z / TILESIZE) % 2 == 0;
	else
		red = (int)(point.z / TILESIZE) % 2 != 0;

	if (red)
	{
		col.z = 1.0f; // R
		col.y = 0.25f;
		col.x = 0.25f; // B
	}
	else
	{
		col.z = 0.25f; // R
		col.y = 0.25f;
		col.x = 1.0f; // B
	}

	return col;
}

__device__ void SetColorFromS3(SDL_Color& pixel, S3 s3)
{
	pixel.r = (unsigned char)(s3.x * 255.);
	pixel.g = (unsigned char)(s3.y * 255.);
	pixel.b = (unsigned char)(s3.z * 255.);
}

__device__ void SetColorFromS3WithLight(SDL_Color& pixel, S3 s3, float& light)
{
	s3.x *= light;
	s3.y *= light;
	s3.z *= light;

	SetColorFromS3(pixel, s3);
}

__device__ void SetGroundColor(SDL_Color& pixel, float& light, CVector3& point)
{
	S3 col = GetGroundColor(point);

	SetColorFromS3WithLight(pixel, col, light);
}

__device__ void SphereReflect(SDL_Color& pixel, S3& lightS3, CVector3& oldPoint, CVector3& oldRayDir, float& oldLight)
{
	if (oldLight < 0.1f)
		oldLight = 0.1f;

	CVector3 normal = GetNormal(oldPoint);
	CVector3 point = oldPoint + normal * 0.01f;

	CVector3 rayDir = CVector3::Reflect(oldRayDir, GetNormal(point));

	HitObject hitObj;
	float dist = RayMarchHit(point, rayDir, hitObj);

	if (hitObj == HitObject::None)
	{
		pixel.r = pixel.g = pixel.b = (unsigned char)(oldLight * 10.);
		return;
	}

	point = oldPoint + rayDir * dist;

	float light = SimpleLight(point, lightS3);
	light = pow(light, 0.45454545454f);

	S3 col = S3(0,0,0);

	switch (hitObj)
	{
	case HitObject::Ground:
		col = GetGroundColor(point);

		col.x *= light * oldLight;
		col.y *= light * oldLight;
		col.z *= light * oldLight;

		SetColorFromS3(pixel, col);
		break;

	default:
	case HitObject::None:
		pixel.r = pixel.g = pixel.b = (unsigned char)(oldLight * 255.);
		break;

	case HitObject::Sphere:
		//SphereReflect(pixel, lightS3, point);
		pixel.r = pixel.g = pixel.b = (unsigned char)(oldLight * 255.);
		break;
	}
}

__global__ void Shader(SDL_Color* pixels, int screenX, int screenY, S3 camS3, float camRotY, S3 lightS3)
{
	int index = threadIdx.x + blockIdx.x * THREAD_COUNT;

	if (index > screenX * screenY) // offscreen
		return;

	int screenCoordY = index / screenX;
	int screenCoordX = index - screenX * screenCoordY;

	CVector2 screenCoord = CVector2(screenCoordX, screenCoordY);

	CVector2 iResolution = CVector2(screenX, screenY);

	CVector2 uv = (screenCoord - iResolution * 0.5f) / iResolution.y;

	CVector3 rayDir = CVector3(uv.x, -uv.y, 1.0f).Normalize();

	rayDir = CVector3::RotateByY(rayDir, camRotY);

	CVector3 cam = CVector3(camS3.x, camS3.y, camS3.z);

	HitObject hitObj;
	float dist = RayMarchHit(cam, rayDir, hitObj);

	if (hitObj == HitObject::None)
	{
		pixels[index].r = pixels[index].g = pixels[index].b = 0;
		return;
	}

	CVector3 point = cam + rayDir * dist;

	float light = GetLight(point, lightS3);
	light = pow(light, 0.45454545454f);

	switch (hitObj)
	{
		case HitObject::Ground:
			SetGroundColor(pixels[index], light, point);
		break;

		default:
		case HitObject::None:
			pixels[index].r = pixels[index].g = pixels[index].b = 0;
			break;

		case HitObject::Sphere:
			SphereReflect(pixels[index], lightS3, point, rayDir, light);
			break;
	}
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