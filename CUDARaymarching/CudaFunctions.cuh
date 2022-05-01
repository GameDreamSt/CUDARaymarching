#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

using namespace std;

bool CudaSetDevice()
{
	cudaError_t err = cudaSetDevice(0);

	if (err != cudaSuccess)
		cout << "cudaSetDevice failed!\nDo you have a CUDA-capable GPU installed?\n" << cudaGetErrorString(err) << '\n';

	return err == cudaSuccess;
}

bool CCudaMalloc(void** devPtr, size_t size)
{
	cudaError_t err = cudaMalloc(devPtr, size);

	if (err != cudaSuccess)
		cout << "cudaMalloc failed!\n" << cudaGetErrorString(err) << '\n';

	return err == cudaSuccess;
}
#define CudaMalloc(x,y) CCudaMalloc((void**)&x,y)

bool CudaDeviceSynchronize()
{
	cudaError_t err = cudaDeviceSynchronize();

	if (err != cudaSuccess)
		cout << "cudaDeviceSynchronize returned error code %d after launching addKernel!\n" << cudaGetErrorString(err) << '\n';

	return err == cudaSuccess;
}

bool CudaCopyFromGPU(void* host_ptr, void* gpu_ptr, size_t size)
{
	cudaError_t err = cudaMemcpy(host_ptr, gpu_ptr, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
		cout << "cudaMemcpy from device failed!\n" << cudaGetErrorString(err) << '\n';

	return err == cudaSuccess;
}

bool CudaCopyToGPU(void* host_ptr, void* gpu_ptr, size_t size)
{
	cudaError_t err = cudaMemcpy(gpu_ptr, host_ptr, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
		cout << "cudaMemcpy to device failed!\n" << cudaGetErrorString(err) << '\n';

	return err == cudaSuccess;
}