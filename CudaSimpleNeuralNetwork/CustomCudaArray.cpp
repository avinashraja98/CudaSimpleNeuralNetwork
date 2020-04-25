#include "CustomCudaArray.h"
#include "cuda_runtime.h"

CustomCudaArray::CustomCudaArray(size_t sizeX, size_t sizeY)
{
	//float *memory = nullptr;
	//cudaMallocManaged(&memory, sizeX * sizeY * sizeof(float));
	//dataPtr = std::shared_ptr<float>(memory, [&](float* ptr) {cudaDeviceSynchronize(); cudaFree(ptr); });
	dataPtr = std::shared_ptr<float[]>(new float[sizeX * sizeY], [&](float* ptr) { delete[] ptr; });
}


CustomCudaArray::~CustomCudaArray()
{
}

float& CustomCudaArray::operator[](const int index) {
	return dataPtr.get()[index];
}

const float& CustomCudaArray::operator[](const int index) const {
	return dataPtr.get()[index];
}
