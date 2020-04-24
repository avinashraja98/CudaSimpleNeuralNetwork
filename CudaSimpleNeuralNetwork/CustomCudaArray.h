#pragma once

#include <memory>

class CustomCudaArray
{
public:
	CustomCudaArray(size_t sizeX = 1, size_t sizeY = 1);
	~CustomCudaArray();

	std::shared_ptr<float> dataPtr;

	float& operator[](const int index);
	const float& operator[](const int index) const;
};

