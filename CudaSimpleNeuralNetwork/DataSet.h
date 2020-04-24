#pragma once

#include <stdint.h>
#include <memory>
#include <vector>
#include "CustomCudaArray.h"

class DataSet
{
private:
	uint32_t getInt32(char* inpArr);
public:
	DataSet();
	~DataSet();

	struct data
	{
		CustomCudaArray image;
		CustomCudaArray label;

		data() : image(CustomCudaArray(28, 28 * 60000)), label(CustomCudaArray(1 * 60000)) {}
	};

	std::vector<data> training_data;
	std::vector<data> testing_data;
};
