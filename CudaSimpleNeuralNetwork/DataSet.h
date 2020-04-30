#pragma once

#include <stdint.h>
#include <memory>
#include <vector>
#include "CustomCudaArray.h"

class DataSet
{
public:
	DataSet();
	~DataSet();

	struct data
	{
		CustomCudaArray image;
		CustomCudaArray label;

		data() : image(CustomCudaArray(28 * 28, 1000)), label(CustomCudaArray(10, 1000)) {}
	};

	std::vector<data> training_data;
	std::vector<data> testing_data;
private:
	std::vector<data> extractData(char* imgFile, char* lblFile);
};
