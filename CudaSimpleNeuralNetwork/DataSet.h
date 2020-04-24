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
	};

	std::vector<data> training_data;
	std::vector<data> testing_data;
};
