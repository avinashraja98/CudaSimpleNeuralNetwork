#pragma once

#include <stdint.h>

class DataSet
{
public:
	DataSet();
	~DataSet();

	struct data
	{
		uint8_t *image;
		uint8_t label;
	};

	data *training_data;
	data *testing_data;
};

