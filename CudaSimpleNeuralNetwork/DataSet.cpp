#include "DataSet.h"
#include <fstream>
#include <iostream>

uint32_t getInt32(char * inpArr)
{
	uint32_t result = (unsigned char)inpArr[0] << 32 | (unsigned char)inpArr[1] << 16 | (unsigned char)inpArr[2] << 8 | (unsigned char)inpArr[3];
	return result;
}

std::vector<DataSet::data> DataSet::extractData(char * imgFile, char * lblFile)
{
	std::ifstream imgFilePtr(imgFile, std::ios::binary);
	std::ifstream labelFilePtr(lblFile, std::ios::binary);

	std::vector<data> out;

	if (imgFilePtr.is_open() && labelFilePtr.is_open())
	{
		char buf[4];
		uint32_t count = 0;
		uint32_t row_num = 0;
		uint32_t col_num = 0;

		// Check Magic Number
		imgFilePtr.read((char*)&buf, sizeof(buf));
		if (getInt32(buf) != 0x00000803)
		{
			// error not mnist image file
			imgFilePtr.close();
			labelFilePtr.close();
			return out;
		}
		labelFilePtr.read((char*)&buf, sizeof(buf));
		if (getInt32(buf) != 0x00000801)
		{
			// error not mnist label file
			imgFilePtr.close();
			labelFilePtr.close();
			return out;
		}

		// Check count
		imgFilePtr.read((char*)&buf, sizeof(buf));
		count = getInt32(buf);
		labelFilePtr.read((char*)&buf, sizeof(buf));
		if (getInt32(buf) != count)
		{
			// error count mismatch
			imgFilePtr.close();
			labelFilePtr.close();
			return out;
		}

		// Check height and width from img file
		imgFilePtr.read((char*)&buf, sizeof(buf));
		row_num = getInt32(buf);
		imgFilePtr.read((char*)&buf, sizeof(buf));
		col_num = getInt32(buf);
		if (row_num != 28 && col_num != 28)
		{
			// error wrong img size
			imgFilePtr.close();
			labelFilePtr.close();
			return out;
		}

		// Get data and build object
		for (int k = 0; k < count; k++)
		{
			char labelBuf;
			char imgBuf[28 * 28];

			data currentExample;
			imgFilePtr.read((char*)&imgBuf, sizeof(imgBuf));
			for (int i = 0; i < 28 * 28; i++) {
				currentExample.image[i] = (float)((unsigned char)imgBuf[i]) / 255.0;
			}

			labelFilePtr.read((char*)&labelBuf, sizeof(labelBuf));
			float currentLabel = (float)((unsigned char)labelBuf);
			for (int j = 0; j < 10; j++)
			{
				currentExample.label[j] = (int)currentLabel == j ? 1 : 0;
			}
			out.push_back(currentExample);
		}
		imgFilePtr.close();
		labelFilePtr.close();
	}
	return out;
}

DataSet::DataSet()
{
	training_data = extractData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
	testing_data = extractData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
}

DataSet::~DataSet()
{
}
