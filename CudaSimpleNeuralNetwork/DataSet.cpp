#include "DataSet.h"
#include <fstream>
#include <iostream>

uint32_t DataSet::getInt32(char * inpArr)
{
	uint32_t result = (unsigned char)inpArr[0] << 32 | (unsigned char)inpArr[1] << 16 | (unsigned char)inpArr[2] << 8 | (unsigned char)inpArr[3];
	return result;
}

DataSet::DataSet()
{
	std::ifstream trainImgFile("data/train-images.idx3-ubyte", std::ios::binary);
	std::ifstream trainLabelFile("data/train-labels.idx1-ubyte", std::ios::binary);

	if (trainImgFile.is_open() && trainLabelFile.is_open())
	{
		char buf[4];
		uint32_t count = 0;
		uint32_t row_num = 0;
		uint32_t col_num = 0;

		// Check Magic Number
		trainImgFile.read((char*)&buf, sizeof(buf));
		if (getInt32(buf) != 0x00000803)
		{
			// error not mnist image file
			trainImgFile.close();
			trainLabelFile.close();
			return;
		}
		trainLabelFile.read((char*)&buf, sizeof(buf));
		if (getInt32(buf) != 0x00000801)
		{
			// error not mnist label file
			trainImgFile.close();
			trainLabelFile.close();
			return;
		}

		// Check count
		trainImgFile.read((char*)&buf, sizeof(buf));
		count = getInt32(buf);
		trainLabelFile.read((char*)&buf, sizeof(buf));
		if (getInt32(buf) != count && count != 60000)
		{
			// error count mismatch
			trainImgFile.close();
			trainLabelFile.close();
			return;
		}

		// Check height and width from img file
		trainImgFile.read((char*)&buf, sizeof(buf));
		row_num = getInt32(buf);
		trainImgFile.read((char*)&buf, sizeof(buf));
		col_num = getInt32(buf);
		if (row_num != 28 && col_num != 28)
		{
			// error wrong img size
			trainImgFile.close();
			trainLabelFile.close();
			return;
		}

		// Get data and build object
		// cudaMallocManaged crashes if called multiple times. So now its called once.
		//data currentExample;
		for (int k = 0; k < count; k++)
		{
			char labelBuf;
			char imgBuf[28 * 28];
			
			data currentExample;
			trainImgFile.read((char*)&imgBuf, sizeof(imgBuf));
			for (int i = 0; i < 28 * 28; i++) {
				currentExample.image[i] = (float)((unsigned char)imgBuf[i]) / 255.0;
			}

			trainLabelFile.read((char*)&labelBuf, sizeof(labelBuf));			
			currentExample.label[0] = (float)((unsigned char)labelBuf);
			training_data.push_back(currentExample);
		}
		//training_data.push_back(currentExample);
		trainImgFile.close();
		trainLabelFile.close();
	}
}

DataSet::~DataSet()
{
}
