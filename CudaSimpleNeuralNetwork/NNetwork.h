#pragma once

#include <vector>
#include "Layer.h"

class NNetwork
{
private:
	std::vector<Layer*> layers;
	
	CustomCudaArray Y;
	CustomCudaArray dY;

	float learningRate;
public:
	NNetwork(float learningRate = 0.01);
	~NNetwork();

	void addLayer(Layer* layer);
};

