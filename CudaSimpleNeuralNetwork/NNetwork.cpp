#include "NNetwork.h"



NNetwork::NNetwork(float learningRate)
{
	this->learningRate = learningRate;
}


NNetwork::~NNetwork()
{
}

void NNetwork::addLayer(Layer * layer)
{
	this->layers.push_back(layer);
}

CustomCudaArray NNetwork::forward(CustomCudaArray X)
{
	CustomCudaArray Z = X;

	for (auto layer : layers) {
		Z = layer->forward(Z);
	}

	Y = Z;
	return Y;
}
