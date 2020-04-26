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
