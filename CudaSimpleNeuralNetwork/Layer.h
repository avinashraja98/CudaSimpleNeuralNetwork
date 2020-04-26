#pragma once

#include <string>
#include "CustomCudaArray.h"
#include "Activation.h"

class Layer
{
private:
	std::string name;
	Activation* activation;
	size_t numUnits;
	size_t numInputs;

	CustomCudaArray W;	// Weights
	CustomCudaArray b;	// Biases

	CustomCudaArray Z;	// W.A + b
	CustomCudaArray A;	// Input
	CustomCudaArray dA;

	const float random_init_threshold = 0.01f;

	void fillWithZeroes(CustomCudaArray& inp);
	void fillWithRandom(CustomCudaArray& inp);
public:	
	Layer(std::string name, size_t numInputs, size_t numUnits, ActivationTypes actType = ActivationTypes::ReLU);
	~Layer();
		
	CustomCudaArray& forward(CustomCudaArray& A);
	CustomCudaArray& back(CustomCudaArray& dZ, float learningRate = 0.01);

	std::string getName() { return this->name; };
};

