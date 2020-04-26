#pragma once

#include <string>
#include "CustomCudaArray.h"
#include "Activation.h"

class Layer
{
private:
	std::string name;
	Activation activation;

	CustomCudaArray W;	// Weights
	CustomCudaArray b;	// Biases

	CustomCudaArray Z;	// W.A + b
	CustomCudaArray A;	// Input
	CustomCudaArray dA;
public:	
	Layer(std::string name, size_t numUnits, ActivationTypes actType);
	~Layer();
		
	CustomCudaArray& forward(CustomCudaArray& A);
	CustomCudaArray& back(CustomCudaArray& dZ, float learningRate = 0.01);

	std::string getName() { return this->name; };
};

