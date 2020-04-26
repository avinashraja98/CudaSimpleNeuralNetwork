#include "Layer.h"
#include "ReLUActivation.h"
#include <random>

void Layer::fillWithZeroes(CustomCudaArray & inp)
{
	size_t inpSize = inp.getSize();
	for (int i = 0; i < inpSize; i++) {
		inp[i] = (float)0;
	}
}

void Layer::fillWithRandom(CustomCudaArray & inp)
{
	std::default_random_engine generator;
	std::normal_distribution<float> normal_distribution(0.0, 1.0);
	
	size_t inpSize = inp.getSize();

	for (int i = 0; i < inpSize; i++) {
		inp[i] = normal_distribution(generator) * random_init_threshold;
	}
}

Layer::Layer(std::string name, size_t numInputs, size_t numUnits, ActivationTypes actType) : W(numInputs, numUnits), b(1, numUnits)
{
	this->name = name;
	this->numUnits = numUnits;
	this->numInputs = numInputs;

	fillWithZeroes(b);
	fillWithRandom(W);

	switch (actType)
	{
	case ActivationTypes::ReLU:
		activation = new ReLUActivation();
		break;
	default:
		activation = new ReLUActivation();
		break;
	}
}

Layer::~Layer()
{
	delete activation;
}
