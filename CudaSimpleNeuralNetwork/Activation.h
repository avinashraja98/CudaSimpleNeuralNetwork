#pragma once

#include "CustomCudaArray.h"

enum class ActivationTypes
{
	ReLU
};

// Activation Interface
class Activation
{
public:
	virtual void activate(CustomCudaArray& inp) = 0;
};

