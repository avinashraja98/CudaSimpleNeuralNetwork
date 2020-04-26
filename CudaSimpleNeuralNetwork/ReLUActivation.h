#pragma once
#include "Activation.h"
class ReLUActivation : public Activation
{
public:	
	ReLUActivation();
	~ReLUActivation();

	void activate(CustomCudaArray& inp);
};

