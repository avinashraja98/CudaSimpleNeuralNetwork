
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <time.h>
#include <iostream>
#include "DataSet.h"
#include "NNetwork.h"

int main()
{
	srand(time(NULL));
	std::cout << "hello\n";
	
	DataSet d;

	NNetwork net;
	net.addLayer(new Layer("hiddenLayer1", 28 * 28, 32, ActivationTypes::ReLU));
	
	return 0;
}
