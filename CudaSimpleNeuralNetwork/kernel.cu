
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

	CustomCudaArray Z = net.forward(d.training_data.front().image);

	int a = 1;
	
	return 0;
}
