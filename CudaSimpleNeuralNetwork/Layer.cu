#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Layer.h"
#include "ReLUActivation.h"

#include <assert.h>
#include <random>

__global__ void forwardKernel(float* W, float* A, float* Z, float* b,
	int W_x_dim, int W_y_dim,
	int A_x_dim, int A_y_dim) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int Z_x_dim = W_x_dim;
	int Z_y_dim = A_y_dim;

	float Z_value = 0;

	if (row < Z_x_dim && col < Z_y_dim) {
		/*
		for (int i = 0; i < W_y_dim; i++) {
			Z_value += W[row * W_y_dim + i] * A[i * A_y_dim + col];
		}
		Z[row * Z_y_dim + col] = Z_value + b[row];
		*/
		Z[0] = 1.0f;
		int a = 1;
	}
}

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

Layer::Layer(std::string name, size_t numInputs, size_t numUnits, ActivationTypes actType) : W(numUnits, numInputs), b(numUnits)
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

CustomCudaArray & Layer::forward(CustomCudaArray & A)
{
	// Z = W.A + b
	// For matrix multiplication
	assert(W.getY() == A.getX());

	Z[0] = 0.0;
	this->A = A;

	Z.resizeAndReset(W.getX(), A.getY());

	dim3 block_size(16, 16);
	
	dim3 num_of_blocks(((unsigned int)W.getX() + block_size.x - 1) / block_size.x,
		((unsigned int)A.getY() + block_size.y - 1) / block_size.y);

	forwardKernel<<<num_of_blocks, block_size >>>(W.dataPtr.get(),
		A.dataPtr.get(),
		Z.dataPtr.get(),
		b.dataPtr.get(),
		(int)W.getX(), (int)W.getY(),
		(int)A.getX(), (int)A.getY());

	cudaDeviceSynchronize();

	return Z;
}
