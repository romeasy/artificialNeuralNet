/*
 * neuralNet.cpp
 *
 *  Created on: Mar 20, 2015
 *      Author: roman
 */

#include <cassert>
#include "neuralNet.h"


/** PUBLIC **/

NeuralNet::NeuralNet(float _learningRate, std::vector<unsigned int> _architecture) :
           learningRate(_learningRate), architecture(_architecture)
{
	// initialize weight matrices randomly
	// note that for bias, we have one more neuron in source layer...
	for (unsigned int i=0; i<architecture.size()-1; ++i) {
		weights.push_back( arma::randn(architecture.at(i)+1, architecture.at(i+1)) );
	}
}


void NeuralNet::trainNetwork (arma::mat _samples, arma::mat _labels)
{
	std::vector<arma::vec> derivatives;              // the derivative for every neuron
	std::vector<arma::vec> output;                   // the output calculated by the forward pass in each neuron
	std::vector<arma::vec> summedInput;              // the summed input of each neuron
	initialize(summedInput, output, derivatives);

	for (unsigned int i = 0; i<_samples.n_cols; ++i) {
		//_samples.col(i).print("input");
		feedForward(_samples.col(i), summedInput, output);
		feedBackward(_labels.col(i), summedInput, output, derivatives);
		updateWeights(output, derivatives);
		initialize(summedInput, output, derivatives);
	}
}


arma::mat NeuralNet::classifySamples (arma::mat _samples)
{
	std::vector<arma::vec> output;
	std::vector<arma::vec> summedInput;
	std::vector<arma::vec> derivatives;
	arma::mat results = arma::zeros(architecture.at(architecture.size()-1), _samples.n_cols);
	initialize(summedInput, output, derivatives);

	printf("results dims: (%d x %d)\n", results.n_cols, results.n_rows);
	for (unsigned int i=0; i<_samples.n_cols; ++i) {
		arma::vec result = feedForward(_samples.col(i), summedInput, output);
		//result.print("result from feedForward");
		results.col(i) = result;
	}
	return results;
}


arma::vec NeuralNet::feedForward(arma::vec _input, std::vector<arma::vec>& summedInput, std::vector<arma::vec>& output)
{
	// output of first layer is the input vector (plugging in)
	output.at(0) = _input;
	// in every other case, the forward pass is the weighted sum from the previous layer
	for (unsigned int layer = 0; layer<architecture.size()-1; ++layer) {
		//printf("feeding forward to layer -> [%d]\n", layer);
		// instead of iterating in the layer, a simple matrix vector product does the job
		summedInput.at(layer+1) =  weights.at(layer).t() * addBias(output.at(layer));
		//summedInput.at(layer+1).print("feeded forward");
		output.at(layer+1) = sigmoid(summedInput.at(layer+1));
		//output.at(layer+1).print("output forward");
	}
	return output.at(architecture.size()-1); // return last layer
}


arma::mat NeuralNet::getWeights (unsigned int _layer)
{
	return weights.at(_layer);
}


/** PRIVATE **/


void NeuralNet::initialize (std::vector<arma::vec>& summedInput,
		                    std::vector<arma::vec>& output,
		                    std::vector<arma::vec>& derivatives)
{
	// initialize members
	output = std::vector<arma::colvec> (architecture.size());
	derivatives = std::vector<arma::colvec> (architecture.size());
	summedInput = std::vector<arma::colvec> (architecture.size());
	int layerNum = 0;
	for (unsigned int& i : architecture) {
		output.at(layerNum) = arma::ones(i);
		derivatives.at(layerNum) = arma::ones(i);
		summedInput.at(layerNum) = arma::ones(i);
		++layerNum;
	}
}


arma::vec NeuralNet::sigmoid (arma::vec input)
{
	return 1 / (1 + arma::exp(-input));
}


void NeuralNet::feedBackward (arma::vec _labels, std::vector<arma::vec>& summedInput,
		                      std::vector<arma::vec> output, std::vector<arma::vec>& derivatives)
{
	// compute gradients for output layer
	unsigned int i = architecture.size()-1;
	arma::vec sigm = output.at(i);
	derivatives.at(i) = sigm % (1-sigm) % (output.at(i) - _labels);
	//derivatives.at(i) = sigm % (1-sigm) % (_labels - output.at(i));
	//(_labels-output.at(i)).print("t-o");
	//_labels.print("expected");

	// compute the other gradients layerwise
	for (int layer = architecture.size()-2; layer>=0; --layer) {
		arma::vec sigm = output.at(layer);
		//printf("dims of weights at layer %d =  (%d x %d)\n", layer, weights.at(layer).n_cols, weights.at(layer).n_rows);
		arma::mat W = weights.at(layer).submat(0, 0, weights.at(layer).n_rows-2, weights.at(layer).n_cols-1);
		//W.print("W");
		//derivatives.at(layer+1).print("deriv");
		derivatives.at(layer) = (sigm % (1-sigm)) % (W * derivatives.at(layer+1));
	}
}


void NeuralNet::updateWeights (std::vector<arma::vec>& output, std::vector<arma::vec>& derivatives)
{
	for (unsigned int layer = 0; layer<architecture.size()-1; ++layer) {
		arma::mat W = weights.at(layer).submat(0, 0, weights.at(layer).n_rows-2, weights.at(layer).n_cols-1);
		arma::mat weightChange = -learningRate * (derivatives.at(layer+1) * addBias(output.at(layer)).t());
		weights.at(layer) = weights.at(layer) + weightChange.t();
	}
}


arma::vec NeuralNet::addBias (arma::vec _v)
{
	_v.resize(_v.n_elem + 1);
	_v(_v.n_elem - 1) = 1;
	return _v;
}
