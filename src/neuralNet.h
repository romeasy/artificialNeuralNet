/*
 * neuralNet.h
 *
 *  Created on: Mar 20, 2015
 *      Author: roman
 */

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include <armadillo>
#include "jsonSerializer/armadillo.h"

class ANNParameters {
public:
	float learningRate;
	std::vector<unsigned int> architecture;
	std::vector<arma::mat> weights;


	void serialize (jsonSerializer::Node& node)
	{
		node["learningRate"] % learningRate;
		node["architecture"] % architecture;
		node["weights"] % weights;
	}
};


/**
 * Implements a neural network. It is a fully connected network but the weights can be initialized
 * to zero and will then never be changed. Hence, also only partially connected networks are supported here.
 */
class NeuralNet {
public:

	/**
	 * Initializes the neural network. Namely, this function initializes members and does
	 * nothing else for the moment.
	 *
	 * @param _learningRate: The learning rate used to update the weights.
	 * @param _architecture: Each element in the vector corresponds to a layer with n neurons.
	 * @param _weights: The initial weights for the network.
	 */
	NeuralNet(float _learningRate, std::vector<unsigned int> _architecture);


	/**
	 * Initialize the network from a previously stored representation.
	 * That is, the method loads weights, architecture and learning rate
	 * and thus constructs a ready-to-use neural network with previously
	 * learned weights.
	 */
	NeuralNet(std::string _path);


	/**
	 * Trains the neural network using the backpropagation algorithm.
	 * It expects a sample matrix as input which is organized as follows:
	 * Each column of the matrix corresponds to one input vector and
	 * each column of the label matrix corresponds to the expected
	 * output for that column of the sample matrix.
	 *
	 * @brief The function adapts the weight matrices and thereby trains the network
	 *
	 * @param _samples: The sample matrix, containing all the training data
	 * @param _labels: A matrix of the same size, containing the expected output
	 */
	void trainNetwork (uint N, arma::mat _samples, arma::mat _labels);


	/**
	 * Trains the neural network using the offline batch update algorithm.
	 * This algorithm sums up many samples (an epoch) and applies the
	 * weight update only after all samples in the epoch were propagated
	 * back and forth the network.
	 * The size of the epoch is here the same as the number of samples.
	 * Thus, N determines how often the samples are feeded to the net.
	 *
	 * @param N: The number of times, every sample is feeded to the net
	 * @param _samples: The sample matrix
	 * @param _labels: The labels for our training samples
	 */
	void batchLearning (uint N, arma::mat _samples, arma::mat _labels);


	/**
	 * Apply classification on a sample set.
	 */
	arma::mat classifySamples (arma::mat _samples);


	/**
	 * Returns the weights for a given layer.
	 *
	 * @param _layer: The number of the layer (weights from layer i to layer i+1)
	 *
	 * @return A matrix containing the weights. Each col corresponds to the weights for
	 * one neuron in layer i+1 (target) while each row corresponds to a neuron in layer
	 * i (source).
	 */
	arma::mat getWeights (unsigned int _layer);


	/**
	 * Save all relevant parameters of the network to a file.
	 *
	 * @param _path: The path to write the parameters to
	 */
	void save (std::string _path);


	/**
	 * Load parameters from a file.
	 *
	 * @param _path: The path to load from
	 */
	void load (std::string _path);


private:

	/**
	 * Initialize the neural network by constructing the output and derivatives
	 * matrices according to the net's architecture.
	 *
	 * @brief Please note that the weights with entry 0 represent no connection
	 *
	 * @param _weights the initial weights for the network.
	 * @param _layers: each element of the vector represents one layer. The number describes the number of neurons
	 */
	void initialize (std::vector<arma::vec>& output, std::vector<arma::vec>& derivatives);


	/**
	 * Implements the forward pass for our network. An input vector is pllugged
	 * into the network and an output is generated according to it.
	 *
	 * @param _input: the input vector. It must have the same dimensions as the first layer
	 *
	 * @return the output vector of our network
	 */
	arma::vec feedForward (arma::vec _input, std::vector<arma::vec>& output);


	/**
	 * Does the backpropagation algorithm to calculate the partial gradients
	 * in the net.
	 * The function expects the forward pass being already finished for a certain
	 * input. That is, the output matrix must already be calculated.
	 *
	 * @brief The function fills the derivatives matrices and does nothing else.
	 *
	 * @param _labels: the expected values corresponding to the currently already fed input
	 */
	void feedBackward (arma::vec _labels,
			           std::vector<arma::vec>& output,
			           std::vector<arma::vec>& derivatives
	);


	/**
	 * Update the weights according to the previously calculated derivatives.
	 * The method changes the weights matrices as side effect but returns nothing.
	 *
	 * @brief This function only works if the derivatives were calculated already
	 *
	 * @see NeuralNet::feedBackward
	 */
	void updateWeights (std::vector<arma::vec>& output, std::vector<arma::vec>& derivatives);


	/**
	 * Calculates the activation function for a vector.
	 *
	 * @param _input: the input vector
	 * @return a vector of the same size as the input containing the sigmoid values
	 */
	arma::vec sigmoid (arma::vec _input);


	/**
	 * Adds the bias neuron to a given vector. That is, return a vector with
	 * a length of n+1 where the content of the returned vector is the same
	 * as the original one, but with a 1 as last element.
	 *
	 * @note Same as transforming the vector to homogenious form
	 *
	 * @param _v: the vector the 1 should be added to
	 *
	 * @return a vector of form (v, 1)^T
	 */
	arma::vec addBias (arma::vec _v);


	/* MEMBERS */
	float learningRate;                              // how fast do we want to update the weights
	std::vector<unsigned int> architecture;          // each element in the vector represents a layer with n neurons
	std::vector<arma::mat> weights;                  // the i-th element of the vector is a matrix from layer i-1 to i.
	                                                 // Each col in the matrix are the incoming neurons for a given neuron
};

#endif /* NEURALNET_H_ */
