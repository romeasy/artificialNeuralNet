/*
 * neuralNet.h
 *
 *  Created on: Mar 20, 2015
 *      Author: roman
 */

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include <armadillo>

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
	void trainNetwork (arma::mat _samples, arma::mat _labels);


	/**
	 * Apply classification on a sample set.
	 */
	arma::mat classifySamples (arma::mat _samples);


	/**
	 * Implements the forward pass for our network. An input vector is pllugged
	 * into the network and an output is generated according to it.
	 *
	 * @param _input: the input vector. It must have the same dimensions as the first layer
	 *
	 * @return the output vector of our network
	 */
	arma::vec feedForward (arma::vec _input,
			               std::vector<arma::vec>& summedInput,
			               std::vector<arma::vec>& output
	);


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

private:

	/**
	 * Initialize the neural network by initializing the weights and
	 *
	 * @brief Please note that the weights with entry 0 represent no connection
	 *
	 * @param _weights the initial weights for the network.
	 * @param _layers: each element of the vector represents one layer. The number describes the number of neurons
	 */
	void initialize (std::vector<arma::vec>& summedInput,
                     std::vector<arma::vec>& output,
                     std::vector<arma::vec>& derivatives
    );


	/**
	 * Calculates the activation function for a vector.
	 *
	 * @param _input: the input vector
	 * @return a vector of the same size as the input containing the sigmoid values
	 */
	arma::vec sigmoid (arma::vec _input);


	/**
	 * Does the backpropagation algorithm to calculate the partial gradients
	 * in the net.
	 * The function expects the forward pass being already finished for a certain
	 * input. That is, the summedInput matrices as well as the output matrices must
	 * already be calculated.
	 *
	 * @brief The function fills the derivatives matrices and does nothing else.
	 *
	 * @param _labels: the expected values corresponding to the currently already fed input
	 */
	void feedBackward (arma::vec _labels,
			           std::vector<arma::vec>& summedInput,
			           std::vector<arma::vec> output,
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


	arma::vec addBias (arma::vec _v);


	/* MEMBERS */
	float learningRate;                              // how fast do we want to update the weights
	std::vector<unsigned int> architecture;          // each element in the vector represents a layer with n neurons
	std::vector<arma::mat> weights;                  // the i-th element of the vector is a matrix from layer i-1 to i.
	                                                 // Each col in the matrix are the incoming neurons for a given neuron
};

#endif /* NEURALNET_H_ */
