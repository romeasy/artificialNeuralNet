/*
 * main.c
 *
 *  Created on: Mar 20, 2015
 *      Author: roman
 */

#include "neuralNet.h"
#include <armadillo>


/**
 * Generate samples from multivariate gaussian.
 * The result is a matrix with each column representing
 * a sample.
 */
arma::mat mvrnorm(int n, arma::vec mu, arma::mat sigma) {
   int ncols = sigma.n_cols;
   arma::mat Y = arma::randn(n, ncols);
   return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}


/**
 * Calculate the number of successful classifications and print some stats.
 */
float calculateSuccessRate (arma::mat guesses, arma::mat labels)
{
	unsigned int positives = 0;
	uint falsePositives = 0;
	uint falseNegatives = 0;
	for (unsigned int i=0; i<guesses.n_cols; ++i) {
		if (arma::norm(guesses.col(i) - labels.col(i)) < 0.5)
			++positives;
		else {
			if (arma::norm(labels.col(i)) > 0)
				++falseNegatives;
			else
				++falsePositives;
		}
	}
	printf("positives: %d\n", positives);
	printf("false positives: %d\tfalse negatives: %d\n", falsePositives, falseNegatives);
	return positives/(float)guesses.n_cols;
}





void testXOR ()
{
	printf("TESTCASE XOR\n");
	unsigned int numOfSamples = 4000000;
	float distance = 5.;
	float variance = 1.5;
	printf("VARIANCE OF TESTDATA: %f\tDISTANCE FROM ORIGIN: %f\n", variance, distance);
	arma::mat samples = arma::ones(3, numOfSamples);
	samples.cols(samples.n_cols/2, samples.n_cols-1).fill(0);
	arma::mat first = mvrnorm(numOfSamples/4, arma::ones(2,1)*distance, arma::eye(2,2)*variance);
	arma::colvec2 mu2;
	mu2 << distance << -distance << arma::endr;
	arma::mat second = mvrnorm(numOfSamples/4, mu2, arma::eye(2,2)*variance);
	arma::mat third = mvrnorm(numOfSamples/4, mu2*(-1), arma::eye(2,2)*variance);
	arma::mat fourth = mvrnorm(numOfSamples/4, arma::ones(2,1)*(-distance), arma::eye(2,2)*variance);

	samples.submat(0, 0, 1, numOfSamples/4-1) = first.t();
	samples.submat(0, numOfSamples/4, 1, numOfSamples/2-1) = fourth.t();
	samples.submat(0, numOfSamples/2, 1, numOfSamples*3/4-1) = second.t();
	samples.submat(0, numOfSamples*3/4, 1, numOfSamples-1) = third.t();
	samples = arma::shuffle(samples.t()).t();
	arma::rowvec labels = samples.row(2);
	printf("CREATED TEST SETUP\n");
	//labels.print("labels");
	//samples.print("samples");

	printf("INITIALIZING ANN\n");
	std::vector<unsigned int> architecture {2, 2, 1};
	NeuralNet ann (0.1, architecture);

	printf("TRAINING\n");
	//ann.trainNetwork(samples.rows(0,1).cols(0, numOfSamples/2), labels.cols(0, numOfSamples/2));
	ann.trainNetwork(1, samples.rows(0,1), labels);

	ann.getWeights(0).print("W_1");
	ann.getWeights(1).print("W_2");

	printf("CLASSIFY\n");
	arma::mat f = mvrnorm(100, arma::ones(2,1)*distance, arma::eye(2,2)*variance).t();
	arma::mat guessesFirstQuadrant = ann.classifySamples(f);
	guessesFirstQuadrant.print("results from (1, 1)");
	arma::mat s = mvrnorm(100, mu2, arma::eye(2,2)*variance).t();
	arma::mat guessesSecondQuadrant = ann.classifySamples(s);
	guessesSecondQuadrant.print("results from (1, -1)");
	arma::mat t = mvrnorm(100, mu2*(-1), arma::eye(2,2)*variance).t();
	arma::mat guessesThirdQuadrant = ann.classifySamples(t);
	guessesThirdQuadrant.print("results from (-1, 1)");
	arma::mat f2 = mvrnorm(100, arma::ones(2,1)*(-distance), arma::eye(2,2)*variance).t();
	arma::mat guessesFourthQuadrant = ann.classifySamples(f2);
	guessesFourthQuadrant.print("results from (-1, -1)");
	printf("Success rate: %f\n", calculateSuccessRate(guessesFirstQuadrant, arma::ones(100).t()));
	printf("Success rate: %f\n", calculateSuccessRate(guessesSecondQuadrant, arma::zeros(100).t()));
	printf("Success rate: %f\n", calculateSuccessRate(guessesThirdQuadrant, arma::zeros(100).t()));
	printf("Success rate: %f\n", calculateSuccessRate(guessesFourthQuadrant, arma::ones(100).t()));
}

int main()
{
	testXOR();
}
