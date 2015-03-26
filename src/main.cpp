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


float calculateSuccessRate (arma::mat guesses, arma::mat labels)
{
	unsigned int positives = 0;
	for (unsigned int i=0; i<guesses.n_cols; ++i) {
		if (arma::norm(guesses.col(i) - labels.col(i)) < 0.3)
			++positives;
	}
	return positives/(float)guesses.n_cols;
}


void testBallClassification ()
{
	printf("SETUP MATRICES\n");
	arma::mat Samples;
	Samples.load("src/ballHistograms2.csv", arma::csv_ascii, true);
	Samples = Samples.cols(0, Samples.n_cols-3).t();
	Samples = arma::shuffle(Samples);
	Samples.col(0).print("first col");
	arma::mat Labels = arma::ones(Samples.n_cols).t();
	printf("Ball Samples is of size (%d x %d)\n", Samples.n_cols, Samples.n_rows);
	printf("Ball Labels is of size (%d x %d)\n", Labels.n_cols, Labels.n_rows);

	printf("TRAINING ANN\n");
	std::vector<unsigned int> architecture {18, 4, 4, 1};
	NeuralNet ann (0.3, architecture);
	ann.trainNetwork(Samples, Labels);
	ann.getWeights(0).print("W 0 -> 1");
	ann.getWeights(1).print("W 1 -> 2");
	ann.getWeights(2).print("W 2 -> 3");

	printf("CLASSIFY DATA\n");
	arma::mat SamplesTest;
	SamplesTest.load("src/ballHistograms.csv", arma::csv_ascii, true);
	SamplesTest = arma::shuffle(SamplesTest.cols(0, SamplesTest.n_cols-2)).t();
	arma::mat guesses = ann.classifySamples(SamplesTest);
	guesses.print("classification");
	printf("success rate on training data: %f\n", calculateSuccessRate(ann.classifySamples(SamplesTest), arma::ones(SamplesTest.n_cols).t()));
}


void testAbalone ()
{
	printf("TESTCASE ABALONE DATASET\n");
	arma::mat Samples;
	Samples.load("src/abalone.csv", arma::csv_ascii, true);
	Samples = arma::shuffle(Samples);
	arma::mat labels_t = Samples.col(0);

	// construct labels
	printf("labels_t is of size (%d x %d)\n", labels_t.n_cols, labels_t.n_rows);
	arma::mat labels = arma::zeros(labels_t.n_rows, 1);
	printf("labels is of size (%d x %d)\n", labels.n_cols, labels.n_rows);
	arma::mat infants = arma::conv_to<arma::mat>::from(arma::any( labels_t.t() == 2 ));
	arma::mat males = arma::conv_to<arma::mat>::from(arma::any( labels_t.t() == 1 ));
	arma::mat females = arma::conv_to<arma::mat>::from(arma::any( labels_t.t() == 0 ));
	printf("infants is of size (%d x %d)\n", infants.n_cols, infants.n_rows);
	labels = males + females*2; // -> output should be 0 for infants, 1 for males, 2 for females
	//labels.col(0) = infants.t();
	//labels.col(1) = males.t();
	//labels.col(2) = females.t();
	//labels.print("LABELS");
	// construct other matrices
	arma::mat trainingSamples = Samples.submat(0, 1, (int)(Samples.n_rows*0.9), Samples.n_cols-1).t() / 200.;
	arma::mat trainingLabels = labels.submat(0, 0, labels.n_rows-1, (int)(labels.n_cols*0.9));
	arma::mat testData = Samples.submat((int)(Samples.n_rows*0.9)+1, 1, Samples.n_rows-1, Samples.n_cols-1).t() / 200.;
	printf("labels size (%d x %d)\n", labels.n_cols, labels.n_rows);
	arma::mat testLabels = labels.submat(0, (int)(labels.n_cols*0.9)+1, labels.n_rows-1, labels.n_cols-1);

	printf("trainingSamples is of size (%d x %d)\n", trainingSamples.n_cols, trainingSamples.n_rows);
	printf("trainingLabels is of size (%d x %d)\n", trainingLabels.n_cols, trainingLabels.n_rows);
	printf("testSamples is of size (%d x %d)\n", testData.n_cols, testData.n_rows);
	printf("testLabels is of size (%d x %d)\n", testLabels.n_cols, testLabels.n_rows);

	printf("loading complete\n");
	std::vector<unsigned int> architecture {8, 10, 1};
	printf("neural net initialized. Commencing with training\n");
	NeuralNet ann (3, architecture);
	ann.trainNetwork(trainingSamples, trainingLabels);

	printf("Training complete!\n");
	ann.getWeights(0).print("W_0");
	ann.getWeights(1).print("W_1");
	//ann.getWeights(2).print("W_2");

	printf("starting classification for remaining data\n");
	arma::mat classifiedData = ann.classifySamples(testData);
	printf("classified data dims: (%d x %d)\n", classifiedData.n_cols, classifiedData.n_rows);
	unsigned int correctOnes = 0;
	for (unsigned int i = 0; i<classifiedData.n_cols; ++i) {
		double error = 1/2. * arma::accu( arma::pow(testLabels.col(i) - classifiedData.col(i), 2) );
		classifiedData.col(i).t().print("classified as");
		testLabels.col(i).t().print("labelled as");
		if (error <= 1.)
			++correctOnes;
	}
	printf("correct classified: %d\n", correctOnes);
	printf("false classified: %d\n", classifiedData.n_cols-correctOnes);
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
	NeuralNet ann (0.5, architecture);
	ann.getWeights(0).print("W_1");
	ann.getWeights(1).print("W_2");

	printf("TRAINING\n");
	//ann.trainNetwork(samples.rows(0,1).cols(0, numOfSamples/2), labels.cols(0, numOfSamples/2));
	ann.trainNetwork(samples.rows(0,1), labels);

	ann.getWeights(0).print("W_1");
	ann.getWeights(1).print("W_2");

	printf("CLASSIFY\n");
	arma::mat f = mvrnorm(100, arma::ones(2,1)*distance, arma::eye(2,2)*variance).t();
	arma::mat guessesF = ann.classifySamples(f);
	guessesF.print("results from (1, 1)");
	arma::mat s = mvrnorm(100, mu2, arma::eye(2,2)*variance).t();
	arma::mat guessesS = ann.classifySamples(s);
	guessesS.print("results from (1, -1)");
	arma::mat t = mvrnorm(100, mu2*(-1), arma::eye(2,2)*variance).t();
	arma::mat guessesT = ann.classifySamples(t);
	guessesT.print("results from (-1, 1)");
	arma::mat f2 = mvrnorm(100, arma::ones(2,1)*(-distance), arma::eye(2,2)*variance).t();
	arma::mat guessesF2 = ann.classifySamples(f2);
	guessesF2.print("results from (-1, -1)");
	printf("Success rate: %f\n", calculateSuccessRate(guessesF, arma::ones(100).t()));
	printf("Success rate: %f\n", calculateSuccessRate(guessesS, arma::zeros(100).t()));
	printf("Success rate: %f\n", calculateSuccessRate(guessesT, arma::zeros(100).t()));
	printf("Success rate: %f\n", calculateSuccessRate(guessesF2, arma::ones(100).t()));
}

int main()
{
	testXOR();
	//testAbalone();
	//testBallClassification();
}
