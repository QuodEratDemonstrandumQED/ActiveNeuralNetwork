#include "ActiveNeuralNetwork.hpp"

int main() {
    NeuralNetwork NN = NeuralNetwork(5, 5, 25);
    NN.Export("TrainingOutput.txt");
    NeuralNetwork NN2 = NeuralNetwork("TrainingOutput.txt");
    NN2.Export("TrainingOutput2.txt");
    return 0;
}