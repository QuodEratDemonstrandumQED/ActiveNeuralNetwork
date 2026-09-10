#include "ActiveNeuralNetwork.hpp"

int main() {
    NeuralNetwork NN = NeuralNetwork(5, 5, 25);
    NN.Export("TrainingOutput.txt");
    return 0;
}