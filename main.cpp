#include "ActiveNeuralNetwork.hpp"

int main() {
    NeuralNetwork NN = NeuralNetwork(5, 5, 25);
    NN.Export("TrainingOutput.txt");
    NeuralNetwork NN2 = NeuralNetwork("TrainingOutput.txt");
    NN2.Export("TrainingOutput2.txt");
    std::random_device rd;
    std::mt19937 gen(rd());
    NeuralNetwork NN3 = NN.Clone(gen, 0.1f, 0.05f);
    NN3.Export("TrainingOutput3.txt");
    return 0;
}