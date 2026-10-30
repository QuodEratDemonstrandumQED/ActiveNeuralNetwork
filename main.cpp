#include "ActiveNeuralNetwork.hpp"

int main() {
    NeuralNetwork NN = NeuralNetwork(5, 5, 25);
    NN.Export("TrainingOutput.nn");
    NeuralNetwork NN2 = NeuralNetwork("TrainingOutput.nn");
    NN2.Export("TrainingOutput2.nn");
    std::random_device rd;
    std::mt19937 gen(rd());
    NeuralNetwork NN3 = NN.Clone(gen, 0.1f, 0.05f);
    NN3.Export("TrainingOutput3.nn");
    float timestep = 0;
    std::vector<std::pair<AxonTerminalComponent*, Signal*>> stored_signals;
    std::vector<Neuron*> ExcitedNeurons;
    for (int i = 0; i < 100; i++) {
        NN3.SetInputs({5, 5, 5, 5, 5});
        std::cout << "Output:" << std::endl;
        for (auto& a : NN3.Step(timestep, stored_signals, ExcitedNeurons)) std::cout << a[0] << " " << a[1] << std::endl;
        std::cout << "\nStored Signals:" << std::endl;
        for (auto& s : stored_signals) std::cout << s.second->Time << std::endl;
        std::cout << "End of " << i << "\n" << std::endl;
    }
    return 0;
}