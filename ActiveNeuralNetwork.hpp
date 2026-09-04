//
// Created by troyg on 23/08/2026.
//

#ifndef QED_ACTIVENEURALNETWORK_HPP
#define QED_ACTIVENEURALNETWORK_HPP

#include <vector>



class Neuron;

/**
 * @brief The terminal of the axon of a neuron. Responsible for determining the time and weight of the connection from one neuron to another, one-way.
 */
class AxonTerminalComponent {
public:
    Neuron* In;
    Neuron* Out;
    float Weight;
    float MyelinSheath;
    std::vector<float[2]> Signals;
    AxonTerminalComponent(Neuron*& in, Neuron*& out) : In(in), Out(out), Weight(1), MyelinSheath(1) { }
};

typedef std::vector<AxonTerminalComponent> AxonComponent;

enum NeuronType { HiddenNeuron, InputNeuron, OutputNeuron };

/**
 * @brief A neuron in a neural network.
 */
class Neuron {
public:
    NeuronType Type;
    AxonComponent Axon;
    float Excitation;
    float ActivationThreshold;
    float LeakRate;
    Neuron(NeuronType type) : Type(type), Excitation(0), ActivationThreshold(1), LeakRate(1) { }
};


class NeuralNetwork {
    std::vector<Neuron> In;
    std::vector<Neuron> Out;
    std::vector<Neuron> Hidden;
    NeuralNetwork(int in, int out, int hidden) {
        for (int i = 0; i < in; i++) In.emplace_back(InputNeuron);
        for (int i = 0; i < hidden; i++) Hidden.emplace_back(HiddenNeuron);
        for (int i = 0; i < out; i++) Out.emplace_back(OutputNeuron);
    }
};


#endif //QED_ACTIVENEURALNETWORK_HPP
