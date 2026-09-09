/**
 * Created by Quod Erat Demonstrandum on 23/08/2026.
*/

#ifndef QED_ACTIVENEURALNETWORK_HPP
#define QED_ACTIVENEURALNETWORK_HPP

#include <vector>
#include <random>
#include <unordered_map>
#include <algorithm>



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
    AxonTerminalComponent(Neuron* in, Neuron* out) : In(in), Out(out), Weight(1), MyelinSheath(1) { }
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
public:
    NeuralNetwork(int in, int out, int hidden) {
        for (int i = 0; i < in; i++) In.emplace_back(InputNeuron);
        for (int i = 0; i < hidden; i++) Hidden.emplace_back(HiddenNeuron);
        for (int i = 0; i < out; i++) Out.emplace_back(OutputNeuron);

        std::unordered_map<int, std::vector<int>> InConnections;
        std::unordered_map<int, std::vector<int>> Connections;
        std::unordered_map<int, std::vector<int>> OutConnections;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, hidden-1);
        for (int i = 0; i < in; ++i) {
            for (int j = 0; j <= std::min(10*in, hidden/5); ++j) {
                int r = dist(gen);
                if (std::find(InConnections[i].begin(), InConnections[i].end(), r) != InConnections[i].end()) j--;
                else InConnections[i].push_back(r);
            }
        }
        for (int i = 0; i < hidden; ++i) {
            for (int j = 0; j <= hidden/100; ++j) {
                int r = dist(gen);
                if (std::find(Connections[i].begin(), Connections[i].end(), r) != Connections[i].end() || std::find(Connections[r].begin(), Connections[r].end(), i) != Connections[r].end()) j--;
                else Connections[i].push_back(r);
            }
        }
        for (int i = 0; i < in; ++i) {
            for (int j = 0; j <= std::min(10*out, hidden/5); ++j) {
                int r = dist(gen);
                if (std::find(OutConnections[i].begin(), OutConnections[i].end(), r) != OutConnections[i].end()) j--;
                else OutConnections[i].push_back(r);
            }
        }

        for (int i = 0; i < in; i++) for (int t : InConnections[i]) In[i].Axon.push_back(AxonTerminalComponent(&In[i], &Hidden[t]));
    }
};


#endif //QED_ACTIVENEURALNETWORK_HPP
