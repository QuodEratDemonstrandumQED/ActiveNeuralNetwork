/**
 * Created by Quod Erat Demonstrandum on 23/08/2026.
*/

#ifndef QED_ACTIVENEURALNETWORK_HPP
#define QED_ACTIVENEURALNETWORK_HPP

#include <iostream>
#include <sstream>
#include <fstream>
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
    Neuron(NeuronType type, float ActivationThreshold, float LeakRate) : Excitation(0) { }
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
        for (int i = 0; i < hidden; i++) for (int t : Connections[i]) Hidden[i].Axon.push_back(AxonTerminalComponent(&Hidden[i], &Hidden[t]));
        for (int i = 0; i < out; i++) for (int t : OutConnections[i]) Hidden[t].Axon.push_back(AxonTerminalComponent(&Hidden[t], &Out[i]));
    }
    NeuralNetwork(std::string fpath) {
        std::ifstream File(fpath);
        if (!File.is_open()) {
            std::cerr << "Failed to open file: " + fpath << std::endl;
            return;
        }
        std::stringstream buf;
        buf << File.rdbuf();
        std::string FileContents = buf.str();
        File.close();
        NeuronType cType = InputNeuron;
        float ActivationThreshold;
        std::string working_on = "";
        int o;
        for (int i = 0; i < FileContents.length(); i++) {
            char c = FileContents[i];
            bool breakout = false;
            switch (c) {
                case '|':
                    working_on = "";
                    switch (cType) {
                        case InputNeuron:
                            cType = HiddenNeuron;
                            break;
                        case HiddenNeuron:
                            cType = OutputNeuron;
                            break;
                        case OutputNeuron:
                            throw std::invalid_argument("Invalid file formatting for importing of weights for neural network. File path: " + fpath);
                    }
                case ' ':
                    if (working_on.empty()) continue; else if (ActivationThreshold > 0) {
                        switch (cType) {
                            case InputNeuron:
                                In.emplace_back(cType, ActivationThreshold, std::stoi(working_on));
                                break;
                            case HiddenNeuron:
                                Hidden.emplace_back(cType, ActivationThreshold, std::stoi(working_on));
                                break;
                            case OutputNeuron:
                                Out.emplace_back(cType, ActivationThreshold, std::stoi(working_on));
                                break;
                        }
                    } else ActivationThreshold = std::stoi(working_on);
                    working_on = "";
                    break;
                case '~':
                    breakout = true;
                    o = i;
                    break;
                default:
                    working_on += c;
            }
            if (breakout) break;
        }
        working_on = "";
        int a;
        for (int i = o; i < FileContents.length(); i++) {
            char c = FileContents[i];
            switch (c) {
                case '>':
                    a = std::stoi(working_on);
                    working_on = "";
                    break;
                case ' ': {
                    int b = std::stoi(working_on);
                    working_on = "";
                    if (a < In.size()) In[a].Axon.emplace_back(&In[a], &Hidden[b]);
                    else if (b > Hidden.size()) Hidden[a].Axon.emplace_back(&Hidden[a], &Out[b-Hidden.size()]);
                    else Hidden[a].Axon.emplace_back(&Hidden[a], &Hidden[b]);
                    break;
                }
                default:
                    working_on += c;
            }
        }
    }
    void Export(std::string fpath) {
        std::ofstream File(fpath);
        if (!File.is_open()) {
            std::cerr << "Error creating or opening file with path: " + fpath << std::endl;
            return;
        }
        for (Neuron n : In) File << n.ActivationThreshold << " " << n.LeakRate << " ";
        File << "| ";
        for (Neuron n : Hidden) File << n.ActivationThreshold << " " << n.LeakRate << " ";
        File << "| ";
        for (Neuron n : Out) File << n.ActivationThreshold << " " << n.LeakRate << " ";
        File << "~ ";
        for (int i = 0; i < In.size(); i++) for (AxonTerminalComponent ac : In[i].Axon) File << i << ">" << std::distance(Hidden.begin(), std::find(Hidden.begin(), Hidden.end(), *ac.Out)) << " ";
        for (int i = 0; i < Hidden.size(); i++) for (AxonTerminalComponent ac : Hidden[i].Axon) {
            auto it = std::find(Hidden.begin(), Hidden.end(), *ac.Out);
            if (it != Hidden.end()) File << i + In.size() << ">" << std::distance(Hidden.begin(), std::find(Hidden.begin(), Hidden.end(), *ac.Out)) << " ";
            else File << i + In.size() << ">" << std::distance(Out.begin(), std::find(Out.begin(), Out.end(), *ac.Out)) + Hidden.size() << " ";
        }
        File.close();
    }
};

#endif //QED_ACTIVENEURALNETWORK_HPP