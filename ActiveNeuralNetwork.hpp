/**
 * Created by Quod Erat Demonstrandum on 23/08/2026.
*/

#ifndef QED_ACTIVENEURALNETWORK_HPP
#define QED_ACTIVENEURALNETWORK_HPP

#include <iostream>
#include <sstream>
#include <fstream>
#include <array>
#include <vector>
#include <random>
#include <unordered_map>
#include <algorithm>
#include <iterator>
#include <utility>
#include <ranges>
#include <cmath>



struct Signal {
    float Time;
    int Index;
};

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
    std::vector<Signal> Signals;
    AxonTerminalComponent(Neuron* in, Neuron* out) : In(in), Out(out), Weight(1), MyelinSheath(1) { }
    AxonTerminalComponent(Neuron* in, Neuron* out, float weight, float myelin_sheath) : In(in), Out(out), Weight(weight), MyelinSheath(myelin_sheath) { }
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
    float TimeSinceLastSignal;
    std::vector<std::array<float, 2>> RecievedSignals;
    Neuron(NeuronType type) : Type(type), Excitation(0), ActivationThreshold(1), LeakRate(1), TimeSinceLastSignal(1e9) { }
    Neuron(NeuronType type, float activation_threshold, float leak_rate) : Type(type), Excitation(0), ActivationThreshold(activation_threshold), LeakRate(leak_rate), TimeSinceLastSignal(1e9) { }
    Neuron(const Neuron& n) : Excitation(0.0f), TimeSinceLastSignal(1e9) {
        Type = n.Type;
        Axon = n.Axon;
        ActivationThreshold = n.ActivationThreshold;
        LeakRate = n.LeakRate;
    }

    float ExcitationLoss(float timestep) {
        float time = 0;
        for (std::array<float, 2> signal : RecievedSignals) {
            float t = signal[0] - time;
            if (Excitation > 0.0f) Excitation = std::max(Excitation - LeakRate * std::min(TimeSinceLastSignal, t), 0.0f);
            else if (Excitation < 0.0f) Excitation = std::min(Excitation + LeakRate * std::min(TimeSinceLastSignal, timestep), 0.0f);
            time = signal[0];
            Excitation += signal[1];
        }
        float t = timestep-time;
        if (Excitation > 0.0f) Excitation = std::max(Excitation - LeakRate * std::min(TimeSinceLastSignal, t), 0.0f);
        else if (Excitation < 0.0f) Excitation = std::min(Excitation + LeakRate * std::min(TimeSinceLastSignal, timestep), 0.0f);
    }
};


/**
 * @brief A neural network.
 */
class NeuralNetwork {
    std::vector<Neuron*> In;
    std::vector<Neuron*> Hidden;
    std::vector<Neuron*> Out;
    float ShortestPath;
public:
    NeuralNetwork(int in, int out, int hidden) {
        std::unordered_map<int, std::vector<int>> InConnections;
        std::unordered_map<int, std::vector<int>> Connections;
        std::unordered_map<int, std::vector<int>> OutConnections;

        for (int i = 0; i < in; i++) In.push_back(new Neuron(InputNeuron));
        for (int i = 0; i < hidden; i++) Hidden.push_back(new Neuron(HiddenNeuron));
        for (int i = 0; i < out; i++) Out.push_back(new Neuron(OutputNeuron));

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

        for (int i = 0; i < in; i++) for (int t : InConnections[i]) In[i]->Axon.push_back(AxonTerminalComponent(In[i], Hidden[t]));
        for (int i = 0; i < hidden; i++) for (int t : Connections[i]) Hidden[i]->Axon.push_back(AxonTerminalComponent(Hidden[i], Hidden[t]));
        for (int i = 0; i < out; i++) for (int t : OutConnections[i]) Hidden[t]->Axon.push_back(AxonTerminalComponent(Hidden[t], Out[i]));

        ShortestPath = 1.0f;
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
        ShortestPath = -1.0f;
        NeuronType cType = InputNeuron;
        float ActivationThreshold = 0;
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
                    break;
                case ' ':
                    if (working_on.empty()) continue; if (ActivationThreshold > 0) {
                        switch (cType) {
                            case InputNeuron:
                                In.push_back(new Neuron(cType, ActivationThreshold, std::stof(working_on)));
                                break;
                            case HiddenNeuron:
                                Hidden.push_back(new Neuron(cType, ActivationThreshold, std::stof(working_on)));
                                break;
                            case OutputNeuron:
                                Out.push_back(new Neuron(cType, ActivationThreshold, std::stof(working_on)));
                                break;
                        }
                        ActivationThreshold = 0;
                    } else ActivationThreshold = std::stof(working_on);
                    working_on = "";
                    break;
                case '~':
                    breakout = true;
                    o = i + 1;
                    break;
                default:
                    working_on += c;
            }
            if (breakout) break;
        }
        working_on = "";
        int a = 0;
        int b = 0;
        float w = 0;
        for (int i = o; i < FileContents.length(); i++) {
            char c = FileContents[i];
            switch (c) {
            case '>':
                a = std::stoi(working_on);
                working_on = "";
                break;
            case ' ': {
                if (working_on.empty()) break;
                if (b == 0) {
                    b = std::stoi(working_on);
                    working_on = "";
                } else if (w == 0) {
                    w = std::stof(working_on);
                    working_on = "";
                } else {
                    float ms = std::stof(working_on);
                    if (ShortestPath != 0.0f) ShortestPath = std::min(ShortestPath, 1.0f/ms);
                    else ShortestPath = 1.0f/ms;
                    working_on = "";
                    if (a < In.size()) In[a]->Axon.emplace_back(In[a], Hidden[b], w, ms);
                    else if (b >= Hidden.size()) Hidden[a - In.size()]->Axon.emplace_back(Hidden[a - In.size()], Out[b - Hidden.size()], w, ms);
                    else Hidden[a - In.size()]->Axon.emplace_back(Hidden[a - In.size()], Hidden[b], w, ms);
                    a = 0; b = 0; w = 0; ms = 0;
                    break;
                }
            } default:
                working_on += c;
            }
        }
    }
    NeuralNetwork(const NeuralNetwork& NN) {
        for (Neuron* n : NN.In) {
            In.push_back(new Neuron(*n));
        }
        for (Neuron* n : NN.Hidden) {
            Hidden.push_back(new Neuron(*n));
        }
        for (Neuron* n : NN.Out) {
            Out.push_back(new Neuron(*n));
        }
        for (Neuron*& n : In) for (AxonTerminalComponent& at : n->Axon) at.Out = Hidden[std::distance(NN.Hidden.begin(), std::find(NN.Hidden.begin(), NN.Hidden.end(), at.Out))];
        for (Neuron*& n : Hidden) for (AxonTerminalComponent& at : n->Axon) {
            auto it = std::find(NN.Hidden.begin(), NN.Hidden.end(), at.Out);
            if (it != NN.Hidden.end()) {
                at.Out = Hidden[std::distance(NN.Hidden.begin(), it)];
            } else at.Out = Out[std::distance(NN.Out.begin(), std::find(NN.Out.begin(), NN.Out.end(), at.Out))];
        }
    }
    ~NeuralNetwork() {
        for (Neuron* n : In) delete n;
        for (Neuron* n : Hidden) delete n;
        for (Neuron* n : Out) delete n;
    }
    void Export(std::string fpath) {
        std::ofstream File(fpath);
        if (!File.is_open()) {
            std::cerr << "Error creating or opening file with path: " + fpath << std::endl;
            return;
        }
        for (Neuron* n : In) File << n->ActivationThreshold << " " << n->LeakRate << " ";
        File << "| ";
        for (Neuron* n : Hidden) File << n->ActivationThreshold << " " << n->LeakRate << " ";
        File << "| ";
        for (Neuron* n : Out) File << n->ActivationThreshold << " " << n->LeakRate << " ";
        File << "~ ";
        for (int i = 0; i < In.size(); i++) for (AxonTerminalComponent ac : In[i]->Axon) File << i << ">" << std::distance(Hidden.begin(), std::find(Hidden.begin(), Hidden.end(), ac.Out)) << " " << ac.Weight << " " << ac.MyelinSheath << " ";
        for (int i = 0; i < Hidden.size(); i++) for (AxonTerminalComponent ac : Hidden[i]->Axon) {
            auto it = std::find(Hidden.begin(), Hidden.end(), ac.Out);
            if (it != Hidden.end()) File << i + In.size() << ">" << std::distance(Hidden.begin(), std::find(Hidden.begin(), Hidden.end(), ac.Out)) << " ";
            else File << i + In.size() << ">" << std::distance(Out.begin(), std::find(Out.begin(), Out.end(), ac.Out)) + Hidden.size() << " ";
            File << ac.Weight << " " << ac.MyelinSheath << " ";
        }
        File.close();
    }
    NeuralNetwork Clone(std::mt19937& gen, float cRate, float wRate) {
        NeuralNetwork clone = NeuralNetwork(*this);
        for (Neuron* n : clone.In) n->Excitation = 0;
        for (Neuron* n : clone.Hidden) n->Excitation = 0;
        for (Neuron* n : clone.Out) n->Excitation = 0;

        if (cRate > 0) {
            int count = 0;
            std::uniform_real_distribution<float> dist(0, 1);
            for (Neuron*& a : clone.Hidden) for (Neuron*& b : clone.Hidden) if (a != b && dist(gen) < cRate/2.0f) {
                bool connected = false;
                for (int i = 0; i < a->Axon.size(); i++) {
                    AxonTerminalComponent ac = a->Axon[i];
                    if (ac.Out == b) {
                        connected = true;
                        b->Axon.erase(b->Axon.begin() + i);
                        break;
                    }
                }
                if (!connected) {
                    for (int i = 0; i < b->Axon.size(); i++) {
                        AxonTerminalComponent ac = b->Axon[i];
                        if (ac.Out == a) {
                            connected = true;
                            a->Axon.erase(a->Axon.begin() + i);
                            break;
                        }
                    }
                    if (!connected) a->Axon.emplace_back(a, b);
                }
            }
        }
        if (wRate > 0) {
            std::uniform_real_distribution<float> dist(-wRate, wRate);
            for (Neuron*& a : clone.Hidden) for (AxonTerminalComponent& at : a->Axon) {
                at.Weight += dist(gen);
                at.MyelinSheath += dist(gen);
                at.MyelinSheath = std::abs(at.MyelinSheath);
                if (at.MyelinSheath < 0.1) at.MyelinSheath = 0.1;
            }
        }
        clone.ShortestPath = 0.0f;
        for (Neuron*& n : clone.In) for (AxonTerminalComponent& ac : n->Axon)
            if (clone.ShortestPath != 0.0f) clone.ShortestPath = std::min(clone.ShortestPath, 1.0f/ac.MyelinSheath);
            else clone.ShortestPath = 1.0f/ac.MyelinSheath;
        return clone;
    }
    std::vector<std::array<float, 2>> Step(float& timestep, std::vector<std::pair<AxonTerminalComponent*, int>>& stored_signals, std::vector<Neuron*> ExcitedNeurons) {
        timestep = ShortestPath;
        std::vector<std::array<float, 2>> output;
        if (stored_signals.size() == 0) {
            for (Neuron*& n : In) for (AxonTerminalComponent ac : n->Axon) if (ac.Signals.size() != 0) for (int i = 0; i < ac.Signals.size(); i++) stored_signals.push_back({&ac, i});
            for (Neuron*& n : Hidden) for (AxonTerminalComponent ac : n->Axon) if (ac.Signals.size() != 0) for (int i = 0; i < ac.Signals.size(); i++) stored_signals.push_back({&ac, i});
        }
        std::vector<std::pair<AxonTerminalComponent*, int>> new_signals;
        std::vector<int> removed_indices;
        for (int i = 0; i < stored_signals.size(); ++i) {
            std::pair<AxonTerminalComponent*, int>& acp = stored_signals[i];
            AxonTerminalComponent*& ac = acp.first;
            Signal& s = ac->Signals[acp.second];
            s.Time += timestep;
            float st = 1/ac->MyelinSheath;
            if (s.Time >= st) {
                ac->Out->Excitation += ac->Weight;
                if (ac->Out->Excitation - (timestep - st) * ac->Out->LeakRate >= ac->Out->ActivationThreshold) {
                    ac->Out->TimeSinceLastSignal = s.Time - st;
                    if (ac->Out->Type == OutputNeuron) output.push_back({1, s.Time - st});
                    else for (AxonTerminalComponent& at : ac->Out->Axon) {
                        int index = at.Signals.size();
                        at.Signals.push_back({s.Time - st, index});
                        new_signals.push_back({&at, index});
                        ac->Out->RecievedSignals.push_back({s.Time - st, at.Weight});
                    }
                }
                for (int i = s.Index; i < ac->Signals.size(); i++) ac->Signals[i].Index--;
                ac->Signals.erase(ac->Signals.begin() + s.Index);
                removed_indices.push_back(i);
            }
        }
        for (Neuron* n : Hidden) {
            if (n->Excitation > 0.0f) n->Excitation = std::max(n->Excitation - n->LeakRate * std::min(n->TimeSinceLastSignal, timestep), 0.0f);
            else if (n->Excitation < 0.0f) n->Excitation = std::min(n->Excitation + n->LeakRate * std::min(n->TimeSinceLastSignal, timestep), 0.0f);
        }
        for (Neuron* n : Out) {
            if (n->Excitation > 0.0f) n->Excitation = std::max(n->Excitation - n->LeakRate * std::min(n->TimeSinceLastSignal, timestep), 0.0f);
            else if (n->Excitation < 0.0f) n->Excitation = std::min(n->Excitation + n->LeakRate * std::min(n->TimeSinceLastSignal, timestep), 0.0f);
        }

        for (const int& i : std::views::reverse(removed_indices)) removed_indices.erase(removed_indices.begin() + i);
        stored_signals.reserve(stored_signals.size() + new_signals.size());
        stored_signals.insert(stored_signals.end(), new_signals.begin(), new_signals.end());
        std::sort(stored_signals.begin(), stored_signals.end(), [](const std::pair<AxonTerminalComponent*, int>& a, const std::pair<AxonTerminalComponent*, int>& b) {
            return (1.0f/a.first->MyelinSheath)-a.first->Signals[a.second].Time < (1.0f/b.first->MyelinSheath)-b.first->Signals[b.second].Time;
        });
        return output;
    }
};

#endif //QED_ACTIVENEURALNETWORK_HPP