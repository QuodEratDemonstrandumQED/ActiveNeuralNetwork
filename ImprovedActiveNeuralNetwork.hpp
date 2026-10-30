#ifndef QED_IANN
#define QED_IANN

#include <random>
#include <algorithm>
#include <vector>
#include <exception>
#include <stdexcept>


namespace QED {
	/**
	 * @brief Improved Active Neural Network
	 * 
	 * The improved version of the badly-written ActiveNeuralNetwork.hpp project, meant to do things more efficiently.
	 *
	 * Timestep: min(shortest connection, shortest inter-signal time)
	 */
	namespace IANN {
		// Forward declarations
		class Neuron;
		class Axon;
		class NeuralNetwork;

		/**
		 * @brief A signal sent through an Axon from one Neuron to another.
		 */
		struct Signal {
			float Time;
			float Weight;
			Axon* Sender;
			unsigned AxonIndex;
		};

		enum NeuronType { InputNeuron, HiddenNeuron, OutputNeuron };

		/**
		 * @brief The terminal of an Axon.
		 * 
		 * Singular: Telodendron
		 * Plural:   Telodendria
		 * 
		 * Points to the target neuron to send signals to.
		 */
		class Telodendron {
			Neuron* Target;
			Axon* Sender;
			float Weight;
		public:
			Telodendron(Neuron*& tgt, Axon*& snd, float wgt) : Target(tgt), Sender(snd), Weight(wgt) { }
			Telodendron Copy(NeuralNetwork*& nn, Axon*& ax);
			Signal* SendSignal(Signal s, float t);
			bool HasConnection(Neuron* tgt) {
				if (tgt == Target) return true;
				return false;
			}
		};

		class Axon {
			std::vector<Telodendron> Telodendria;
			unsigned MyelinSheath;
			float Weight;
			std::vector<Signal> Signals;
			Neuron* Sender;
		public:
			Axon(std::vector<Telodendron>& tda, float wgt) : Telodendria(tda), Weight(wgt) { }
			Axon(Neuron* sender) : Weight(1), Sender(sender) { }
			Signal* SendSignal(float t) {
				Signals.emplace_back(t, Weight, this, (unsigned) Signals.size());
				return &Signals[Signals.size()-1];
			}
			void Connect(Neuron* tgt) {
				Telodendria.emplace_back(tgt, this, 1);
			}
			bool HasConnection(Neuron* tgt) {
				for (Telodendron& tdn : Telodendria) if (tdn.HasConnection(tgt)) return true;
				return false;
			}
		};

		class Neuron {
			float LeakRate;
			float ExcitationThreshold;
			float Excitation;
			float LastChange;
			float SignalInterval;
			float LastSignal;
			Axon AxonComponent;
		public:
			unsigned _INDEX;
			NeuronType Type;
			Neuron(unsigned index, NeuronType type) : LeakRate(0.1), ExcitationThreshold(1), Excitation(0), LastChange(0), SignalInterval(1), LastSignal(1e9), _INDEX(index), Type(type), AxonComponent(this) { }
			Signal* RecieveSignal(Signal& s, float t) {
				Excitation += s.Weight;
				Excitation -= (LastChange - t) * LeakRate;
				LastChange = -t;
				if (LastSignal - t >= SignalInterval) return AxonComponent.SendSignal(t);
				return nullptr;
			}
			void UpdateLastChange(float ts) {
				Excitation -= (LastChange - ts) * LeakRate;
				LastChange = ts-LastChange;
			}
			void Connect(Neuron* tgt) {
				AxonComponent.Connect(tgt);
			}
			bool HasConnection(Neuron* tgt) {
				return AxonComponent.HasConnection(tgt);
			}
		};

		class NeuralNetwork {
		public:
			std::vector<Neuron*> Input;
			std::vector<Neuron*> Hidden;
			std::vector<Neuron*> Output;
			NeuralNetwork(unsigned inputs, unsigned hidden, unsigned outputs, std::mt19937& gen) {
				for (int i = 0; i < inputs; ++i) Input.emplace_back(i, InputNeuron);
				for (int i = 0; i < hidden; ++i) Hidden.emplace_back(i, HiddenNeuron);
				for (int i = 0; i < outputs; ++i) Output.emplace_back(i, OutputNeuron);

				std::uniform_int_distribution<unsigned> iDist(0, inputs);
				std::uniform_int_distribution<unsigned> hDist(0, hidden);
				std::uniform_int_distribution<unsigned> oDist(0, outputs);

				for (unsigned i = 0; i < inputs; ++i) {
					Neuron* tgt = Hidden[hDist(gen)];
					Input[i]->Connect(tgt);
				}
				for (unsigned i = 0; i < inputs; ++i) {
					Neuron* snd = Input[iDist(gen)];
					Neuron* tgt = Hidden[hDist(gen)];
					if (!snd->HasConnection(tgt)) snd->Connect(tgt);
					else --i;
				}

				for (unsigned i = 0; i < 3*hidden; ++i) {
					Neuron* snd = Input[i/3];
					Neuron* tgt = Hidden[hDist(gen)];
					if (!snd->HasConnection(tgt) && !tgt->HasConnection(snd)) snd->Connect(tgt);
					else --i;
				}
				for (unsigned i = 0; i < 2*hidden; ++i) {
					Neuron* snd = Hidden[hDist(gen)];
					Neuron* tgt = Hidden[hDist(gen)];
					if (!snd->HasConnection(tgt) && !tgt->HasConnection(snd)) snd->Connect(tgt);
					else --i;
				}

				for (unsigned i = 0; i < 2*outputs; ++i) {
					Neuron* snd = Input[hDist(gen)];
					Neuron* tgt = Output[i/2];
					if (!snd->HasConnection(tgt)) snd->Connect(tgt);
					else --i;
				}
			}
		};

		inline Telodendron Telodendron::Copy(NeuralNetwork*& nn, Axon*& ax) {
			if (Target->Type == InputNeuron) return Telodendron(nn->Input[Target->_INDEX], ax, Weight);
			if (Target->Type == HiddenNeuron) return Telodendron(nn->Hidden[Target->_INDEX], ax, Weight);
			if (Target->Type == OutputNeuron) return Telodendron(nn->Output[Target->_INDEX], ax, Weight);
			throw new std::logic_error("\n\nTelodendria Target Neuron has no type. Must be either InputNeuron, HiddenNeuron, or OutputNeuron.\n");
		}
		inline Signal* Telodendron::SendSignal(Signal s, float t) {
			s.Weight *= Weight;
			return Target->RecieveSignal(s, t);
		}
	}
}

#endif // QED_IANN