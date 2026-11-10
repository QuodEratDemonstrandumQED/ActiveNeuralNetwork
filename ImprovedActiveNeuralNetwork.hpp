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

		/**
		 * @brief The various types of neurons: Input, Hidden, and Output.
		 */
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

			/**
			 * @brief Creates a copy of the Telodendron.
			 *
			 * @param nn The Neural Network that the Telodendron is a part of.
			 * @param ax The Axon that is the parent of the Telodendron.
			 * @return The new Telodendron that was created from the Telodendron.
			 */
			Telodendron Copy(NeuralNetwork*& nn, Axon*& ax);

			/**
			 * @brief Sends a Signal to the Neuron that the Telodendron connects to.
			 *
			 * @param s The Signal being sent.
			 * @param t The time that the signal has overtraveled for.
			 * @return The Signal being sent.
			 */
			Signal* SendSignal(Signal s, float t);

			/**
			 * @brief 
			 *
			 * @param tgt The Neuron to connect to.
			 * @return If there's already a connection between the two Neurons, then true, otherwise false.
			 */
			bool HasConnection(Neuron* tgt) {
				if (tgt == Target) return true;
				return false;
			}
		};

		/**
		 * @brief The tail of a Neuron that sends Signals.
		 */
		class Axon {
			std::vector<Telodendron> Telodendria;
			unsigned MyelinSheath;
			float Weight;
			std::vector<Signal> Signals;
			Neuron* Sender;
		public:
			Axon(std::vector<Telodendron>& tda, float wgt, Neuron* sender) : Telodendria(tda), Weight(wgt), Sender(sender) { }
			Axon(Neuron* sender) : Weight(1), Sender(sender) { }

			/**
			 * @brief Sends a Signal down the Axon.
			 *
			 * @param t The time that the Signal has already traveled for.
			 * @return The Signal being sent.
			 */
			Signal* SendSignal(float t) {
				Signals.emplace_back(t, Weight, this, (unsigned) Signals.size());
				return &Signals[Signals.size()-1];
			}

			/**
			 * @brief Connect the Axon to a Neuron.
			 *
			 * @param tgt The target Neuron to connect to.
			 */
			void Connect(Neuron* tgt) {
				Telodendria.emplace_back(tgt, this, 1);
			}

			/**
			 * @brief Check if there's already a connection between two Neurons before trying to establish a new one.
			 *
			 * @param tgt The Neuron that the Axon's parent is trying to connect to.
			 * @return If there's already a connection between the two Neurons, then true, otherwise false.
			 */
			bool HasConnection(Neuron* tgt) {
				for (Telodendron& tdn : Telodendria) if (tdn.HasConnection(tgt)) return true;
				return false;
			}
		};

		/**
		 * @brief A Neuron, the basis of a Neural Network.
		 */
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

			/**
			 * @brief Makes the Neuron receive a Signal.
			 *
			 * @param s The Signal being received.
			 * @param t The time during which the Signal was received.
			 * @return The Signal emitted to the Axon to keep track of.
			 */
			Signal* RecieveSignal(Signal& s, float t) {
				Excitation += s.Weight;
				Excitation -= (LastChange - t) * LeakRate;
				LastChange = -t;
				if (LastSignal - t >= SignalInterval) return AxonComponent.SendSignal(t);
				return nullptr;
			}

			/**
			 * @brief Updates the last change of the Neuron.
			 *
			 * @param ts The timestep.
			 */
			void UpdateLastChange(float ts) {
				Excitation -= (LastChange - ts) * LeakRate;
				LastChange = ts-LastChange;
			}

			/**
			 * @brief Connects one Neuron to another by adding another Telodendron to the Axon.
			 *
			 * @param tgt The target Neuron to connect to.
			 */
			void Connect(Neuron* tgt) {
				AxonComponent.Connect(tgt);
			}

			/**
			 * @brief Check if there's already a connection between two Neurons before trying to establish a new one.
			 *
			 * @param tgt The Neuron trying to connect to this Neuron.
			 * @return If there's already a connection between the two Neurons, then true, otherwise false.
			 */
			bool HasConnection(Neuron* tgt) {
				return AxonComponent.HasConnection(tgt);
			}
		};

		class NeuralNetwork {
		public:
			std::vector<Neuron*> Input;
			std::vector<Neuron*> Hidden;
			std::vector<Neuron*> Output;

			/**
			 * @param inputs The number of inputs, as an unsigned integer.
			 * @param hidden The number of hidden Neurons, as an unsigned integer.
			 * @param outputs The number of outputs, as an unsigned integer.
			 * @param gen The Mersenne Twister 19937 Pseudo-Random Number Generator used to randomise the inter-Neuronal connections and weights of the Neural Network.
			 */
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