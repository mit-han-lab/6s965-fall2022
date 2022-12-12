# Lecture 23: Noise Robust Quantum ML (Part II)

## Note Information

| Title       | Noise Robust Quantum ML (Part II)                                                   |
|-------------|-----------------------------------------------------------------------------------------------------------------|
| Lecturer    | Hanrui Wang, Song Han                                                                                                        |
| Date        | 12/01/2022                                                                                                      |
| Note Author | Janice Yang (janicey)                                                                                                 |
| Description | Continue TorchQuantum Examples, Introduce Robust Quantum Circuit Architecture Search |


## Torch Quantum 

### TQ for Statevector Simulation
- We can use TQ to perform matrix-vector multiplcation between gate matrix and statevector
    - Method 1: `import torchquantum.functional as tqf`, then `tqf.h(q_dev, wires=1)`
    - Method 2: `h_gate = tq.H()`, then `h_gate(q_dev, wires=3)`
- The `tq.QuantumState` class can also store the statevectors
- There is also support for batch size changing and automatic gradient computation
- There is also support to convert torch quantum models to other frameworks, like `Qiskit`

### Benefits for using TQ
- Speedup over other frameworks such as Pennylane


## Robust Quantum Circuit Architecture Search

### Parameterized Quantum Circuits (PQC)
- PQC's are quantum circuits with fixed gates and parameterized gates 
- Challenges of PQCs: Noise.
    - Noise degrades PQC reliability
    - More parameters increase the noise-free accuracy (due to increased learning capacity) but degrade the measured accuracy
- Therefore, finding the best architecture is critical
- Anohter challenge: Large Design Space. Need to consider: 
    - Type of gates
    - Number of gates
    - Position of gates

### QuantumNAS Framework
- Automatically & Efficiently Search
    - Train one "SuperCircuit" providing parameters to many "SubCircuits"
- Need to find a noise-robust quantum circuit
    - Add quantum noise feedback in the search loop
    - Co-search the circuit architecture and qubit mapping
- Four steps: 
    - SuperCircuit Construction and Training
    - Noise-Adaptive Evolutionary Co-Search of Subcircuit 
    - Train the Searched SubCircuit
    - Iterative Quantum Gate Pruning


### SuperCircuit & SubCircuit
- Construct design space
- SuperCircuit: the circuit with the largest number of gates in the design space
- SubCircuit: each candidate circuit in the design space, a subset of the SuperCircuit
- Why use SuperCircuit?
    - Enables efficient search of architecture candidates without training each
    - SubCircuit inherits from SuperCircuit
    - SubCircuits that perform well with inherited paramters also perform well with parameters trained from scratch
- In one SuperCircuit Training step: 
    - Sample a gate subset of a SuperCircuit (a SubCircuit)
        - Can use either Front Sampling and Restricted Sampling
    - Only use the subset to perform the task and update the parameters in the subset
    - Paramter updates are cumulative across steps
- Front sampling: Only the front several blocks and front several gates can be sampled
- Restricted Sampling: restrict the difference between SubCircuits of two consecutive steps. 
    - For example: restrict to at most 4 different layers
- A subcircuit's inherited parameters from SuperCircuit can provide reliable estimates of accuracy


### Noise Adaptive Evolutionary Search
- Search the best SubCircuit and its qubit mapping on target device
- Mutation and crossover create new SubCircuit candidates

### Iterative Quantum Gate Pruning
- Some gates have parameters close to 0
    - Rotation gate with angle close to 0 has small impact on the results
- Iteratively prune small-magnitude gates and fine-tune the remaining parameters
    - This reduces quantum noise, so can increase accuracy on the real device


### Evaluation
- Benchmarks: 
    - QML Classification tasks: MNIST 10-class, 4-class, 2-class, Fashion 4-class, 2-class, Vowel 4-class
    - VQE task molecules: H2, H2O, LiH, CH4, BeH2
- Quantum Devices: 
    - IBMQ
    - Qubits: 5 to 65
    - Quantum Volume: 8 to 128

QML Results: 
- QuantumNAS search results delays the accuracy peak and enables more circuit parameters
- Consistent improvements on diverse design spaces, including U3+CU3, ZZ+RY, RXYZ, and ZX+XX Spaces
- Scalable to large number of qubits (15, 16, even up to 65)
- Quantum Gate Pruning improves accuracy by 3% on average


### Other Search Frameworks
- Can use graph transformer for circuit fidelity estimation, a faster method to evaluate real device performance
- Can also use RL for Architecture Search
- Quantum circuit architecture search for variational quantum algorithms

## Robust PQC Parameter Training

### Multi-Node QNN
- Each node contains an encoder, trainable layers, and measurement

### Post-Measurement Normalization 
- Used for error mitigation
- Normalize the measurement outcome along the batch dimension
- The outcome with normalization better matches the noise-free simulation 

### Noise Injection 
- Inject noise during training on a classical simulator
- Two errors to inject: Pauli error and readout error
- Readout error: 
    - Even if the state is 100% in a certain state, there is a chance that the readout value can be different
- Pauli error: 
    - Small probability that the device actually performs the the wrong gate
- We can obtain the probability matrix of these errors happening, and inject this noise during training

### Post-Measurement Quantization
- Quantize measurement outcomes 
    - Has a denoising effect
    - Small errors will be mitigated
    - Improves Signal to Noise ratio
- Can use a loss term to encourage measurement outcomes to be close to centroids

### Evaluation
- Tasks include MNIST-10, MNIST-4, MNIST-2, Fashion-MNIST-10, Fashion-MNIST-4, Fashion-MNIST-2, Vowel-4, Vowel-2
- Quantum Devices: 
    - IBMQ
    - Qubits number: 5 to 15
    - Quantum Volume: 8 to 12

Consistent Improvement on Various Benchmarks
- On IBM Santiago device
- Improvements after normalization, noise injection, and quantization

Visualization of Feature Space
- QuantumNAT stretches the dsitribution of features across the two feature dimensions for each qubit