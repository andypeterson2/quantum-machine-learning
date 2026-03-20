# Digit Classifier
 
Contains several neural network architectures (quantum, hybrid, and classical) that were evaluated on the MNIST dataset of handwritten digits.

## File Structure

### data

Where the raw datasets are downloaded and stored.

### preprocessing

Dataset-specific preprocessing methods and scripts to convert the raw datasets from the data folder into python objects.

### models

Python scripts containing different model architectures, all of which extend nn.Module.

### train

Python scripts containing methods for training, validation, and evaluation of a given model.

### qintegration

Python scripts that makes quantum circuits (with parameters) act like trainable neural network layers.

```qutils.py``` contains execution strategies for quantum circuits that generate ndarray outputs.

```qcircuits.py``` contains the ```ParametricCircuit``` class which formats a given quantum circuit as a function which take in a list of parameters and inputs to generate an ndarry output.

```qmodule.py``` defines gradient calculations for ```ParametricCircuit``` and formats it into a pytorch-compatible ```nn.Module```. (Can be easily refactored to work with any function that takes in a list of parameters and inputs to generate an ndarray)

## Architectures

### Quantum Layer (QLayer)

A pytorch-compatible neural network layer that executes an arbitrary parametric quantum circuit.

The parameters of the circuit are trainable, and can be optimized to minimize a given loss.

### Quadratic Layer (Quadratic)

Instead of learning the best weights to use for a linear combination of the inputs (as a Linear layer does), the Quadratic layer tries to learn the best weights to use for a "quadratic combination" of the inputs.

Essentially, wheras every feature in the Linear layer is of the form ```dot(x, w)```, every feature in the Quadratic layer is of the form ```dot(x.T @ x | x, W)```, or ```sum([x_i * x_j * W_ij for 1 ≤ i, j ≤ n])```, where n is the number of inputs. This implementation combines both the Quadratic layer with a Linear layer.

### Polynomial Layer (Polynomial)

Seeking to emulate the result of multiple Quadratic layers in series, the Polynomial layer consists of features that are linear combinations of some specied number of "polynomial combinations" of all the inputs.

Each feature is of the form ```sum([Wj0 * product([x_i ** W_ji for 1 ≤ i ≤ n]) for 1 ≤ j ≤ p)])```, where n is the number of inputs and p is the number of polynomial combinations.
