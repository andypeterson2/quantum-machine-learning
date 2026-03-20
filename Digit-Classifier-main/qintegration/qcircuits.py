from typing import Callable

import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit import Parameter, ParameterVector
import numpy as np
from qintegration.qutils import QCSampler, QCExecutor



class ParametricCircuit():
    """
    Parameterized quantum circuits, aka quantum circuits with reassignable values. 

    Attributes
    ----------
    params : ParameterVector
        The vector containing the circuit parameters
    inputs : ParameterVector
        The vector containing the circuit inputs
    qc : QuantumCircuit
        The QuantumCircuit that represents this ParametricCircuit
    executor : QCExecutor
        The executor with which to execute the circuit

    Methods
    -------
    run(w : list[float], x : list[float]) -> np.ndarray:
        Runs the circuit using its executor
    """
    def __init__(
            self, 
            qc_builder: Callable[[ParameterVector, ParameterVector], QuantumCircuit], 
            num_params: int, 
            input_dim: int, 
            executor: QCExecutor
        ):
        """
        Default constructor for ParametricCircuit
    
        Parameters
        ----------
        qc_builder : (ParameterVector, ParameterVector) -> QuantumCircuit
            The 'parametric' quantum circuit builder that takes in the parameters and inputs to return the final circuit
        num_params : int
            The number of parameters
        input_dim : int
            The number of inputs
        executor : QCExecutor
            The executor with which to execute the circuit
        """
        self.params = ParameterVector('params', num_params)
        self.inputs = ParameterVector('inputs', input_dim)
        self.qc: QuantumCircuit = qc_builder(self.params, self.inputs)
        self.executor = executor

    def run(self, w: list[float], x: list[float]) -> np.ndarray:
        """
        Run the parametric circuit
    
        Parameters
        ----------
        w : list[float]
            The values of the parameters
        x : list[float]
            The values of the inputs
    
        Returns
        -------
        output : np.ndarray
            The result of running the ParametricCircuit with its executor
        """

        bound = self.qc.assign_parameters({self.params: w, self.inputs: x}, inplace=False, flat_input=False)
        output = self.executor.run(bound)
        return output

# Alternate implementation, same performance
# class ParametricCircuit():
#     def __init__(self, qc_builder, num_params, input_dim, executor: QCExecutor):
#         self.qc_builder = qc_builder
#         self.executor = executor

#         # For visualization only
#         self.qc = qc_builder(ParameterVector('inputs', input_dim), ParameterVector('params', num_params))

#     def run(self, w, x):
#         bound = self.qc_builder(x, w)
#         output = self.executor.run(bound)
#         return output

class ExampleCircuit(ParametricCircuit):
    """
    An example of a ParametricCircuit
    """
    def __init__(self, input_dim, executor = QCSampler()):
        super().__init__(ExampleCircuit.builder, 2 * input_dim, input_dim, executor)

    def builder(params, inputs):
        qc = QuantumCircuit(len(inputs), len(inputs))
        for i in range(len(inputs)):
            qc.rx(inputs[i], i)
        qc.barrier()

        # changeable
        for i in range(len(inputs)):
            qc.rxx(params[2 * i], i, (i+1) % len(inputs))
            qc.rzz(params[2 * i + 1], i, (i+1) % len(inputs))
            pass
        qc.barrier()

        qc.measure_all()

        return qc