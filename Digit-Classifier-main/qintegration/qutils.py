from typing import Optional

from abc import ABC, abstractmethod

import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import numpy as np



class QCExecutor(ABC):
    """
    The executor that runs a quantum circuit.

    Methods
    -------
    run(qc: QuantumCircuit) -> np.ndarray:
        Run the quantum circuit.
    """
    @abstractmethod
    def run(self, qc: QuantumCircuit) -> np.ndarray:
        """
        Run the quantum circuit
    
        Parameters
        ----------
        qc : QuantumCircuit
            The quantum circuit that is being run
    
        Returns
        -------
        output : np.ndarray
            The result of running the quantum circuit
        """
        pass



class QCSampler(QCExecutor):
    """
    A QCExecutor that runs a quantum circuit by sampling it many times.

    Attributes
    ----------
    backend: Never
        The backend that runs the quantum circuit
    interpret: Interpret
        The interpreter that takes the outputs of the samples and returns the final output
    shots: int
        The number of times to samples the quantum circuit for each run

    Methods
    -------
    run(qc: QuantumCircuit, shots = None) -> np.ndarray:
        Sample the quantum circuit a number of times, then return the interpreted results. Unless explicitly stated, 
        use the number of shots given when the sampler was initialized.
    """
    class Interpret(ABC):
        """
        The interpretor for QCSampler. Takes the outputs of the samples and returns the final output
        """
        def __call__(self, counts: dict[str, int]) -> np.ndarray:
            pass

    class IndependentInterpret(Interpret):
        """
        An interpretor for QCSampler that returns the mean of each qubit independently.
        """
        def __call__(self, counts: dict[str, int]) -> np.ndarray:
            output_dim = len(next(iter(counts)).split(' ')[0])

            output = np.zeros(output_dim, dtype=np.float32)
            for outcome in counts:
                # np.fromstring(outcome, dtype=np.int8, sep='')
                for bit in range(output_dim):
                    if outcome[bit] == '1':
                        output[bit] += counts[outcome]

            return output / output.sum()
        


    def __init__(self, interpret: Interpret = IndependentInterpret(), shots : int = 2**13) -> None:
        """
        Default constructor for QCSampler
    
        Parameters
        ----------
        interpret : QCSampler.Interpret
            The function to use to interpret the results of the shots
        shots : int
            The number of times to run the circuit per execution
        """
        super().__init__()
        self.backend = Aer.get_backend('qasm_simulator')
        self.interpret = interpret
        self.shots = shots

    def run(self, qc : QuantumCircuit, shots : Optional[int] = None) -> np.ndarray:
        """
        Run the quantum circuit by sampling shots times then interpreting the results.
    
        Parameters
        ----------
        qc : QuantumCircuit
            The quantum circuit that is being run.
        shots : int
            The number of times to sample. If None, use the number of shots given when instantiating the sampler.
    
        Returns
        -------
        output : np.ndarray
            The interpreted results of sampling the quantum circuit shots times.
        """
        counts = self.sample(qc, shots or self.shots)
        return self.interpret(counts)

    def sample(self, qc: QuantumCircuit, shots: int) -> dict[str, int]:
        """
        Sample the quantum circuit shots times
    
        Parameters
        ----------
        qc : QuantumCircuit
            The quantum circuit that is being run.
        shots : int
            The number of times to sample.
    
        Returns
        -------
        output : dict[str, int]
            A mapping of each encountered outcome to its frequency.
        """
        new_circuit = transpile(qc, self.backend)
        job_sim = self.backend.run(new_circuit, shots=shots)
        result_sim = job_sim.result()
        counts = result_sim.get_counts()

        return counts
