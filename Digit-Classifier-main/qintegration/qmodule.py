from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch import vmap
import numpy as np
from qintegration.qcircuits import ParametricCircuit, ExampleCircuit

class run_circuit(Function):
    """
    A torch.autograd Function that calculates the forwards and backwards passes for a given ParametricCircuit. 
    Essentially lets torch recognize the ParametricCircuit as a differentiable function.
    """

    @staticmethod
    def forward(ctx, pc, w, x_batch):
        if not hasattr(ctx, 'pc'):
            ctx.pc = pc

        w_list = w.tolist()
        values = []
        for s in range(len(x_batch)):
            x_list = x_batch[s].tolist()
            values.append(ctx.pc.run(w_list, x_list))
        values = np.array(values)
        
        result = torch.tensor(values)
        ctx.save_for_backward(result, w, x_batch)
        return result
    
    def estimate_partial_derivative(f: Callable, v: torch.Tensor, pos: int, delta : float = 0.2) -> torch.Tensor:
        """
        Estimates the partial derivative of the function f with respect to the selected element of vector v.
        This is done by calculating the rate of change of function f from v[pos] - delta to v[pos] + delta.
    
        Parameters
        ----------
        f : Callable
            A function of vector v
        v : torch.Tensor
            The vector whose element is the variable of differentiation
        pos : int
            The index of the varible of differentiation in vector v
        delta : float
            The finite change with which to estimate the partial derivative.
    
        Returns
        -------
        df_dv : torch.Tensor
            The (estimated) partial derivative of function f with respect to the selected element of vector v
        """
        v_delta = F.one_hot(torch.tensor([pos]), num_classes=v.shape[-1]).flatten()

        v_plus = v + v_delta # Shift up by delta
        fv_plus = f(v_plus)
        
        v_minus = v - v_delta # Shift down by delta
        fv_minus = f(v_minus)

        df_dv = torch.tensor(fv_plus - fv_minus) / 2 / delta

        return df_dv
    
    @staticmethod
    def backward(ctx, grad_output):
        # Obtain paramaters 
        forward_tensor, w, x_batch = ctx.saved_tensors
        w_list = w.tolist()
        grad_output = grad_output[0]
        
        batch_df_dw, batch_df_dx = [], []
        for j in range(len(x_batch)):
            x = x_batch[j]
            x_list = x.tolist()

            # Calculate gradient (weights) for current input
            df_dw = []
            for k in range(w.shape[-1]):
                df_dw_k = run_circuit.estimate_partial_derivative(
                    f = lambda w : ctx.pc.run(w.tolist(), x_list), 
                    v = w, 
                    pos = k
                )
                df_dw.append(torch.dot(df_dw_k, grad_output))
            batch_df_dw.append(df_dw)

            # Calculate gradient (inputs) for current input
            df_dx = []
            for k in range(x.shape[-1]):
                df_dx_k = run_circuit.estimate_partial_derivative(
                    f = lambda x : ctx.pc.run(w_list, x.tolist()), 
                    v = x, 
                    pos = k
                )
                df_dx.append(torch.dot(df_dx_k, grad_output))
            batch_df_dx.append(df_dx)

        batch_df_dw = torch.tensor(batch_df_dw).sum(dim=0)
        batch_df_dx = torch.flip(torch.tensor(batch_df_dx), dims=[1])

        return None, batch_df_dw, batch_df_dx

class QLayer(nn.Module):
    """
    The nn.Module that acts as a multi-headed trainable ParametricCircuit. 
    Each head stores an instance of the ParametricCircuit with its own set of parameter values.
    The final output of the QLayer is a reduction of the concatenation of the outputs of all of its heads.

    Attributes
    ----------
    qc : ParametricCircuit
        The circuit that is being run / trained
    heads : nn.ModuleList[QLayer.Head]
        The list of heads that constitute the quantum layer.
    reduce : Callable
        The reduction function / model that reduces the concatenated outputs of all the heads into the final output
    """
    class Head(nn.Module):
        """
        The nn.Module that acts as a single-headed trainable ParametricCircuit.

        Attributes
        ----------
        qc : ParametricCircuit
            The circuit that is being run / trained
        w : nn.Parameter
            The parameters that are being trained to optimize the circuit
        """
        def __init__(self, qc: ParametricCircuit):
            """
            Default constructor for QLayer.Head
        
            Parameters
            ----------
            qc : ParametricCircuit
                The parametric quantum circuit to train / run
            """

            super().__init__()
            self.qc = qc
            self.w = nn.Parameter(torch.zeros(len(qc.params)))

        def forward(self, x):
            x = run_circuit.apply(self.qc, self.w, x)
            return x
    
    def __init__(self, num_heads, qc: ParametricCircuit, reduce: bool=True):
        """
        Default constructor for QLayer
    
        Parameters
        ----------
        num_heads : int
            The number of heads (independently trained ParametricCircuits) for this QLayer
        qc : ParametricCircuit
            The parametric quantum circuit for each head to train / run
        reduce : bool
            Whether to apply the default reduction strategy of using a nn.Linear or not to have any reduction
        """
        super().__init__()
        self.qc = qc
        self.heads = nn.ModuleList([QLayer.Head(qc) for _ in range(num_heads)])
        num_outputs = self.heads[0](torch.zeros((1, len(qc.inputs)))).shape[1]
        self.reduce = nn.Linear(num_heads * num_outputs, num_outputs) if reduce else (lambda x : x)

    def forward(self, x):
        x = torch.concat([head(x) for head in self.heads], dim=1)
        x = self.reduce(x)
        return x

class ExampleQLayer(QLayer):
    """
    An example of a QLayer that uses ExampleCircuit
    """
    def __init__(self, input_dim, num_heads=1):
        super().__init__(num_heads, ExampleCircuit(input_dim))