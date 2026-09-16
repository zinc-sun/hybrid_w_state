import numpy as np
from numpy import kron, trace, sqrt, allclose
from functools import reduce
import dill
import sympy as sp

def kron_all(*ops):
    return reduce(kron, ops)

def normalize(dm):
    return dm / trace(dm)

def get_reduced_density_matrix(rho, keep_qubits, n_qubits=None):
    """
    Computes the reduced density matrix by tracing out all qubits *not* in keep_qubits.
    
    Args:
        rho (np.ndarray): The input density matrix of shape (2**n, 2**n).
        keep_qubits (list): Indices of the qubits to keep (0-based).
        n_qubits (int, optional): Total number of qubits. If None, inferred from rho.
        
    Returns:
        np.ndarray: The reduced density matrix of shape (2**k, 2**k) where k = len(keep_qubits).
    """
    rho = np.asarray(rho)
    
    # Infer system size if not provided
    if n_qubits is None:
        n_qubits = int(np.log2(rho.shape[0]))
    
    # Validation
    if rho.shape != (2**n_qubits, 2**n_qubits):
        raise ValueError(f"Shape {rho.shape} is inconsistent with {n_qubits} qubits.")

    # 1. Identify which qubits to trace out
    all_qubits = set(range(n_qubits))
    keep_set = set(keep_qubits)
    trace_qubits = sorted(list(all_qubits - keep_set))
    sorted_keep = sorted(list(keep_set)) # Ensure indices are sorted for the output basis

    # 2. Reshape rho into a tensor with 2N axes: (2, 2, ..., 2)
    # Axes 0 to N-1 correspond to 'rows' (kets), N to 2N-1 correspond to 'cols' (bras).
    tensor_shape = [2] * (2 * n_qubits)
    rho_tensor = rho.reshape(tensor_shape)

    # 3. Permute axes to group 'keep' indices and 'trace' indices
    # We want the final structure: [Keep_Rows, Keep_Cols, Trace_Rows, Trace_Cols]
    # Row index i corresponds to axis i. Col index i corresponds to axis i + n_qubits.
    
    perm = (
        sorted_keep +                           # Keep Rows
        [q + n_qubits for q in sorted_keep] +   # Keep Cols
        trace_qubits +                          # Trace Rows
        [q + n_qubits for q in trace_qubits]    # Trace Cols
    )
    
    rho_permuted = np.transpose(rho_tensor, perm)

    # 4. Reshape to separate the 'keep' part from the 'trace' part
    # Dimension of kept system: 2^k
    # Dimension of traced system: 2^(N-k)
    dim_keep = 2 ** len(sorted_keep)
    dim_trace = 2 ** len(trace_qubits)
    
    # Shape becomes (dim_keep, dim_keep, dim_trace, dim_trace)
    rho_grouped = rho_permuted.reshape((dim_keep, dim_keep, dim_trace, dim_trace))

    # 5. Perform the partial trace by summing diagonal elements of the trace subsystem
    # We contract the last two axes (axis 2 and axis 3).
    rho_reduced = np.trace(rho_grouped, axis1=2, axis2=3)

    return rho_reduced

identity = np.array([[1, 0], [0, 1]])
x_gate = np.array([[0, 1], [1, 0]])
z_gate = np.array([[1, 0], [0, -1]])
i_gate = np.array([[1, 0], [0, 1j]])
proj_0 = np.array([[1, 0], [0, 0]])
proj_1 = np.array([[0, 0], [0, 1]])

ket_0 = np.array([[1],[0]])
bra_0 = ket_0.T
ket_1 = np.array([[0],[1]])
bra_1 = ket_1.T

kets = [ket_0, ket_1]
bras = [bra_0, bra_1]

qubit_positions = ['l1', 'r1', 'u1', 'l2', 'r2', 'u2', 'l3', 'r3', 'u3']
state_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]

qubit_mapping = dict(zip(qubit_positions, state_indices))

ket_00_fix = kron_all(identity, ket_0, identity, identity, identity, identity, ket_0, identity, identity)
bra_00_fix = ket_00_fix.T
proj_00_fix = ket_00_fix @ bra_00_fix

def get_extended_ket(state_str, pos_1, pos_2):
    '''
    state_str: '00', '01', '10', '11'
    idx_0 : 'l1', 'l2'...
    idx_1 : 'l1', 'l2'...
    '''

    state_1 = state_str[0]
    state_2 = state_str[1]

    idx_1 = qubit_mapping[pos_1]
    idx_2 = qubit_mapping[pos_2]

    result_lst = [identity]*9

    result_lst[idx_1] = kets[int(state_1)]
    result_lst[idx_2] = kets[int(state_2)]

    return kron_all(*result_lst)

def get_parity_projector(parity, pos_1, pos_2):
    if parity == 'odd':
        ket_01 = get_extended_ket('01', pos_1, pos_2)
        ket_10 = get_extended_ket('10', pos_1, pos_2)
        bra_01 = ket_01.T
        bra_10 = ket_10.T

        return ket_01 @ bra_01 + ket_10 @ bra_10
    
    elif parity == 'even':
        ket_00 = get_extended_ket('00', pos_1, pos_2)
        ket_11 = get_extended_ket('11', pos_1, pos_2)
        bra_00 = ket_00.T
        bra_11 = ket_11.T

        return ket_00 @ bra_00 + ket_11 @ bra_11
    else:
        raise Exception('parity not specified')
    
def get_bell_projector(bell, pos_1, pos_2):
    if bell == 'phi+':
        ket_00 = get_extended_ket('00', pos_1, pos_2)
        ket_11 = get_extended_ket('11', pos_1, pos_2)
        bra_00 = ket_00.T
        bra_11 = ket_11.T
        return (1/2) * (ket_00 + ket_11) @ (bra_00 + bra_11)
    elif bell == 'psi+':
        ket_01 = get_extended_ket('01', pos_1, pos_2)
        ket_10 = get_extended_ket('10', pos_1, pos_2)
        bra_01 = ket_01.T
        bra_10 = ket_10.T
        return (1/2) * (ket_01 + ket_10) @ (bra_01 + bra_10)
    elif bell == 'psi-':
        ket_01 = get_extended_ket('01', pos_1, pos_2)
        ket_10 = get_extended_ket('10', pos_1, pos_2)
        bra_01 = ket_01.T
        bra_10 = ket_10.T
        return (1/2) * (ket_01 - ket_10) @ (bra_01 - bra_10)
    elif bell == 'phi-':
        ket_00 = get_extended_ket('00', pos_1, pos_2)
        ket_11 = get_extended_ket('11', pos_1, pos_2)
        bra_00 = ket_00.T
        bra_11 = ket_11.T
        return (1/2) * (ket_00 - ket_11) @ (bra_00 - bra_11)
    else:
        raise Exception("Bell state not specified")
    
def get_pauli_correction(pauli, pos):
    result_lst = [identity]*9
    idx = qubit_mapping[pos]
    if pauli == "X":
        result_lst[idx] = x_gate
        return kron_all(*result_lst)
    elif pauli == "Y":
        result_lst[idx] = x_gate @ z_gate
        return kron_all(*result_lst)
    elif pauli == "Z":
        result_lst[idx] = z_gate
        return kron_all(*result_lst)
    elif pauli == "i":
        result_lst[idx] = i_gate
        return kron_all(*result_lst)
    elif pauli == "I":
        return kron_all(*result_lst)
    else:
        raise Exception("Pauli not correct")
    
def merge_state(dm):
    input_state = kron_all(dm, dm, dm)

    output_state = proj_case_GHZ @ input_state @ proj_case_GHZ.T
    output_state = get_reduced_density_matrix(output_state, [0, 5, 7], 9)
    prob_suc = trace(output_state) * 8
    output_state= normalize(output_state)
    return  correction @ H3 @ output_state @ H3.T @ correction.T , prob_suc

proj_case_GHZ = get_bell_projector('phi+', 'r2', 'u3') @ get_bell_projector('phi+', 'u1', 'l2') @ get_bell_projector('phi+', 'r1', 'l3') @ get_parity_projector('even', 'r2', 'u3') @ get_parity_projector('even', 'u1', 'l2') @ get_parity_projector('even', 'r1', 'l3')

H = 1/(sqrt(2)) * np.array([[1, 1],[1, -1]])
H3 = kron_all(H, H, H)
correction = kron_all(identity, identity, z_gate)