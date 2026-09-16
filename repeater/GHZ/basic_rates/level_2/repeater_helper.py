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
    
def merge_state_GHZ(dm):
    input_state = kron_all(dm, dm, dm)

    output_state = proj_case_GHZ @ input_state @ proj_case_GHZ.T
    output_state = get_reduced_density_matrix(output_state, [0, 5, 7], 9)
    prob_suc = trace(output_state) * 8
    output_state= normalize(output_state)
    return  correction @ H3 @ output_state @ H3.T @ correction.T , prob_suc

def merge_state_W(dm):
    # We have to check the probabilities here.
    input_state = kron_all(dm, dm, dm)

    output_state_1 = proj_case1 @ input_state @ proj_case1.T
    output_state_1 = get_reduced_density_matrix(output_state_1, [0, 5, 7], 9)
    prob_suc_1 = trace(output_state_1) * 12
    output_state_1 = normalize(output_state_1)

    output_state_2 = pauli_correction @ proj_case2 @ input_state @ proj_case2.T @ pauli_correction.T
    output_state_2 = get_reduced_density_matrix(output_state_2, [0, 5, 7], 9)
    prob_suc_2= trace(output_state_2) * 12 
    output_state_2 = normalize(output_state_2)
    
    prob_suc_total = prob_suc_1 + prob_suc_2
    output_state = (prob_suc_1 * output_state_1 + prob_suc_2 * output_state_2) / prob_suc_total
    return output_state, prob_suc_total

def proj_W_to_bell(dm):
    # We post select the case where the the first qubit is projected to zero. Then the second and third qubits are projected to a \Psi^+ state. Note that we need to apply a bit flip to the second qubit.
    state = kron(identity, x_gate) @ (kron_all(bra_0, identity, identity)  @ dm @ kron_all(ket_0, identity, identity)) @ kron(identity, x_gate).T
    prob = trace(state)
    state = normalize(state)
    return  state, prob
    
def get_final_state_G_W_W(dm_GHZ, dm_W):

    dm_bell, prob_bell = proj_W_to_bell(dm_W)
    # qubit index [0, 1, 2, 3, 4, 5, 6]
    joint_state = kron_all(dm_GHZ, dm_bell, dm_bell)

    # Teleportation circuit

    # First CNOT acts on (1, 3)
    CNOT_1 = kron_all(identity, proj_0, identity, identity, identity, identity, identity) + kron_all(identity, proj_1, identity, x_gate, identity, identity, identity)

    # First H acts on (1)
    H_1 = kron_all(identity, h_gate, identity, identity, identity, identity, identity)
    
    # Second CNOT acts on (2, 5)
    CNOT_2 = kron_all(identity, identity, proj_0, identity, identity, identity, identity) + kron_all(identity, identity, proj_1, identity, identity, x_gate, identity)

    # First H acts on (2)
    H_2 = kron_all(identity, identity, h_gate, identity, identity, identity, identity)

    joint_state = H_1 @ CNOT_1 @ (H_2 @ CNOT_2 @ joint_state @ CNOT_2.T @ H_2.T) @ CNOT_1.T @ H_1.T

    # Apply teleportation, measure and apply the Pauli correction 

    # Stage one

    # Measure out (1, 3)
    bra_00 = kron_all(identity, bra_0, identity, bra_0, identity, identity, identity)
    ket_00 = bra_00.T

    
    joint_state_00 = bra_00 @ joint_state @ ket_00

    joint_state =   joint_state_00

    # Stage two

    # Measure out (1, 3)
    bra_00 = kron_all(identity, bra_0, identity, bra_0, identity)
    ket_00 = bra_00.T

    joint_state_00 = bra_00 @ joint_state @ ket_00

    joint_state =   joint_state_00 

    # For this to happen, the two W states have to be projected to the Bell state in correct orientation.
    return normalize(joint_state), prob_bell * prob_bell


def get_final_state_G_G_W(dm_GHZ, dm_W):

    dm_bell, prob_bell = proj_W_to_bell(dm_W)

    # 8 qubit state
    joint_state = kron_all(dm_GHZ, dm_GHZ, dm_bell)

    ket_case_00 = kron_all(identity, identity, ket_0, ket_0, identity, identity, identity, identity)
    ket_case_11 = kron_all(identity, identity, ket_1, ket_1, identity, identity, identity, identity)
    bra_case_00 = ket_case_00.T
    bra_case_11 = ket_case_11.T
    ket_bell = sqrt(1/2) * (ket_case_00 + ket_case_11)
    bra_bell = ket_bell.T
    joint_state = bra_bell @ joint_state @ ket_bell
    joint_state = normalize(joint_state)

    CNOT_op = kron_all(identity, proj_0, identity, identity, identity, identity) + kron_all(identity, proj_1, identity, identity, x_gate, identity)
    H_op = kron_all(identity, h_gate, identity, identity, identity, identity)
    joint_state = H_op @ CNOT_op @ joint_state @ CNOT_op.T @ H_op.T
    ket_case_00 = kron_all(identity, ket_0, identity, identity, ket_0, identity) 
    bra_case_00 = ket_case_00.T
    joint_state = bra_case_00 @ joint_state @ ket_case_00

    ket_case_plus = sqrt(1/2) * (kron_all(identity, ket_0, identity, identity) + kron_all(identity, ket_1, identity, identity))
    bra_case_plus = ket_case_plus.T

    joint_state = bra_case_plus @ joint_state @ ket_case_plus

    # For this to happen, the W state have to be projected to the Bell state in a correct (and predefined) orientation.
    return normalize(joint_state), prob_bell

def get_final_state_G_G_G(dm_GHZ):
    joint_state = kron_all(dm_GHZ, dm_GHZ, dm_GHZ)
    proj_joint_swap = get_bell_projector('phi+', 'r2', 'u3') @ get_bell_projector('phi+', 'u1', 'l2') @ get_bell_projector('phi+', 'r1', 'l3')
    joint_state = proj_joint_swap @ joint_state @ proj_joint_swap.T
    joint_state = get_reduced_density_matrix(joint_state, [0, 5, 7], 9)


    return normalize(joint_state)

identity = np.array([[1, 0], [0, 1]])
x_gate = np.array([[0, 1], [1, 0]])
z_gate = np.array([[1, 0], [0, -1]])
i_gate = np.array([[1, 0], [0, 1j]])
h_gate = sqrt(1/2) * np.array([[1, 1], [1, -1]])
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

proj_case1 = get_bell_projector('psi+', 'u1', 'l2') @ get_bell_projector('psi+', 'r1', 'l3') @ get_parity_projector('even', 'r2', 'u3') @ get_parity_projector('odd', 'u1', 'l2') @ get_parity_projector('odd', 'r1', 'l3')

proj_case2 = proj_00_fix @ get_bell_projector('phi+', 'u1', 'l2') @ get_bell_projector('psi+', 'r2', 'u3') @ get_parity_projector('odd', 'r2', 'u3') @ get_parity_projector('even', 'u1', 'l2') @ get_parity_projector('even', 'r1', 'l3')
pauli_correction = get_pauli_correction("X", 'l1')

proj_case_GHZ = get_bell_projector('phi+', 'r2', 'u3') @ get_bell_projector('phi+', 'u1', 'l2') @ get_bell_projector('phi+', 'r1', 'l3') @ get_parity_projector('even', 'r2', 'u3') @ get_parity_projector('even', 'u1', 'l2') @ get_parity_projector('even', 'r1', 'l3')

H = 1/(sqrt(2)) * np.array([[1, 1],[1, -1]])
H3 = kron_all(H, H, H)
correction = kron_all(identity, identity, z_gate)