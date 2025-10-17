"""
nD_QHCS_random.py - Random number generation using nD-QHCS
Updated version for latest Qiskit

This module extends the nD_QHCS_run.py functionality to generate
random numbers suitable for AFQIRHSI encryption operations.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.circuit.library import CRZGate


def kinetic_layer(qc, qubits, phase):
    """Apply the kinetic energy operator to a set of qubits"""
    for q in qubits:
        qc.h(q)
        qc.rz(phase, q)
        qc.h(q)


def two_site_cphase(qc, ctrl, targ, phi):
    """Apply a controlled phase rotation between two qubits"""
    qc.append(CRZGate(phi), [ctrl, targ])


def create_qhcs_circuit(n, k, A, B, dt, trotter_steps=1):
    """
    Create a quantum circuit implementing the nD-QHCS Hamiltonian evolution
    
    Parameters:
    -----------
    n: int
        Dimension of the system
    k: int
        Number of qubits per dimension
    A, B: float
        Coupling constants (chaotic window parameters)
    dt: float
        Time step
    trotter_steps: int
        Number of Trotter steps for the evolution
        
    Returns:
    --------
    QuantumCircuit
        Circuit implementing the QHCS evolution
    """
    # Create quantum registers for each coordinate
    coords = [QuantumRegister(k, f"q{i}") for i in range(n)]
    creg = ClassicalRegister(n * k, "c")
    qc = QuantumCircuit(*coords, creg)
    
    # Calculate phases for the Hamiltonian terms
    phase_T = -dt / 2.0  # Kinetic term
    phi_A = -dt * A      # First potential term
    phi_B = dt * B       # Second potential term
    
    # Apply Hadamard gates to create |+...+> state
    for i in range(n):
        for j in range(k):
            qc.h(coords[i][j])
    
    # Apply Trotter steps
    for _ in range(trotter_steps):
        # Kinetic energy term: Σ p²/2
        for i in range(n):
            kinetic_layer(qc, coords[i], phase_T)
        
        # Potential energy terms: Σ (A q_{i+1}q_{i-1} - B q_{i-2}q_{i-1})
        for i in range(n):
            msq_ip1 = coords[(i + 1) % n][0]  # most-significant qubit, i+1
            msq_im1 = coords[(i - 1) % n][0]  # most-significant qubit, i-1
            msq_im2 = coords[(i - 2) % n][0]  # most-significant qubit, i-2
            
            # Apply controlled phase rotations
            two_site_cphase(qc, msq_ip1, msq_im1, phi_A)
            two_site_cphase(qc, msq_im2, msq_im1, phi_B)
    
    # Add measurements
    for i in range(n):
        for j in range(k):
            qc.measure(coords[i][j], creg[i * k + j])
    
    return qc


def generate_qhcs_random_numbers(n=4, k=1, A=40.0, B=30.0, dt=0.04, 
                               trotter_steps=3, shots=1000, seed=None):
    """
    Generate random numbers using the nD-QHCS model
    
    Parameters:
    -----------
    n, k: int
        Dimension parameters for QHCS
    A, B: float
        Coupling constants (chaotic window)
    dt: float
        Time step
    trotter_steps: int
        Number of Trotter steps
    shots: int
        Number of measurements
    seed: int or None
        Random seed for reproducibility
        
    Returns:
    --------
    ndarray
        Array of random numbers
    """
    # Create the QHCS circuit
    qc = create_qhcs_circuit(n, k, A, B, dt, trotter_steps)
    
    # Run the simulation using the updated Qiskit API
    simulator = Aer.get_backend('qasm_simulator')
    
    # Set up simulation options
    sim_options = {}
    if seed is not None:
        sim_options['seed_simulator'] = seed
    
    # Run the circuit
    job = simulator.run(qc, shots=shots, **sim_options)
    result = job.result()
    counts = result.get_counts()
    
    # Convert measurements to random numbers
    rand_list = []
    for bitstring, freq in counts.items():
        value = int(bitstring, 2)
        rand_list.extend([value] * freq)
    
    # Shuffle the random numbers
    np.random.shuffle(rand_list)
    
    return np.array(rand_list)


def generate_encryption_parameters(num_pixels, n=4, k=1, A=40.0, B=30.0, 
                                dt=0.04, trotter_steps=3, seed=None):
    """
    Generate random numbers for AFQIRHSI encryption
    
    Parameters:
    -----------
    num_pixels: int
        Number of pixels in the image
    n, k, A, B, dt, trotter_steps: 
        QHCS parameters
    seed: int or None
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing:
        - random_sub: Random numbers for substitution
        - random_perm: Random numbers for permutation
    """
    # Calculate required number of random numbers
    # For substitution: 4 * num_pixels (theta_x, theta_y, theta_z, theta_i)
    # For permutation: num_pixels
    required_numbers = 5 * num_pixels
    
    # Generate random numbers with QHCS
    random_numbers = generate_qhcs_random_numbers(
        n=n, k=k, A=A, B=B, dt=dt, 
        trotter_steps=trotter_steps, 
        shots=required_numbers,
        seed=seed
    )
    
    # Split the random numbers for different purposes
    return {
        'random_sub': random_numbers[:4*num_pixels],  # For substitution
        'random_perm': random_numbers[4*num_pixels:]  # For permutation
    }


if __name__ == '__main__':
    # Example usage
    image_size = (2, 2)  # 2x2 image
    num_pixels = image_size[0] * image_size[1]
    
    # Generate parameters
    params = generate_encryption_parameters(
        num_pixels=num_pixels,
        n=4, k=1, A=40.0, B=30.0, dt=0.04,
        trotter_steps=3,
        seed=42  # For reproducibility
    )
    
    # Print the parameters
    print(f"Generated {len(params['random_sub'])} random numbers for substitution")
    print(f"First 5 values: {params['random_sub'][:5]}")
    
    print(f"\nGenerated {len(params['random_perm'])} random numbers for permutation")
    print(f"Values: {params['random_perm']}")