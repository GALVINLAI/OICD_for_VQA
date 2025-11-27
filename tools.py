import numpy as np

def random_hermitian(p):
    M = np.random.randn(p, p) + 1j * np.random.randn(p, p)
    return (M + M.conj().T) / 2

def random_skew_hermitian(p):
    M = np.random.randn(p, p) + 1j * np.random.randn(p, p)
    return (M - M.conj().T) / 2

def random_complex(n):
    return np.random.randn(n, n) + 1j * np.random.randn(n, n)

def comm(X, Y):
    return X.dot(Y) - Y.dot(X)

def inner_product(A, B):
    return np.trace(A.conj().T @ B)

def real_inner_product(A, B):
    return np.trace(A.conj().T @ B).real

def uniform_state(n):
    """Return the uniform superposition |ψ⟩ = (1/√2^n) ∑_x |x⟩ for n qubits."""
    p = 2**n
    psi = np.ones(p, dtype=complex) / np.sqrt(p)
    return psi
def random_unitary(n: int) -> np.ndarray:
    """
    Generate a random n x n unitary matrix distributed according to Haar measure.

    Args:
        n: Dimension of the unitary matrix.

    Returns:
        A random unitary matrix U of shape (n, n).
    """
    # Create a random complex matrix with i.i.d. standard normal entries
    A = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2)

    # Perform QR decomposition
    Q, R = np.linalg.qr(A)

    # Make diagonal of R have unit magnitude
    # Compute phases
    diag_R = np.diag(R)
    lambda_phases = diag_R / np.abs(diag_R)

    # Form the phase-correction diagonal matrix
    D = np.diag(lambda_phases.conj())

    # Return the Haar-distributed unitary
    return Q @ D

def Hess(Omega, H, rho):
    return comm(H, comm(Omega, rho)) + 0.5 * comm(comm(H, rho), Omega)

def L(Omega, H, rho):
    return comm(H, comm(Omega, rho)) + comm(comm(H, Omega), rho)

def is_hermitian(A, tol=1e-10):
    result = np.allclose(A, A.conj().T, atol=tol)
    if result:
        print("✅ The matrix is Hermitian")
    else:
        print("❌ The matrix is NOT Hermitian")
    return result

def is_skew_hermitian(A, tol=1e-10):
    result = np.allclose(A, -A.conj().T, atol=tol)
    if result:
        print("✅ The matrix is Skew-Hermitian")
    else:
        print("❌ The matrix is NOT Skew-Hermitian")
    return result

def is_symmetric(A, tol=1e-10):
    result = np.allclose(A, A.T, atol=tol)
    print("✅ Symmetric" if result else "❌ Not Symmetric")
    return result

def is_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    if U.shape[0] != U.shape[1]:
        print("❌ Not Unitary (not square)")
        return False

    I = np.eye(U.shape[0], dtype=U.dtype)
    # Check U†U ≈ I and UU† ≈ I
    left = U.conj().T @ U
    right = U @ U.conj().T
    cond1 = np.allclose(left, I, atol=tol)
    cond2 = np.allclose(right, I, atol=tol)

    if cond1 and cond2:
        print("✅ Unitary")
        return True
    else:
        print("❌ Not Unitary")
        return False
def matrix_rank_info(A, tol=1e-10):
    r = np.linalg.matrix_rank(A, tol=tol)
    full = r == A.shape[0]
    print(
        f"Matrix rank: {r} / {A.shape[0]} "
        f"{'✅ Full rank matrix' if full else '❌ Not full rank'}"
    )
    return r, full

def is_psd(A, tol=1e-10, verbose=False):
    """
    Check whether a Hermitian matrix A is:
      - Positive definite (PD): all eigenvalues > 0
      - Positive semi-definite (PSD): all eigenvalues >= 0
    Prints the classification and returns a string label.
    """
    eigvals = np.linalg.eigvalsh(A)

    print("Eigenvalues:", eigvals) if verbose else None

    if np.all(eigvals > tol):
        print("✅ Matrix is Positive Definite (PD): all eigenvalues > 0")
        return "PD"
    elif np.all(eigvals >= -tol):
        print("✅ Matrix is Positive Semi-Definite (PSD): all eigenvalues ≥ 0")
        return "PSD"
    else:
        print("❌ Matrix is NOT Positive Semi-Definite")
        return "Not PSD"


import numpy as np

def is_periodic(f, T, domain=(0, None), num_samples=100, tol=1e-6):
    """
    Test whether f(x + T) == f(x) (within tolerance) on a sampling of the domain.

    Parameters
    ----------
    f : callable
        A real-valued function of one variable, f(x).
    T : float
        The candidate period.
    domain : tuple (a, b) or (a, None)
        Interval on which to test periodicity. If b is None, tests on [a, a+T].
    num_samples : int
        Number of points to sample in the interval.
    tol : float
        Tolerance for |f(x+T) - f(x)|.

    Returns
    -------
    bool
        True if |f(x+T) - f(x)| <= tol at all sampled points; False otherwise.
    """
    a, b = domain
    if b is None:
        b = a + T

    # sample grid
    xs = np.linspace(a, b, num_samples, endpoint=False)
    # compute difference
    diffs = np.abs(np.array([f(x + T) - f(x) for x in xs]))
    return np.all(diffs <= tol)

