"""
Efficient Synthesis of Linear Reversible Circuits.

Implementation of the algorithm from:
    K. N. Patel, I. L. Markov, J. P. Hayes,
    "Efficient Synthesis of Linear Reversible Circuits" (quant-ph/0302002).

A linear reversible circuit on n wires is represented by a non-singular n x n
matrix A over GF(2).  Synthesizing the circuit is equivalent to reducing A to
the identity using elementary row operations, each of which corresponds to a
C-NOT gate.  A row operation "add row c to row t" is the gate CNOT(control=c,
target=t).

Two synthesis routines are provided:

  * gaussian_synth   -- the standard O(n^2) baseline (plain GF(2) elimination)
  * patel_markov_synth -- the O(n^2 / log n) algorithm of the paper

Both return a list of (control, target) tuples.  Applying those gates in order
to the identity reproduces A, which is verified by `apply_circuit`.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

Gate = Tuple[int, int]  # (control, target) of a C-NOT


# --------------------------------------------------------------------------- #
# GF(2) helpers
# --------------------------------------------------------------------------- #
def random_invertible_gf2(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly-ish random non-singular n x n matrix over GF(2)."""
    while True:
        A = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
        if is_invertible_gf2(A):
            return A


def is_invertible_gf2(A: np.ndarray) -> bool:
    """Check non-singularity over GF(2) by attempting row reduction."""
    M = A.copy().astype(np.uint8)
    n = M.shape[0]
    for col in range(n):
        pivot = -1
        for row in range(col, n):
            if M[row, col]:
                pivot = row
                break
        if pivot == -1:
            return False
        if pivot != col:
            M[[col, pivot]] = M[[pivot, col]]
        for row in range(n):
            if row != col and M[row, col]:
                M[row] ^= M[col]
    return True


def apply_circuit(n: int, gates: List[Gate]) -> np.ndarray:
    """
    Build the matrix produced by applying `gates` (in order) to the identity.

    Each gate CNOT(control, target) performs the row operation
    row[target] += row[control]  (mod 2), i.e. left-multiplication by the
    corresponding elementary matrix.
    """
    M = np.eye(n, dtype=np.uint8)
    for control, target in gates:
        M[target] ^= M[control]
    return M


def verify(A: np.ndarray, gates: List[Gate]) -> bool:
    """Check that applying `gates` to the identity reproduces A (over GF(2))."""
    n = A.shape[0]
    return np.array_equal(apply_circuit(n, gates) % 2, A % 2)


# --------------------------------------------------------------------------- #
# Baseline: standard Gaussian elimination over GF(2)
# --------------------------------------------------------------------------- #
def gaussian_synth(A_in: np.ndarray) -> List[Gate]:
    """
    Synthesize A using plain Gaussian elimination over GF(2).

    We reduce A to the identity with row operations.  A row op
    "row[i] += row[j]" applied to the *working matrix* corresponds, in the
    circuit acting on the input vector, to the C-NOT that adds row j into
    row i.  We record the operations that reduce A and reverse-account for
    them so that `verify` passes.

    Strategy:  We compute the sequence of elementary operations E_k ... E_1
    with E_k...E_1 A = I.  Then A = E_1^{-1} ... E_k^{-1}.  Over GF(2) an
    elementary "add row j to row i" matrix is its own inverse, so
    A = (E_1)(E_2)...(E_k) read as gates in *reverse* order, each being the
    same add-operation.  Concretely we collect ops while reducing and emit
    them reversed.
    """
    A = A_in.copy().astype(np.uint8)
    n = A.shape[0]
    ops: List[Gate] = []  # row-reduction operations (control -> target)

    def add_row(src: int, dst: int) -> None:
        A[dst] ^= A[src]
        ops.append((src, dst))

    for col in range(n):
        # find / create a pivot at (col, col)
        if A[col, col] == 0:
            for row in range(col + 1, n):
                if A[row, col]:
                    add_row(row, col)
                    break
        # eliminate every other 1 in this column
        for row in range(n):
            if row != col and A[row, col]:
                add_row(col, row)

    # A is now the identity.  ops reduced A to I, so reversing them
    # (each op is self-inverse over GF(2)) synthesizes A from I.
    return list(reversed(ops))


# --------------------------------------------------------------------------- #
# Patel-Markov-Hayes algorithm
# --------------------------------------------------------------------------- #
def _lwr_cnot_synth(A: np.ndarray, n: int, m: int) -> List[Gate]:
    """
    Reduce the lower-triangular part of A to the identity (Algorithm 1,
    Lwr_CNOT_Synth).  Operates in place on A and returns the row operations
    performed, in the order they were applied.
    """
    circuit: List[Gate] = []
    num_sections = math.ceil(n / m)

    for sec in range(num_sections):
        col_start = sec * m
        col_end = min((sec + 1) * m, n)  # exclusive

        # ---- Step A: remove duplicate sub-rows in this column section ----
        patt: dict = {}  # sub-row pattern -> first row index seen
        for row_ind in range(col_start, n):
            sub = tuple(int(x) for x in A[row_ind, col_start:col_end])
            if all(v == 0 for v in sub):
                continue  # all-zero sub-rows are never "duplicated away"
            if sub not in patt:
                patt[sub] = row_ind
            else:
                src = patt[sub]
                A[row_ind] ^= A[src]
                circuit.append((src, row_ind))  # Step A C-NOT

        # ---- Gaussian elimination on the remaining entries in section ----
        for col_ind in range(col_start, col_end):
            # Step B: ensure a 1 on the diagonal
            diag_one = A[col_ind, col_ind] == 1
            for row_ind in range(col_ind + 1, n):
                if A[row_ind, col_ind] == 1:
                    if not diag_one:
                        A[col_ind] ^= A[row_ind]
                        circuit.append((row_ind, col_ind))  # Step B C-NOT
                        diag_one = True
                    # Step C: clear the 1 below the diagonal
                    A[row_ind] ^= A[col_ind]
                    circuit.append((col_ind, row_ind))  # Step C C-NOT

    return circuit


def patel_markov_synth(A_in: np.ndarray, m: int | None = None) -> List[Gate]:
    """
    Synthesize A using the Patel-Markov-Hayes algorithm (Algorithm 1).

    Returns a list of C-NOT gates (control, target) that, applied in order to
    the identity, reproduce A over GF(2).

    `m` is the column-section width.  If None, the paper's heuristic
    m = round((log2 n) / 2) is used (clamped to at least 1).
    """
    A = A_in.copy().astype(np.uint8)
    n = A.shape[0]
    if m is None:
        m = max(1, round(math.log2(n) / 2)) if n > 1 else 1

    # 1) reduce lower-triangular part.  Applying circuit_l (in order) as
    #    left row operations turns A into an upper-triangular matrix U:
    #        L_op @ A = U,   L_op = product of the circuit_l elementaries.
    circuit_l = _lwr_cnot_synth(A, n, m)  # A is now upper triangular (== U)

    # 2) transpose and reduce again.  This reduces U^T to the identity:
    #        Uu_op @ U^T = I.
    A = A.T.copy()
    circuit_u = _lwr_cnot_synth(A, n, m)

    # 3) Combine the two stages.  This mirrors the paper's
    #        circuit = [reverse(circuit_u) | circuit_l]
    #    with the control/target of the upper-stage gates switched (because
    #    that stage operated on the transpose).  Concretely the synthesis is
    #    the upper-stage gates (control/target switched, original order) placed
    #    before the lower-stage gates in reverse order.
    upper_part = [(t, c) for (c, t) in circuit_u]
    lower_part = list(reversed(circuit_l))
    synthesis = upper_part + lower_part
    return synthesis


# --------------------------------------------------------------------------- #
# Self-test when run directly
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("Verifying correctness on random matrices...")
    for n in range(2, 40):
        for _ in range(20):
            A = random_invertible_gf2(n, rng)

            g_gauss = gaussian_synth(A)
            assert verify(A, g_gauss), f"Gaussian failed at n={n}"

            g_pmh = patel_markov_synth(A)
            assert verify(A, g_pmh), f"Patel-Markov failed at n={n}"
    print("All correctness checks passed.")