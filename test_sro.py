"""Checks for the neighbour shells and the Warren Cowley objective.

Run from the repo root with the root on the path:
    PYTHONPATH=. python diagnostics/test_sro.py

It imports the real functions from sro.py and the real lattice from
sqs_builder.py, so it tests the code that actually runs.
"""

import numpy as np

from sro import neighbour_shells, objective
from sqs_builder import make_sites, cell_vectors


def shell_sizes(shell, n_sites):
    """How many neighbours each site has in this shell.

    The shell is stored as flat pair arrays, so count how many times each
    site appears as a centre.
    """
    centre, neighbour = shell
    return set(np.bincount(centre, minlength=n_sites))


def test_neighbour_counts():
    """Each shell must hold the textbook number of neighbours."""
    print("Test 1: neighbour counts")
    for lattice, expected in [("bcc", [8, 6, 12, 24]),
                              ("fcc", [12, 6, 24, 12])]:
        frac = make_sites(lattice, (3, 3, 3))
        cell = cell_vectors(3.0, (3, 3, 3))
        shells, used = neighbour_shells(frac, cell, 3.0, lattice)
        print("  ", lattice, "sites:", len(frac), "shells used:", used)
        for shell, number in zip(shells, used):
            counts = shell_sizes(shell, len(frac))
            print("      shell", number, "count", counts,
                  "expected", expected[number - 1])
    print()


def test_b2_ordered():
    """
    B2 ordering on bcc. Species A on the corner sublattice, B on the body
    centre. Shell 1 is all unlike, shells 2 and 3 are all like.
    Every alpha is plus or minus one, so each shell contributes 4.
    Shell 4 is dropped in a 3x3x3 cell, so the expected objective is
    1.0 * 4 + 0.5 * 4 + 0.25 * 4 = 7.0
    """
    print("Test 2: B2 ordered bcc")
    a = 3.0
    frac = make_sites("bcc", (3, 3, 3))
    cell = cell_vectors(a, (3, 3, 3))
    shells, used = neighbour_shells(frac, cell, a, "bcc")

    # the basis alternates corner, body centre, corner, body centre, ...
    species = np.array([i % 2 for i in range(len(frac))])
    conc = np.array([0.5, 0.5])

    value, alphas = objective(species, shells, used, 2, conc)
    for alpha, number in zip(alphas, used):
        print("   shell", number, "alpha:")
        print(alpha)
    print("   shells used:", used)
    print("   objective:", round(value, 6), " expected 7.0")
    print()


def test_random():
    """A random ternary should give alphas near zero and a small objective."""
    print("Test 3: random ternary on fcc")
    a = 3.6
    frac = make_sites("fcc", (3, 3, 3))          # 108 sites
    cell = cell_vectors(a, (3, 3, 3))
    shells, used = neighbour_shells(frac, cell, a, "fcc")

    counts = [36, 36, 36]
    species = np.array([0] * counts[0] + [1] * counts[1] + [2] * counts[2])
    conc = np.array(counts) / len(species)

    rng = np.random.default_rng(0)
    values = []
    for _ in range(20):
        rng.shuffle(species)
        value, alphas = objective(species, shells, used, 3, conc)
        values.append(value)

    print("   sites:", len(frac), " shells used:", used)
    print("   objective over 20 random shuffles:")
    print("      mean", round(np.mean(values), 4),
          " min", round(np.min(values), 4),
          " max", round(np.max(values), 4))
    print("   a perfect SQS would be 0.0")
    print()


if __name__ == "__main__":
    test_neighbour_counts()
    test_b2_ordered()
    test_random()
