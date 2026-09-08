"""Diagnostic. How good is each shell in the structures the search returns?

Run from the repo root with the root on the path:
    PYTHONPATH=. python diagnostics/shell_check.py

Prints the sum of squared alphas shell by shell for a few search results,
then the same numbers for plain random configurations. If the search is
working, its numbers should be well below the random ones.
"""

import numpy as np

from sro import neighbour_shells, sro_matrix
from sqs_builder import make_sites, cell_vectors
from sqs_search import search, random_species


def shell_values(species, shells, n_species, conc):
    """Sum of squared alphas for each shell."""
    values = []
    for shell in shells:
        alpha = sro_matrix(species, shell, n_species, conc)
        values.append(np.sum(alpha ** 2))
    return values


def print_row(label, values):
    """One line of numbers, however many shells there are."""
    print("  " + label + "   " + "   ".join("%.4f" % v for v in values))


if __name__ == "__main__":
    lattice = "fcc"
    a = 3.6
    supercell = (3, 3, 3)
    counts = [36, 36, 36]
    steps = 5000000

    frac = make_sites(lattice, supercell)
    cell = cell_vectors(a, supercell)
    shells, used = neighbour_shells(frac, cell, a, lattice, n_shells=4)

    print("sites:", len(frac), " shells used:", used)
    print()

    n_species = len(counts)
    conc = np.array(counts) / sum(counts)

    header = "  seed   obj      " + "   ".join("shell%d " % n for n in used)
    print("search results, five seeds, %d steps each" % steps)
    print(header)
    for seed in range(5):
        best, report = search(frac, cell, a, lattice, counts,
                              max_steps=steps, seed=seed)
        values = shell_values(best, shells, n_species, conc)
        print_row("%4d   %.4f" % (seed, report["objective"]), values)
    print()

    print("plain random configurations, for reference")
    print("  seed   " + "   ".join("shell%d " % n for n in used))
    rng = np.random.default_rng(0)
    for seed in range(5):
        species = random_species(counts, rng)
        values = shell_values(species, shells, n_species, conc)
        print_row("%4d" % seed, values)
