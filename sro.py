"""Neighbour shells and the Warren Cowley short range order objective.

This file only receives positions. The lattice itself is built in
sqs_builder.py, so there is one lattice builder in the repo and no chance
of two copies drifting apart.

Each shell is stored as two flat index arrays called centre and
neighbour. Together they list every neighbour pair in the cell for that
shell. Counting how many pairs of each species combination there are is
then one bincount call, instead of a Python loop over every site. The
numbers are the same as the loop version, only much faster, which is what
makes fifty thousand search steps possible.

The checks for this file are in diagnostics/test_sro.py.
"""

import numpy as np


# distance in units of the lattice constant, and the neighbour count
SHELL_TABLE = {
    "bcc": [(np.sqrt(3) / 2, 8),
            (1.0, 6),
            (np.sqrt(2.0), 12),
            (np.sqrt(11) / 2, 24)],
    "fcc": [(1 / np.sqrt(2), 12),
            (1.0, 6),
            (np.sqrt(1.5), 24),
            (np.sqrt(2.0), 12)],
}

WEIGHTS = [1.0, 0.5, 0.25, 0.125]


def neighbour_shells(frac, cell, a, lattice, n_shells=4):
    """
    Neighbour pair lists for the first n_shells shells.

    Each shell is returned as a pair of flat arrays (centre, neighbour).
    If site 7 has site 12 as a neighbour in this shell, then somewhere in
    the arrays there is a position where centre is 7 and neighbour is 12.

    A shell is skipped if it reaches further than half the shortest box
    vector, because the minimum image convention would count it wrongly.
    Returns the shells and the shell numbers that were actually used.
    """
    table = SHELL_TABLE[lattice][:n_shells]
    tol = 0.05 * a
    box = np.array([cell[0][0], cell[1][1], cell[2][2]])
    limit = 0.5 * box.min()

    shells = []
    used = []
    for number, (factor, expected) in enumerate(table, start=1):
        target = factor * a
        if target >= limit:
            continue

        centre = []
        neighbour = []
        for i in range(len(frac)):
            diff = frac - frac[i]
            diff = diff - np.round(diff)      # minimum image convention
            vec = diff @ cell
            dist = np.sqrt(np.sum(vec ** 2, axis=1))
            found = np.where(np.abs(dist - target) < tol)[0]
            centre.append(np.full(len(found), i))
            neighbour.append(found)

        shells.append((np.concatenate(centre), np.concatenate(neighbour)))
        used.append(number)

    return shells, used


def sro_matrix(species, shell, n_species, conc):
    """
    Warren Cowley parameters for one shell.
    alpha[i][j] = 1 - P(j is a neighbour of i) / c[j]
    Zero means random. Negative means i prefers j. Positive means i avoids j.
    """
    centre, neighbour = shell

    # Turn each pair into a single number, so one bincount counts them all.
    # Pair (i, j) becomes i * n_species + j, which is unique for each pair.
    pair = species[centre] * n_species + species[neighbour]
    counts = np.bincount(pair, minlength=n_species * n_species)
    counts = counts.reshape(n_species, n_species).astype(float)

    total = counts.sum(axis=1, keepdims=True)
    probability = counts / total
    return 1.0 - probability / conc


def objective(species, shells, used, n_species, conc):
    """
    Sum of squared SRO parameters over the shells that were used.
    The weight is looked up by shell number, so a dropped shell does not
    shift the weights of the shells that remain.
    """
    value = 0.0
    alphas = []
    for shell, number in zip(shells, used):
        alpha = sro_matrix(species, shell, n_species, conc)
        value += WEIGHTS[number - 1] * np.sum(alpha ** 2)
        alphas.append(alpha)
    return value, alphas
