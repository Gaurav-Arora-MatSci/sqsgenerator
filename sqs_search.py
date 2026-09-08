"""Swap search for an SQS.

Swaps two sites of different species and keeps the swap if the objective
does not get worse. When it gets stuck it restarts from a fresh random
configuration and keeps the best result found overall.

There is no time limit. The search runs for max_steps, or stops early if
it reaches a perfect score. The twenty random reference configurations
are measured before the search starts, so the comparison numbers exist
even if the search itself gains nothing.
"""

import time
import numpy as np

from sro import neighbour_shells, objective


def random_species(counts, rng):
    """An array with the requested number of each species, shuffled."""
    species = []
    for index, number in enumerate(counts):
        species += [index] * number
    species = np.array(species)
    rng.shuffle(species)
    return species


def random_baseline(counts, shells, used, n_species, conc, rng, n_trials=20):
    """
    Objective of plain random configurations in the same cell.
    The raw objective depends on cell size and composition, so this gives
    a reference point the search result can be compared against.
    Returns the mean and the best of the trials. Beating the best is the
    meaningful test, since twenty random draws are cheap.
    """
    values = []
    for _ in range(n_trials):
        species = random_species(counts, rng)
        value, alphas = objective(species, shells, used, n_species, conc)
        values.append(value)
    return float(np.mean(values)), float(np.min(values))


def search(frac, cell, a, lattice, counts,
           max_steps=500000, stuck_limit=1000, seed=None):
    """
    Return the best configuration found and a report.
    Stops at max_steps, or earlier if the objective reaches zero.
    """
    rng = np.random.default_rng(seed)

    shells, used = neighbour_shells(frac, cell, a, lattice, n_shells=4)

    n_sites = len(frac)
    if sum(counts) != n_sites:
        raise ValueError("counts must add up to %d sites" % n_sites)

    n_species = len(counts)
    conc = np.array(counts) / n_sites

    # Measured first, on purpose. If the search runs the full budget and
    # gains little, these numbers still say how good the result is.
    random_mean, random_best = random_baseline(counts, shells, used,
                                               n_species, conc, rng)

    species = random_species(counts, rng)
    current, alphas = objective(species, shells, used, n_species, conc)

    best_species = species.copy()
    best_value = current
    best_alphas = alphas

    steps = 0
    restarts = 0
    rejections = 0
    start = time.time()

    while steps < max_steps:
        steps += 1

        # pick two sites holding different species
        i = rng.integers(n_sites)
        j = rng.integers(n_sites)
        if species[i] == species[j]:
            continue

        species[i], species[j] = species[j], species[i]
        trial, alphas = objective(species, shells, used, n_species, conc)

        if trial <= current:
            current = trial
            rejections = 0
            if trial < best_value:
                best_value = trial
                best_species = species.copy()
                best_alphas = alphas
        else:
            species[i], species[j] = species[j], species[i]   # undo
            rejections += 1

        if best_value < 1e-9:
            break

        if rejections >= stuck_limit:
            species = random_species(counts, rng)
            current, alphas = objective(species, shells, used,
                                        n_species, conc)
            rejections = 0
            restarts += 1

    # largest deviation from random in each shell, as a bond fraction
    max_alpha = [float(np.max(np.abs(alpha))) for alpha in best_alphas]

    if best_value > 0:
        better_than_mean = random_mean / best_value
        better_than_best = random_best / best_value
    else:
        better_than_mean = None
        better_than_best = None

    report = {
        "objective": best_value,
        "random_mean": random_mean,
        "random_best": random_best,
        "better_than_mean": better_than_mean,
        "better_than_best": better_than_best,
        "max_alpha": max_alpha,
        "worst_alpha": max(max_alpha),
        "shells_used": used,
        "alphas": best_alphas,
        "steps": steps,
        "restarts": restarts,
        "seconds": time.time() - start,
    }
    return best_species, report
