"""Build a BCC or FCC special quasirandom structure as a POSCAR.

Only bcc and fcc. Both are cubic, so the neighbour shell distances are
fixed multiples of the lattice constant. HCP is not supported, because
its shell distances depend on c over a and the first shell splits when
c over a is not ideal.
"""

import random
import string

import numpy as np

from sqs_search import search

# Fractional coordinates of the basis inside one conventional cell.
BASIS = {
    "bcc": np.array([[0.0, 0.0, 0.0],
                     [0.5, 0.5, 0.5]]),
    "fcc": np.array([[0.0, 0.0, 0.0],
                     [0.0, 0.5, 0.5],
                     [0.5, 0.0, 0.5],
                     [0.5, 0.5, 0.0]]),
}

ID_CHARACTERS = string.ascii_uppercase + string.digits


def make_build_id(length=6):
    """A short random tag written on the first line of the POSCAR.

    Always random, never derived from the inputs, so two identical
    requests still get different tags. It does not use the search seed,
    so a fixed seed still gives a fresh tag on every build.
    """
    return "".join(random.choice(ID_CHARACTERS) for _ in range(length))


def cell_vectors(lattice_parameter, supercell):
    """Cell vectors for the full supercell. Cubic, so a diagonal matrix."""
    nx, ny, nz = supercell
    return np.diag([nx * lattice_parameter,
                    ny * lattice_parameter,
                    nz * lattice_parameter])


def make_sites(lattice, supercell):
    """Fractional coordinates of every site in the supercell."""
    basis = BASIS[lattice]
    nx, ny, nz = supercell
    repeat = np.array([nx, ny, nz], dtype=float)

    sites = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                shift = np.array([i, j, k], dtype=float)
                for b in basis:
                    sites.append((b + shift) / repeat)
    return np.array(sites)


def resolve_counts(composition, composition_mode, n_sites):
    """Turn a composition into integer atom counts that sum to n_sites."""
    values = np.array(composition, dtype=float)

    if composition_mode == "counts":
        counts = values.astype(int)
        if counts.sum() != n_sites:
            raise ValueError(
                "counts sum to %d but the cell has %d sites"
                % (counts.sum(), n_sites))
        return counts

    # Percent mode. Normalise first, so 50/50 and 0.5/0.5 both work.
    fractions = values / values.sum()
    ideal = fractions * n_sites
    counts = np.floor(ideal).astype(int)

    # Largest remainder method. Give the leftover sites to the species
    # whose ideal count was cut the most by the floor.
    leftover = n_sites - counts.sum()
    order = np.argsort(-(ideal - counts))
    for index in order[:leftover]:
        counts[index] += 1
    return counts


def poscar_text(comment, cell, elements, counts, sites):
    """Return the POSCAR file contents as one string. Writes nothing."""
    lines = []
    lines.append(comment)
    lines.append("1.0")
    for vector in cell:
        lines.append("  %18.12f %18.12f %18.12f" % tuple(vector))
    lines.append("  " + "  ".join(elements))
    lines.append("  " + "  ".join(str(c) for c in counts))
    lines.append("Direct")
    for site in sites:
        lines.append("  %18.12f %18.12f %18.12f" % tuple(site))
    return "\n".join(lines) + "\n"


def report_text(build_id, lattice, lattice_parameter, supercell,
                elements, counts, requested, seed, report):
    """Return the human readable report as one string."""
    total = int(counts.sum())
    lines = []

    lines.append("SQS build id: %s" % build_id)
    lines.append("=" * 60)
    lines.append("")

    lines.append("Structure")
    lines.append("  lattice           : %s" % lattice.upper())
    lines.append("  lattice parameter : %.4f angstrom" % lattice_parameter)
    lines.append("  supercell         : %d x %d x %d" % tuple(supercell))
    lines.append("  total atoms       : %d" % total)
    lines.append("  search seed       : %s" % seed)
    lines.append("")

    lines.append("Composition")
    lines.append("  element   atoms   requested %   achieved %")
    for element, count, want in zip(elements, counts, requested):
        got = 100.0 * float(count) / total
        lines.append("  %-8s %5d %12.2f %11.2f"
                     % (element, int(count), want, got))
    lines.append("")

    lines.append("Quality")
    lines.append("  shells used            : %s" % report["shells_used"])
    lines.append("  objective              : %.6f" % report["objective"])
    lines.append("  random mean of 20      : %.6f" % report["random_mean"])
    lines.append("  random best of 20      : %.6f" % report["random_best"])
    lines.append("  worst alpha            : %.4f" % report["worst_alpha"])
    lines.append("  max alpha per shell    : %s"
                 % [round(v, 4) for v in report["max_alpha"]])
    if report["better_than_mean"] is None:
        lines.append("  times better than mean : objective is zero")
        lines.append("  times better than best : objective is zero")
    else:
        lines.append("  times better than mean : %.1f"
                     % report["better_than_mean"])
        lines.append("  times better than best : %.1f"
                     % report["better_than_best"])
    lines.append("")
    lines.append("  The objective is the weighted sum of squared Warren")
    lines.append("  Cowley parameters. Zero is a perfect quasirandom cell.")
    lines.append("  The random numbers are measured before the search")
    lines.append("  starts, from 20 plain random configurations in the")
    lines.append("  same cell at the same composition. Beating the best of")
    lines.append("  those 20 is the meaningful test, because 20 random")
    lines.append("  draws are cheap.")
    lines.append("")

    for alpha, number in zip(report["alphas"], report["shells_used"]):
        lines.append("Warren Cowley alpha, shell %d" % number)
        lines.append("           " + "".join("%10s" % e for e in elements))
        for element, row in zip(elements, alpha):
            lines.append("  %-8s " % element
                         + "".join("%10.4f" % v for v in row))
        lines.append("")

    lines.append("  Zero means random. Negative means the row element")
    lines.append("  prefers the column element as a neighbour. Positive")
    lines.append("  means it avoids it.")
    lines.append("")

    lines.append("Search")
    lines.append("  steps    : %d" % report["steps"])
    lines.append("  restarts : %d" % report["restarts"])
    lines.append("  seconds  : %.1f" % report["seconds"])

    return "\n".join(lines) + "\n"


def write_file(path, text):
    """Save text to disk. Only used when running from the shell."""
    with open(path, "w") as handle:
        handle.write(text)


def build_sqs(lattice,
              lattice_parameter,
              elements,
              composition,
              composition_mode="percent",
              supercell=(3, 3, 3),
              seed=None,
              max_steps=500000,
              output_path=None):
    """Build an SQS on a bcc or fcc lattice.

    Returns a dict holding the POSCAR text under the key poscar and the
    plain text report under the key report_text. Files are written only
    if output_path is given.
    """
    lattice = lattice.lower()
    if lattice not in BASIS:
        raise ValueError("lattice must be bcc or fcc, got %s" % lattice)
    if len(elements) != len(composition):
        raise ValueError("elements and composition must have the same length")

    sites = make_sites(lattice, supercell)
    cell = cell_vectors(lattice_parameter, supercell)
    counts = resolve_counts(composition, composition_mode, len(sites))

    labels, report = search(sites, cell, lattice_parameter, lattice, counts,
                            max_steps=max_steps, seed=seed)

    # POSCAR needs the atoms grouped by species, in the order of elements.
    order = np.argsort(labels, kind="stable")
    sites = sites[order]

    build_id = make_build_id()
    comment = ("SQS-%s  %s  %s  %dx%dx%d  a %.4f  max_alpha %.4f  seed %s"
               % (build_id, lattice.upper(), " ".join(elements),
                  supercell[0], supercell[1], supercell[2],
                  lattice_parameter, report["worst_alpha"], seed))
    text = poscar_text(comment, cell, elements, counts, sites)

    total = int(counts.sum())
    requested = np.array(composition, dtype=float)
    requested = 100.0 * requested / requested.sum()

    notes = report_text(build_id, lattice, lattice_parameter, supercell,
                        elements, counts, requested, seed, report)

    if output_path is not None:
        write_file(output_path, text)
        write_file("report.txt", notes)

    result = {
        "build_id": build_id,
        "lattice": lattice,
        "supercell": list(supercell),
        "total_atoms": total,
        "counts": {e: int(c) for e, c in zip(elements, counts)},
        "requested_percent": {e: round(float(p), 2)
                              for e, p in zip(elements, requested)},
        "achieved_percent": {e: round(100.0 * float(c) / total, 2)
                             for e, c in zip(elements, counts)},
        "seed": seed,
        "shells_used": report["shells_used"],
        "objective": round(report["objective"], 6),
        "random_mean": round(report["random_mean"], 6),
        "random_best": round(report["random_best"], 6),
        "worst_alpha": round(report["worst_alpha"], 4),
        "max_alpha_per_shell": [round(v, 4) for v in report["max_alpha"]],
        "better_than_best_random": (None if report["better_than_best"] is None
                                    else round(report["better_than_best"], 1)),
        "steps": report["steps"],
        "restarts": report["restarts"],
        "seconds": round(report["seconds"], 1),
        "poscar": text,
        "report_text": notes,
    }
    return result


if __name__ == "__main__":
    result = build_sqs(
        lattice="fcc",
        lattice_parameter=3.5,
        elements=["Ni", "Fe"],
        composition=[80, 20],
        composition_mode="percent",
        supercell=(3, 3, 3),
        seed=None,
        output_path="POSCAR",
    )
    print(result["report_text"])
    print("first line of the POSCAR")
    print(result["poscar"].splitlines()[0])
