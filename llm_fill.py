"""Turn a plain sentence into values for the form.

The model only reads the sentence. It does no arithmetic and enforces no
limits. Everything countable is worked out here in Python, and the form
itself does the real validation when Build is pressed.

An earlier version asked the model to check atom counts against a limit.
It got the answer wrong in both directions, once claiming 432 was over a
500 limit, and once claiming 432 exceeded 432. Counting is the sort of
thing a language model is bad at and a computer is perfect at, so the
counting moved here.

The API key belongs to the visitor. It arrives with the request, is used
once, and is dropped when the request ends. It is never written to disk,
never put in a log, and never kept in memory between requests.
"""

import anthropic

MODEL = "claude-haiku-4-5"

SYSTEM = (
    "You read a short description of a crystal structure and report what "
    "it says. You do not build anything, you do not check any limits, and "
    "you do not do arithmetic. Report the numbers as given. "
    "Only bcc and fcc are supported. If the description asks for hcp or "
    "any other lattice, use the report_problem tool. "
    "If the description gives atom counts, put them in counts and leave "
    "percentages out. If it gives percentages or fractions, put them in "
    "percentages and leave counts out. Do not convert between the two. "
    "If the description does not give a supercell, pick a cubic one that "
    "suits the request. If it does not give a lattice parameter, use a "
    "reasonable value for the main element. Say so in the note. "
    "If the description is unclear, or asks for something other than a "
    "bcc or fcc alloy cell, use the report_problem tool."
)

TOOLS = [
    {
        "name": "fill_form",
        "description": "Report the settings the description asks for.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lattice": {
                    "type": "string",
                    "enum": ["bcc", "fcc"],
                },
                "elements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Chemical symbols, two to five of them.",
                },
                "percentages": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Use only if the description gives percentages or "
                        "fractions. One per element, same order."
                    ),
                },
                "counts": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "Use only if the description gives atom numbers. "
                        "One per element, same order."
                    ),
                },
                "lattice_parameter": {
                    "type": "number",
                    "description": "Lattice constant a, in angstrom.",
                },
                "supercell": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Three repeats along a, b and c.",
                },
                "note": {
                    "type": "string",
                    "description": (
                        "One short sentence saying what was assumed, for "
                        "anything the description did not state."
                    ),
                },
            },
            "required": ["lattice", "elements", "lattice_parameter",
                         "supercell"],
        },
    },
    {
        "name": "report_problem",
        "description": "Use when the description is not a bcc or fcc alloy cell.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "One plain sentence saying what is wrong.",
                },
            },
            "required": ["reason"],
        },
    },
]


def to_percentages(counts, sites):
    """Convert atom counts to percentages, checking they fill the cell.

    Six decimal places is enough for the form's largest remainder method
    to recover the original counts exactly.
    """
    total = sum(counts)
    if total != sites:
        raise ValueError(
            "Those atom counts add up to %d, but that cell has %d sites. "
            "Adjust the counts or the supercell." % (total, sites))
    return [round(100.0 * count / total, 6) for count in counts]


def tidy(values, atoms_per_cell):
    """Check the reply and turn it into plain form values.

    All the counting happens here. The model is not asked to do any of it.
    """
    elements = values.get("elements") or []
    supercell = values.get("supercell") or []
    percentages = values.get("percentages")
    counts = values.get("counts")

    if not 2 <= len(elements) <= 5:
        raise ValueError("The description names %d elements. Two to five "
                         "are allowed." % len(elements))
    if len(supercell) != 3:
        raise ValueError("The supercell needs three numbers.")

    if counts:
        if len(counts) != len(elements):
            raise ValueError("There are %d elements but %d atom counts."
                             % (len(elements), len(counts)))
        nx, ny, nz = supercell
        sites = nx * ny * nz * atoms_per_cell[values["lattice"]]
        percentages = to_percentages(counts, sites)
    elif percentages:
        if len(percentages) != len(elements):
            raise ValueError("There are %d elements but %d percentages."
                             % (len(elements), len(percentages)))
    else:
        raise ValueError("The description gives no composition.")

    return {
        "lattice": values["lattice"],
        "elements": elements,
        "percentages": percentages,
        "lattice_parameter": values["lattice_parameter"],
        "supercell": supercell,
        "note": values.get("note", ""),
    }


def describe_to_form(api_key, description, atoms_per_cell):
    """Ask the model to read the description, then tidy up its reply.

    Returns a dict of form values. Raises ValueError with a plain message
    on any failure, including a bad key or a description the model could
    not use.
    """
    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM,
            tools=TOOLS,
            tool_choice={"type": "any"},   # force one of the two tools
            messages=[{"role": "user", "content": description}],
        )
    except anthropic.AuthenticationError:
        raise ValueError("That API key was not accepted.")
    except anthropic.RateLimitError:
        raise ValueError("The API rate limit was reached. Try again shortly.")
    except Exception as error:
        raise ValueError("The API call failed: %s" % error)

    for block in response.content:
        if block.type != "tool_use":
            continue
        if block.name == "report_problem":
            raise ValueError(block.input.get("reason", "the request was unclear"))
        if block.name == "fill_form":
            return tidy(dict(block.input), atoms_per_cell)

    raise ValueError("The model did not return any settings.")