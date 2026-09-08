"""Turn a plain sentence into values for the form.

The model never builds anything. It only reads the sentence and returns
the numbers that go in the boxes. The visitor then looks at those boxes
and presses Build, and the same checks run as always. So a mistake by
the model is visible before any time is spent on a search.

The API key belongs to the visitor. It arrives with the request, is used
once, and is dropped when the request ends. It is never written to disk,
never put in a log, and never kept in memory between requests.
"""

import anthropic

MODEL = "claude-haiku-4-5"

# The limits are passed in from app.py rather than written here, because
# an earlier version hardcoded them and got the bcc ceiling wrong. The
# model then refused a legal 6x6x6 bcc cell and invented a reason for it.
SYSTEM_TEMPLATE = (
    "You read a short description of a crystal structure and return the "
    "settings for a form. You never build anything. "
    "Only bcc and fcc are supported. If the description asks for hcp or "
    "any other lattice, use the report_problem tool. "
    "Between two and five elements are allowed. "
    "Percentages must add up to 100. If the description gives atom counts "
    "or ratios, convert them to percentages and give at least four "
    "decimal places, so the counts can be recovered exactly. "
    "Keep the supercell close to cubic and at least %(min_repeat)d in "
    "every direction. "
    "The number of atoms is nx times ny times nz times %(bcc)d for bcc, "
    "or times %(fcc)d for fcc, and must not exceed %(max_atoms)d. Work "
    "this out for the cell you are about to return, and do not assume a "
    "ceiling. For reference, the largest cubic cell allowed is "
    "%(bcc_max)dx%(bcc_max)dx%(bcc_max)d for bcc and "
    "%(fcc_max)dx%(fcc_max)dx%(fcc_max)d for fcc. "
    "If the description gives atom counts, check they add up to the "
    "number of sites in the cell you return. If they do not, use "
    "report_problem and say what the mismatch is. "
    "If the description does not give a supercell, pick a sensible cubic "
    "one within the limit. "
    "If the description does not give a lattice parameter, use a "
    "reasonable value for the main element. "
    "If anything essential is missing or contradictory, use the "
    "report_problem tool and say plainly what is missing."
)


def build_system(max_atoms, min_repeat, atoms_per_cell):
    """Fill the prompt with the limits the form actually enforces."""
    bcc = atoms_per_cell["bcc"]
    fcc = atoms_per_cell["fcc"]

    # Largest n where n cubed times the basis size still fits.
    def largest_cube(per_cell):
        n = min_repeat
        while (n + 1) ** 3 * per_cell <= max_atoms:
            n += 1
        return n

    return SYSTEM_TEMPLATE % {
        "min_repeat": min_repeat,
        "max_atoms": max_atoms,
        "bcc": bcc,
        "fcc": fcc,
        "bcc_max": largest_cube(bcc),
        "fcc_max": largest_cube(fcc),
    }

TOOLS = [
    {
        "name": "fill_form",
        "description": "Return the settings that go in the form boxes.",
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
                        "One per element, same order, adding up to 100."
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
            "required": ["lattice", "elements", "percentages",
                         "lattice_parameter", "supercell"],
        },
    },
    {
        "name": "report_problem",
        "description": "Use when the description cannot be turned into a form.",
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


def check(values):
    """Sanity check what came back. Raises ValueError on nonsense.

    This is not the real validation. The form does that when Build is
    pressed. This only catches a reply so malformed that the page could
    not display it.
    """
    elements = values.get("elements", [])
    percentages = values.get("percentages", [])

    if not 2 <= len(elements) <= 5:
        raise ValueError("the model returned %d elements, not 2 to 5"
                         % len(elements))
    if len(percentages) != len(elements):
        raise ValueError("the model returned a different number of "
                         "percentages and elements")
    if len(values.get("supercell", [])) != 3:
        raise ValueError("the model did not return three repeats")
    return values


def describe_to_form(api_key, description, max_atoms, min_repeat,
                     atoms_per_cell):
    """Ask the model for form values.

    Returns a dict of values on success. Raises ValueError with a plain
    message on any failure, including a bad key or a description the
    model refused.
    """
    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=build_system(max_atoms, min_repeat, atoms_per_cell),
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
            return check(dict(block.input))

    raise ValueError("The model did not return any settings.")