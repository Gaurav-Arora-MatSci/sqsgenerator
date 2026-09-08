"""A one page website around the SQS builder.

Everything comes from a form. There is no language model anywhere in this
app. The visitor picks the lattice, the number of elements, the symbols,
the percentages, the lattice parameter and the supercell. The form is
checked, the search runs, and the structure and the report are offered
for download.

The app keeps no state at all. The POSCAR and the report are placed in
the result page itself, and the download buttons build the files in the
visitor's browser. Nothing is stored on the server and nothing is
written to disk, so any number of copies of this app can run side by
side without knowing about each other. That is what lets it run on Cloud
Run, which starts and stops containers as it likes.
"""

from flask import Flask, request, render_template_string, jsonify

import llm_fill
from sqs_builder import build_sqs, BASIS

app = Flask(__name__)

# Hard limits on what the form will accept.
MAX_ATOMS = 500
MIN_REPEAT = 3

# The step box starts at DEFAULT_STEPS and will not accept more than
# MAX_STEPS. On a 500 atom cell, 50000 steps takes about 6 seconds, so
# 500000 steps sits near 60 seconds and returns well inside the Cloud Run
# request timeout.
DEFAULT_STEPS = 50000
MAX_STEPS = 500000

# Atoms per conventional cell, used to work out the total before building.
ATOMS_PER_CELL = {"bcc": 2, "fcc": 4}


def read_number(text, name, errors, whole=False):
    """Read one number from the form. Records an error and returns None."""
    text = text.strip()
    if not text:
        errors.append("%s is empty." % name)
        return None
    try:
        if whole:
            return int(text)
        return float(text)
    except ValueError:
        errors.append("%s is not a number." % name)
        return None


def read_form(form):
    """Check the whole form.

    Returns (settings, errors). settings is a dict ready to pass to
    build_sqs. errors is a list of plain sentences, empty if the form is
    good. Nothing is built until the list is empty.
    """
    errors = []

    lattice = form.get("lattice", "bcc")
    if lattice not in BASIS:
        errors.append("Lattice must be BCC or FCC.")

    n_elements = int(form.get("n_elements", "2"))

    # Element symbols and percentages, only for the rows in use.
    elements = []
    percents = []
    for n in range(1, n_elements + 1):
        symbol = form.get("element_%d" % n, "").strip()
        if not symbol:
            errors.append("Element %d has no symbol." % n)
        elements.append(symbol)

        value = read_number(form.get("percent_%d" % n, ""),
                            "Percentage for element %d" % n, errors)
        percents.append(value)

    if len(set(elements)) != len(elements):
        errors.append("The same element is listed more than once.")

    # The percentages must add to 100 before anything else happens.
    if None not in percents:
        total = sum(percents)
        if abs(total - 100.0) > 0.01:
            errors.append("The percentages add up to %.2f, not 100." % total)
        for value in percents:
            if value <= 0:
                errors.append("Every percentage must be greater than zero.")
                break

    lattice_parameter = read_number(form.get("lattice_parameter", ""),
                                    "Lattice parameter", errors)
    if lattice_parameter is not None and lattice_parameter <= 0:
        errors.append("Lattice parameter must be greater than zero.")

    repeats = []
    for axis in ["nx", "ny", "nz"]:
        value = read_number(form.get(axis, ""), "Repeat %s" % axis,
                            errors, whole=True)
        repeats.append(value)

    if None not in repeats:
        if min(repeats) < MIN_REPEAT:
            errors.append("Every repeat must be at least %d." % MIN_REPEAT)
        elif lattice in ATOMS_PER_CELL:
            total_atoms = (repeats[0] * repeats[1] * repeats[2]
                           * ATOMS_PER_CELL[lattice])
            if total_atoms > MAX_ATOMS:
                errors.append("That supercell holds %d atoms. The limit is %d."
                              % (total_atoms, MAX_ATOMS))

    steps = read_number(form.get("steps", ""), "Search steps",
                        errors, whole=True)
    if steps is not None:
        if steps < 1:
            errors.append("Search steps must be at least 1.")
        elif steps > MAX_STEPS:
            errors.append("Search steps must be %d or fewer." % MAX_STEPS)

    # An empty seed box means no seed, so every build differs.
    seed = None
    seed_text = form.get("seed", "").strip()
    if seed_text:
        try:
            seed = int(seed_text)
        except ValueError:
            errors.append("Seed must be a whole number, or empty.")

    settings = {
        "lattice": lattice,
        "lattice_parameter": lattice_parameter,
        "elements": elements,
        "composition": percents,
        "composition_mode": "percent",
        "supercell": tuple(repeats) if None not in repeats else None,
        "seed": seed,
        "max_steps": steps,
    }
    return settings, errors


def summary_text(result):
    """The block of text shown on the page after a build."""
    lines = []
    lines.append("build id      : %s" % result["build_id"])
    lines.append("lattice       : %s" % result["lattice"].upper())
    lines.append("supercell     : %d x %d x %d" % tuple(result["supercell"]))
    lines.append("total atoms   : %d" % result["total_atoms"])
    lines.append("seed          : %s" % result["seed"])
    lines.append("")
    lines.append("element   atoms   requested %   achieved %")
    for element in result["counts"]:
        lines.append("%-8s %6d %12.2f %11.2f"
                     % (element,
                        result["counts"][element],
                        result["requested_percent"][element],
                        result["achieved_percent"][element]))
    lines.append("")
    lines.append("shells used         : %s" % result["shells_used"])
    lines.append("objective           : %.6f" % result["objective"])
    lines.append("random mean of 20   : %.6f" % result["random_mean"])
    lines.append("random best of 20   : %.6f" % result["random_best"])
    lines.append("worst alpha         : %.4f" % result["worst_alpha"])
    lines.append("max alpha per shell : %s" % result["max_alpha_per_shell"])
    if result["better_than_best_random"] is None:
        lines.append("times better than best random : objective is zero")
    else:
        lines.append("times better than best random : %.1f"
                     % result["better_than_best_random"])
    lines.append("")
    lines.append("steps    : %d" % result["steps"])
    lines.append("restarts : %d" % result["restarts"])
    lines.append("seconds  : %.1f" % result["seconds"])
    return "\n".join(lines)


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SQS structure builder</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b1120;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0ea5e9;
      --warn: #fbbf24;
      --error: #f87171;
      --input-bg: #0f172a;
      --input-border: #1e293b;
      --input-border-hover: #334155;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      padding: 60px 40px;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--text-main);
      -webkit-font-smoothing: antialiased;
    }

    .container {
      width: 100%;
      max-width: 900px;
      margin: 0;
    }

    h1 {
      font-size: 39px;
      font-weight: 500;
      letter-spacing: -0.02em;
      margin-top: 0;
      margin-bottom: 16px;
      color: #ffffff;
    }

    h3 {
      font-size: 21px;
      font-weight: 500;
      color: #ffffff;
      margin-top: 40px;
      margin-bottom: 16px;
    }

    .note {
      font-size: 18px;
      color: var(--text-muted);
      line-height: 1.6;
      margin-bottom: 24px;
    }

    .note.small {
      font-size: 17px;
      margin-top: 12px;
      margin-bottom: 24px;
    }

    input[type=text], input[type=password], textarea, select {
      background: var(--input-bg);
      border: 1px solid var(--input-border);
      border-radius: 8px;
      color: #e2e8f0;
      font-family: 'JetBrains Mono', monospace;
      font-size: 18px;
      padding: 12px 14px;
      transition: border-color 0.2s, box-shadow 0.2s;
      outline: none;
    }

    select {
      font-family: 'Inter', sans-serif;
    }

    input[type=text]:focus, input[type=password]:focus,
    textarea:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 1px var(--accent);
    }

    input[type=text]:hover, input[type=password]:hover,
    textarea:hover, select:hover {
      border-color: var(--input-border-hover);
    }

    label {
      display: block;
      font-size: 18px;
      font-weight: 500;
      margin-top: 32px;
      margin-bottom: 10px;
      color: #e2e8f0;
    }

    .row {
      display: flex;
      gap: 16px;
      align-items: center;
      margin-bottom: 12px;
    }

    .row .tag {
      font-size: 17px;
      color: var(--text-muted);
      width: 90px;
    }

    .row input.symbol { width: 120px; }
    .row input.percent { width: 120px; }
    .row .unit {
      font-size: 17px;
      color: var(--text-muted);
    }

    .short { width: 120px; }

    .button-group {
      display: flex;
      gap: 16px;
      margin-top: 32px;
    }

    button, .btn-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      font-weight: 600;
      padding: 12px 24px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s ease;
      text-decoration: none;
      border: 1px solid transparent;
      font-family: 'Inter', sans-serif;
    }

    /* Colour comes from the class, never from the element type. A type
       selector such as button[type=button] would outrank .btn-primary and
       quietly turn a primary button grey. */
    .btn-primary {
      background: var(--accent);
      color: #0b1120;
      border-color: var(--accent);
    }

    .btn-primary:hover {
      background: var(--accent-hover);
      border-color: var(--accent-hover);
    }

    .btn-primary:disabled {
      opacity: 0.5;
      cursor: default;
    }

    .btn-secondary {
      background: transparent;
      color: var(--text-muted);
      border-color: var(--input-border);
    }

    .btn-secondary:hover {
      background: var(--input-border-hover);
      color: var(--text-main);
    }

    /* The bar is indeterminate on purpose. The search reports no
       progress back to the page, so a filling bar would be a false
       claim about how far along it is. */
    #working {
      display: none;
      margin-top: 32px;
      max-width: 460px;
    }

    #working .label {
      font-size: 17px;
      color: var(--text-muted);
      margin-bottom: 10px;
    }

    #working .keep-open {
      font-size: 17px;
      color: var(--warn);
      margin-top: 12px;
      line-height: 1.5;
    }

    .bar {
      height: 4px;
      background: var(--input-border);
      border-radius: 2px;
      overflow: hidden;
    }

    .bar span {
      display: block;
      width: 30%;
      height: 100%;
      background: var(--accent);
      animation: slide 1.4s ease-in-out infinite;
    }

    @keyframes slide {
      0%   { transform: translateX(-100%); }
      100% { transform: translateX(333%); }
    }

    @media (prefers-reduced-motion: reduce) {
      .bar span { animation: none; width: 100%; opacity: 0.4; }
    }

    pre {
      background: var(--input-bg);
      border: 1px solid var(--input-border);
      padding: 24px;
      border-radius: 8px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 17px;
      line-height: 1.6;
      color: #e2e8f0;
      white-space: pre-wrap;
      overflow-x: auto;
    }

    .problems {
      border: 1px solid var(--error);
      border-radius: 8px;
      padding: 20px 24px;
      margin-top: 32px;
      color: var(--error);
      font-size: 18px;
      line-height: 1.6;
    }

    .problems ul { margin: 8px 0 0 0; padding-left: 22px; }

    .assist {
      border: 1px solid var(--input-border);
      border-radius: 8px;
      padding: 24px 28px;
      margin-bottom: 8px;
    }

    .assist textarea {
      width: 100%;
      min-height: 80px;
      resize: vertical;
      line-height: 1.5;
      font-family: 'Inter', sans-serif;
    }

    input.wide { width: 100%; max-width: 460px; }

    .hidden-text {
      display: none;
    }

    .id {
      font-family: 'JetBrains Mono', monospace;
      font-size: 18px;
      color: var(--accent);
      display: inline-block;
    }
  </style>
</head>
<body>
<div class="container">
  <h1>SQS structure builder</h1>

  <p class="note">
    Builds a special quasirandom structure on a BCC or FCC lattice.
    Every repeat must be at least {{ min_repeat }}, and the cell must hold
    no more than {{ max_atoms }} atoms. The search stops at the number of
    steps you set, or earlier if it reaches a perfect score.
  </p>

  <div class="assist">
    <h3 style="margin-top: 0;">Fill the form by describing it, optional</h3>
    <p class="note small" style="margin-top: 0;">
      You do not need this. Skip straight to the form below and fill the
      boxes yourself. If you would rather write a sentence, paste your own
      Anthropic API key and the boxes get filled for you. Check them before
      you build. One request costs roughly a tenth of a US cent, so about
      a thousand requests per dollar.
    </p>
    <p class="note small">
      Your key is sent only to Anthropic, used once, and thrown away when
      the request ends. It is never stored on this server and never
      written to a log. It stays in this browser tab until you close it.
    </p>

    <label for="description" style="margin-top: 20px;">Describe the structure</label>
    <textarea id="description" rows="3"
      placeholder="A 4x4x4 BCC cell of W, Re and Ta at 70, 20 and 10 percent, lattice parameter 3.16"></textarea>

    <label for="api_key">Anthropic API key</label>
    <input type="password" id="api_key" class="wide"
           placeholder="sk-ant-..." autocomplete="off">

    <div class="button-group" style="margin-top: 20px;">
      <button type="button" class="btn-secondary" id="fill"
              onclick="fillForm()">Fill the form</button>
    </div>
    <p class="note small" id="assist_message" style="display: none;"></p>
  </div>

  <form method="post" onsubmit="startWorking()">

    <label for="lattice">Lattice</label>
    <select name="lattice" id="lattice" class="short">
      <option value="bcc" {% if form.lattice == 'bcc' %}selected{% endif %}>BCC</option>
      <option value="fcc" {% if form.lattice == 'fcc' %}selected{% endif %}>FCC</option>
    </select>

    <label for="n_elements">Number of elements</label>
    <select name="n_elements" id="n_elements" class="short" onchange="showRows()">
      {% for n in [2, 3, 4, 5] %}
        <option value="{{ n }}" {% if form.n_elements == n|string %}selected{% endif %}>{{ n }}</option>
      {% endfor %}
    </select>

    <label>Elements and composition</label>
    {% for n in [1, 2, 3, 4, 5] %}
      <div class="row element-row" id="row_{{ n }}">
        <span class="tag">Element {{ n }}</span>
        <input type="text" class="symbol" name="element_{{ n }}"
               value="{{ form.get('element_' ~ n, '') }}" placeholder="W">
        <input type="text" class="percent" name="percent_{{ n }}"
               value="{{ form.get('percent_' ~ n, '') }}" placeholder="50">
        <span class="unit">percent</span>
      </div>
    {% endfor %}
    <p class="note small" id="percent-total">
      The percentages must add up to exactly 100.
    </p>

    <label for="lattice_parameter">Lattice parameter, angstrom</label>
    <input type="text" class="short" id="lattice_parameter"
           name="lattice_parameter"
           value="{{ form.get('lattice_parameter', '') }}" placeholder="3.16">

    <label>Supercell repeats along a, b and c</label>
    <div class="row">
      <input type="text" class="short" name="nx" id="nx"
             value="{{ form.get('nx', '') }}" placeholder="4" oninput="showAtoms()">
      <input type="text" class="short" name="ny" id="ny"
             value="{{ form.get('ny', '') }}" placeholder="4" oninput="showAtoms()">
      <input type="text" class="short" name="nz" id="nz"
             value="{{ form.get('nz', '') }}" placeholder="4" oninput="showAtoms()">
    </div>
    <p class="note small" id="atom-count">
      Keep the cell close to cubic. A long thin cell cannot hold a good
      quasirandom structure.
    </p>

    <label for="steps">Search steps</label>
    <input type="text" class="short" id="steps" name="steps"
           value="{{ form.get('steps', default_steps) }}">
    <p class="note small">
      More steps give a lower objective, with diminishing returns. On a
      {{ max_atoms }} atom cell, {{ default_steps }} steps takes about six
      seconds. The largest value accepted is {{ max_steps }}, which keeps
      the build inside the time a browser will wait.
    </p>

    <label for="seed">Seed, optional</label>
    <input type="text" class="short" id="seed" name="seed"
           value="{{ form.get('seed', '') }}" placeholder="leave empty">
    <p class="note small">
      The search starts from a random configuration. Leave the seed empty
      and every build is different. Put a number in and the same form with
      the same number gives the same structure.
    </p>

    <div class="button-group">
      <button type="submit" id="build" class="btn-primary">Build structure</button>
      <button type="button" class="btn-secondary" onclick="window.location='/'">Clear</button>
    </div>

    <div id="working">
      <div class="label">Running, <span id="elapsed">0</span> s elapsed</div>
      <div class="bar"><span></span></div>
      <div class="keep-open">
        The search is running on the server. Do not close or reload this
        page until the result appears.
      </div>
    </div>
  </form>

  {% if errors %}
    <div class="problems">
      Nothing was built. Fix the following and try again.
      <ul>
        {% for message in errors %}<li>{{ message }}</li>{% endfor %}
      </ul>
    </div>
  {% endif %}

  {% if build_id %}
    <h3>Build identifier</h3>
    <p class="note" style="margin-bottom: 8px;"><span class="id">{{ build_id }}</span></p>
    <p class="note small">
      This id is written on the first line of the POSCAR and at the top of
      the report, together with the worst alpha and the seed.
    </p>
  {% endif %}

  {% if summary %}
    <h3>Result</h3>
    <pre>{{ summary }}</pre>

    <!-- The two files travel inside the page. The boxes are hidden, and
         the buttons below turn their contents into a download. The server
         keeps no copy. -->
    <textarea id="poscar_text" class="hidden-text">{{ poscar }}</textarea>
    <textarea id="report_text" class="hidden-text">{{ report }}</textarea>

    <div class="button-group" style="margin-top: 24px;">
      <button type="button" class="btn-primary"
              onclick="saveFile('poscar_text', 'POSCAR')">Download POSCAR</button>
      <button type="button" class="btn-secondary"
              onclick="saveFile('report_text', 'report.txt')">Download report.txt</button>
    </div>
  {% endif %}
</div>

<script>
  var ATOMS_PER_CELL = {bcc: 2, fcc: 4};
  var MAX_ATOMS = {{ max_atoms }};

  // Hide the element rows that are not in use.
  function showRows() {
    var wanted = parseInt(document.getElementById('n_elements').value);
    for (var n = 1; n <= 5; n++) {
      var row = document.getElementById('row_' + n);
      row.style.display = (n <= wanted) ? 'flex' : 'none';
    }
  }

  // Show the atom count while the repeats are typed, so the 500 limit is
  // visible before the form is sent.
  function showAtoms() {
    var nx = parseInt(document.getElementById('nx').value);
    var ny = parseInt(document.getElementById('ny').value);
    var nz = parseInt(document.getElementById('nz').value);
    var lattice = document.getElementById('lattice').value;
    var note = document.getElementById('atom-count');

    if (isNaN(nx) || isNaN(ny) || isNaN(nz)) {
      note.textContent = 'Keep the cell close to cubic. A long thin cell '
                       + 'cannot hold a good quasirandom structure.';
      return;
    }
    var total = nx * ny * nz * ATOMS_PER_CELL[lattice];
    note.textContent = 'That cell holds ' + total + ' atoms. The limit is '
                     + MAX_ATOMS + '.';
  }

  // Show the running message and count seconds. The button is disabled so
  // the search is not started twice by an impatient second click.
  function startWorking() {
    document.getElementById('working').style.display = 'block';
    document.getElementById('build').disabled = true;
    var seconds = 0;
    setInterval(function () {
      seconds += 1;
      document.getElementById('elapsed').textContent = seconds;
    }, 1000);
  }

  // Keep the key for as long as this tab is open, so it does not have to
  // be retyped after every build. sessionStorage is wiped when the tab
  // closes. The key is never sent anywhere except to this app's /parse
  // route, which passes it straight to Anthropic and then forgets it.
  var keyBox = document.getElementById('api_key');
  var saved = sessionStorage.getItem('api_key');
  if (saved) { keyBox.value = saved; }
  keyBox.addEventListener('input', function () {
    sessionStorage.setItem('api_key', keyBox.value);
  });

  function say(message) {
    var box = document.getElementById('assist_message');
    box.textContent = message;
    box.style.display = message ? 'block' : 'none';
  }

  // Send the sentence and the key to the server, then put the answer in
  // the form boxes. Nothing is built here. The visitor still has to look
  // at the boxes and press Build.
  async function fillForm() {
    var key = document.getElementById('api_key').value.trim();
    var text = document.getElementById('description').value.trim();

    if (!text) { say('Write a description first.'); return; }
    if (!key) { say('Paste your API key, or fill the form below yourself.'); return; }

    var button = document.getElementById('fill');
    button.disabled = true;
    say('Reading your description...');

    try {
      var reply = await fetch('/parse', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({api_key: key, description: text})
      });
      var data = await reply.json();

      if (data.error) { say(data.error); return; }

      document.getElementById('lattice').value = data.lattice;
      document.getElementById('n_elements').value = data.elements.length;
      showRows();
      for (var n = 1; n <= data.elements.length; n++) {
        document.getElementsByName('element_' + n)[0].value = data.elements[n - 1];
        document.getElementsByName('percent_' + n)[0].value = data.percentages[n - 1];
      }
      document.getElementById('lattice_parameter').value = data.lattice_parameter;
      document.getElementById('nx').value = data.supercell[0];
      document.getElementById('ny').value = data.supercell[1];
      document.getElementById('nz').value = data.supercell[2];
      showAtoms();

      say('Form filled. Check the values, then press Build structure.'
          + (data.note ? ' ' + data.note : ''));
    } catch (error) {
      say('Could not reach the server. Try again.');
    } finally {
      button.disabled = false;
    }
  }

  // Turn the text held in a hidden box into a downloaded file. The file
  // is assembled in the browser, so the server never has to remember it.
  function saveFile(boxId, filename) {
    var text = document.getElementById(boxId).value;
    var blob = new Blob([text], {type: 'text/plain'});
    var link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  document.getElementById('lattice').addEventListener('change', showAtoms);
  showRows();
  showAtoms();
</script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    build_id = None
    poscar = None
    report = None
    errors = []
    form = {"lattice": "bcc", "n_elements": "2"}

    if request.method == "POST":
        form = request.form.to_dict()
        settings, errors = read_form(request.form)

        if not errors:
            try:
                result = build_sqs(**settings)
                summary = summary_text(result)
                build_id = result["build_id"]
                poscar = result["poscar"]
                report = result["report_text"]
            except Exception as error:
                errors.append("The build failed: %s" % error)

    return render_template_string(PAGE, summary=summary, build_id=build_id,
                                  poscar=poscar, report=report,
                                  errors=errors, form=form,
                                  max_atoms=MAX_ATOMS, min_repeat=MIN_REPEAT,
                                  max_steps=MAX_STEPS,
                                  default_steps=DEFAULT_STEPS)


@app.route("/parse", methods=["POST"])
def parse():
    """Turn a sentence into form values using the visitor's own API key.

    The key arrives in the body of the request, never in the address, so
    it cannot end up in a server log. It is used once and dropped when
    this function returns. Nothing is built here.
    """
    data = request.get_json(silent=True) or {}
    api_key = data.get("api_key", "").strip()
    description = data.get("description", "").strip()

    if not api_key or not description:
        return jsonify({"error": "A description and an API key are both needed."})

    try:
        values = llm_fill.describe_to_form(api_key, description,
                                           ATOMS_PER_CELL)
        return jsonify(values)
    except ValueError as error:
        return jsonify({"error": str(error)})


if __name__ == "__main__":
    # Local testing only. Never run with debug on a public address.
    app.run(debug=True)