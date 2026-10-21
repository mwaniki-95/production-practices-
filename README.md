# Production Python Week 2: From Notebook Code to an Installable Package

In this session I rebuilt notebook code (loading Iris, exploring it, plotting it) as a proper Python package called `production-iris`, imported as `production_iris`. The aim was not new maths. It was to make the code **installable**, **testable**, and **clean**, so it runs on any machine and not only inside one notebook.

| Quality | Tool | How I know it works |
|---|---|---|
| Installable | `pyproject.toml` + `pip` | The package imports from any folder in the venv |
| Testable | `pytest` | All tests pass |
| Clean | `ruff` + `mypy` | Style and type checks pass |

## 1. Project layout

```
production-iris/
├── pyproject.toml
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   └── production_iris/
│       ├── __init__.py
│       ├── data.py
│       ├── eda.py
│       ├── mathutils.py
│       └── visualization/
│           ├── __init__.py
│           └── comparisons.py
└── tests/
    ├── test_data.py
    ├── test_eda.py
    ├── test_mathutils.py
    └── test_visualization.py
```

| File or folder | Job |
|---|---|
| `pyproject.toml` | Package name, version, dependencies, tool settings |
| `src/production_iris/` | The installable code and nothing else |
| `__init__.py` | Marks a folder as a package and re-exports the public functions |
| `data.py` | Loads the dataset |
| `eda.py` | Summary statistics and checks, returned as a dictionary |
| `mathutils.py` | Small calculations with input validation |
| `visualization/` | Plotting functions, in their own subpackage |
| `tests/` | Automated checks, kept outside the package |
| `requirements.txt` | Exact versions that worked, for reproducing the environment |
| `.gitignore` | Files git must not track |

## 2. Why the `src/` layout

`src` means **source**. It is a convention, not a Python rule. `pyproject.toml` points to it with `where = ["src"]`.

- Python cannot import the package by accident from the project folder, so tests run against the **installed** package, the way a real user gets it.
- Code people install is kept apart from tests, config, and notebooks.
- The folder name under `src/` must match the import name (`production_iris`, with an underscore, because a hyphen is not valid in an import).

## 3. Virtual environment

A venv is a private copy of Python plus its own packages, one per project.

```bash
cd ~/ml-projects/production-iris
python3 -m venv .venv
source .venv/bin/activate
which python
```

`which python` must end in `production-iris/.venv/bin/python`. On Windows PowerShell the activation line is `.venv\Scripts\Activate.ps1`.

**Lessons learned**

- `(.venv)` in the prompt only says *some* venv is active, not which one. A venv from another project **follows you** when you `cd`. Always check with `which python`.
- `externally-managed-environment` means pip was run on the **system** Python (PEP 668). Activate the project venv. Never use `--break-system-packages`.
- VS Code picks its own interpreter: `Ctrl+Shift+P`, then **Python: Select Interpreter**.

## 4. `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "production-iris"
version = "0.1.0"
description = "Iris EDA and math helper functions"
requires-python = ">=3.9"
dependencies = ["pandas", "scikit-learn", "matplotlib", "seaborn"]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 79

[tool.ruff.lint]
select = ["E", "W", "F", "N"]
```

| Section | Meaning |
|---|---|
| `[build-system]` | How pip builds the package |
| `dependencies` | Libraries the package needs to run |
| `optional-dependencies [dev]` | Tools only needed while developing |
| `packages.find, where = ["src"]` | Tells pip the code lives in `src/` |
| `[tool.pytest...]`, `[tool.ruff...]` | Settings for the test runner and linter |

The **install name** and the **import name** can differ: `scikit-learn` is installed, `sklearn` is imported, and `production-iris` is installed while `production_iris` is imported.

## 5. Editable install

```bash
pip install -e ".[dev]"
```

- `.` means the project in the current folder, so run it where `pyproject.toml` lives.
- `[dev]` also installs the development tools.
- `-e` (editable) links the installed package to my `src/` folder, so my edits take effect without reinstalling.

**`ModuleNotFoundError: No module named 'production_iris'`** means one of two things: the wrong venv is active, or the package was never installed in this venv. Fix: `which python`, then `pip install -e .`.

## 6. Writing a module: `data.py`

```python
"""Data loading for the Iris dataset."""

import pandas as pd
from sklearn.datasets import load_iris


def load_iris_data() -> pd.DataFrame:
    """Load Iris with 'target' (0, 1, 2) and 'species' columns."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df
```

Rules this file follows:

| Rule | Reason |
|---|---|
| Imports at the top of the file | PEP 8; the dependencies are visible at a glance |
| Load the dataset once, store it in a variable | Avoids repeating the same work |
| Return the DataFrame, never print it | The caller decides what to do with it |
| No code at module level that runs on import | Importing a module must have no side effects |
| No `try/except` that hides a failure | Returning an empty DataFrame only moves the error somewhere harder to trace |

The rule from the error handling lessons applies: **raise where the problem is detected, catch where you can recover.**

## 7. `__init__.py`: the package's front door

```python
"""Production Iris package."""

from .data import load_iris_data
from .eda import exploratory_data_analysis

__all__ = [
    "load_iris_data",
    "exploratory_data_analysis",
]
```

| Import style | Works because |
|---|---|
| `from production_iris.data import load_iris_data` | It names the module |
| `from production_iris import load_iris_data` | `__init__.py` re-exports it |

- The leading dot in `from .data import ...` means **this same folder**.
- `__all__` lists the names the package officially offers. It also stops Ruff calling the imports unused (`F401`).
- Re-exporting means that if I move a function to another file later, users' imports do not break.

## 8. `eda.py`: return data, do not print it

```python
"""Exploratory data analysis for the Iris dataset."""

import pandas as pd


def exploratory_data_analysis(df: pd.DataFrame) -> dict[str, object]:
    """Return summary information about the Iris DataFrame as a dict.

    Raises ValueError if the 'species' column is missing.
    """
    if "species" not in df.columns:
        raise ValueError("DataFrame must contain a 'species' column")

    features = df.drop(columns=["target"], errors="ignore")

    return {
        "dataframe_size": df.shape,
        "data_types": df.dtypes,
        "column_names": df.columns.tolist(),
        "missing_values": df.isnull().sum(),
        "summary_statistics": df.describe(),
        "correlation_matrix": features.corr(numeric_only=True),
        "unique_species": df["species"].unique(),
        "species_count": df["species"].value_counts(),
    }
```

- `describe()` quietly skips text columns, but `corr()` does not. Recent pandas raises a `ValueError` on a text column, so I use `numeric_only=True`.
- `target` is just a numeric copy of `species`, so I drop it before correlating.
- Validation comes **before** the calculation, and it raises a clear `ValueError` instead of letting a confusing `KeyError` appear later.
- Returning a dictionary with named keys is easier to use than a tuple of seven unnamed items.

## 9. Testing with pytest

Two ways to check code:

| Way | Use |
|---|---|
| `python -c "..."` | A quick one-off look at one function |
| `pytest` | A permanent record that re-checks everything on every run |

```bash
python -c "from production_iris import load_iris_data; print(load_iris_data().shape)"
pytest -v
```

The first prints `(150, 6)`. pytest finds files named `test_*.py` and functions named `test_*`, and runs them. My run showed `5 passed, 1 warning`.

Example tests:

```python
import pandas as pd
import pytest

from production_iris import exploratory_data_analysis, load_iris_data


def test_iris_data_shape():
    assert load_iris_data().shape == (150, 6)


def test_eda_requires_species():
    with pytest.raises(ValueError):
        exploratory_data_analysis(pd.DataFrame({"a": [1, 2]}))
```

- `assert` passes if the condition is true.
- `pytest.raises(ValueError)` passes only if the code inside raises that error. This is how I prove validation works.
- A test should check a **result**, not only that code runs without crashing.

**The warning I saw:** `FigureCanvasAgg is non-interactive, and thus cannot be shown` came from a `plt.show()` inside a library function. Library code should **build and return** the figure, and the caller decides whether to show or save it, the same rule as `print()` and `input()`.

## 10. Code quality

```bash
ruff check .
ruff format --check .
mypy src
```

| Tool | Checks | Standard |
|---|---|---|
| `ruff check` | Mistakes and style: unused imports, naming, spacing | PEP 8 |
| `ruff format` | Layout: line length, brackets, quotes | Consistent style |
| `mypy` | Wrong types in calls and returns | PEP 484 |

- Type hints are **not enforced** when the program runs. A type checker such as mypy, or the editor, flags mismatches before you run the code.
- A comma after the last item in a list or dictionary tells the formatter to put **one item per line**, with the closing bracket on its own line. Ruff's message `eda.py:25:6` gave the exact line and column of my misplaced brace.
- Ruff reads the file on disk, so **save before checking**.

## 11. Git for this project

```bash
git init -b main
git status
git add .
git status
git commit -m "chore: scaffold production_iris package layout"
```

- Read `git status` **before** committing. `.venv/`, `__pycache__/`, `*.egg-info/` and the tool caches must not appear.
- There are two `.gitignore` files. `venv` writes its own inside `.venv/`, which hides that folder. My root `.gitignore` covers caches and build files.
- `fatal: not a git repository` means there is no `.git` folder yet, so run `git init -b main`.
- Commit prefixes: `feat` (new code), `fix`, `docs`, `test`, `chore`, `refactor`. Keep the subject short and in the imperative mood.

```bash
pip freeze --exclude-editable > requirements.txt
```

This saves the exact versions of everything installed, minus the package itself. `pyproject.toml` stays the source of truth for what the package needs.

## 12. Errors I hit and what fixed them

| Error | Cause | Fix |
|---|---|---|
| `ni: command not found` | `ni` is PowerShell; I was in bash | `touch` |
| Command fails with odd characters at the start | Paste artifact in the terminal | Type the command by hand |
| `externally-managed-environment` | pip ran on the system Python | Activate the project venv |
| `ModuleNotFoundError: production_iris` | Wrong venv, or package not installed | `which python`, `pip install -e .` |
| `ruff: command not found` | Ruff lives in a different venv | Activate the right venv |
| `fatal: not a git repository` | No `.git` folder | `git init -b main` |
| Ruff says the file would be reformatted | Layout differs from Ruff's style | `ruff format file.py` |

## 13. Week 2 checklist

| Item | Status |
|---|---|
| `src/` layout, `pyproject.toml`, editable install | Done |
| `data.py`, `eda.py`, re-exports in `__init__.py` | Done |
| pytest running and passing | Done |
| Type hints and input validation | Done on the modules written so far |
| Tests that use `pytest.raises` | To do |
| Remove `plt.show()` from library code | To do |
| Logging instead of `print` | To do |
| Config as a dataclass, no globals | To do |
| Model code (train and predict) from the Iris notebook | To do |
| `ruff` and `mypy` on the whole package | To do |
| Fresh clone, install, and test | To do |

## 14. Command cheat sheet

```bash
source .venv/bin/activate        # activate the venv
which python                     # confirm the right one
pip install -e ".[dev]"          # install the package and dev tools
pytest -v                        # run the tests
ruff check .                     # lint
ruff format .                    # format
mypy src                         # type check
git status                       # see what changed
pip freeze --exclude-editable > requirements.txt
```

## 15. Key takeaways

1. A package is **installed**, not copied. `pip install -e .` makes it importable from anywhere in the venv.
2. `src/` keeps installable code separate from everything else.
3. Check `which python` whenever imports or tools misbehave.
4. Importing a module must have no side effects. Functions **return**, they do not print.
5. Raise where the problem is detected and catch where you can recover.
6. `__init__.py` decides what users can import directly.
7. A test records what correct behaviour is, and `pytest.raises` proves the rejection of bad input.
8. Read `git status` before every commit.
