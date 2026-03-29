# Contributing to HeRoS

Thank you for your interest in contributing to HeRoS! This document provides
guidelines and instructions for contributing.

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and
constructive environment for all contributors. Harassment, personal attacks,
and other uncooperative behavior will not be tolerated.

---

## Getting Started

### Prerequisites

- Python ≥ 3.9
- `git`
- A fork of the repository (for external contributors)

### Development Setup

```bash
# Clone your fork
git clone https://github.com/<your-username>/heros.git
cd heros

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run the test suite
python -m pytest tests/ -q

# Verify everything is working
python -c "from heros import HeRoSAgent; print('HeRoS ready!')"
```

---

## How to Contribute

### Reporting Bugs

Before submitting a bug report:

1. Search existing issues to avoid duplicates
2. Run `python -m pytest tests/` to confirm the bug is reproducible
3. Include:
   - Python version, OS
   - Minimal code snippet to reproduce
   - Full traceback
   - Expected vs. actual behavior

### Suggesting Features

Open a GitHub Issue with the label `enhancement` and include:

- Clear problem statement (what use case does this solve?)
- Proposed solution or API design
- Alternatives considered

### Pull Request Process

#### 1. Branch Naming

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
# or
git checkout -b step-N-feature-name   # for numbered roadmap steps
```

#### 2. Development Standards

- **Type hints** — All new functions should include type annotations
- **Docstrings** — Use Google-style docstrings for public APIs
- **Tests** — New features require corresponding tests in `tests/`
- **No breaking changes** — The public API (imports from `heros.*`) should remain stable

```python
def example_function(obs: np.ndarray, *, temperature: float = 0.0) -> list[str]:
    """Short one-line summary.

    Args:
        obs: Observation from the environment.
        temperature: Sampling temperature for the LLM backend.

    Returns:
        List of action strings.

    Raises:
        ValueError: If obs is empty.
    """
    if len(obs) == 0:
        raise ValueError("obs must not be empty")
    ...
```

#### 3. Running Tests

```bash
# Run all tests
python -m pytest tests/ -q

# Run tests for a specific module
python -m pytest tests/test_planner.py -v

# Run with coverage
python -m pytest tests/ --cov=src/heros --cov-report=term-missing
```

**Minimum bar for PRs:** All existing tests must pass (`645+` tests). New
features must include passing tests.

#### 4. Linting

```bash
# Install linting tools (if adding a linter config)
pip install flake8 black

# Format code
black src/ tests/

# Check style
flake8 src/ --max-line-length=100
```

#### 5. Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(planner): add configurable planning depth parameter

Add depth parameter to SubgoalPlanner for controlling the maximum
number of subgoals generated per task. Defaults to 5, range 1-10.

Closes #42
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

#### 6. Documentation

- Update `README.md` if adding new components to the public API
- Add docstrings to all public classes and functions
- Update `CHANGELOG.md` under `[Unreleased]` for user-facing changes

---

## Architecture Notes

### Public vs. Private API

```
heros.*          # Public API — stable, semver-protected
heros._private.* # Private — may change without notice
```

### Component Backends

All major components support pluggable backends:

| Component | Backend | Description |
|---|---|---|
| `SubgoalPlanner` | `llm` | LLM-based task decomposition |
| `SubgoalPlanner` | `rule-based` | Deterministic rule-based (default, no API key needed) |
| `MilestoneCritic` | `llm` | LLM-as-critic with rubric evaluation |
| `MilestoneCritic` | `rule-based` | Rule-based rubric matching (default, no API key needed) |

When implementing new backends, maintain both modes for reproducibility.

### Testing Strategy

- **Unit tests** (`tests/test_*.py`): Test individual components in isolation
- Use mocks for LLM API calls (` unittest.mock.patch`)
- Fixtures in `conftest.py` for shared setup
- All tests must be deterministic (no random failures)

```python
def test_milestone_pass():
    critic = MilestoneCritic(backend="rule-based")
    result = critic.evaluate(
        milestone_rubric="Output must contain 'success'",
        execution_trace="Operation completed successfully",
    )
    assert result.verdict == Verdict.PASS
```

---

## Review Process

1. Automated checks must pass (tests, linting if configured)
2. At least one review approval required
3. Address reviewer feedback or explain why not
4. The reviewer will merge once approved

---

## Labels

| Label | Meaning |
|---|---|
| `bug` | Something isn't working |
| `enhancement` | New feature or improvement |
| `question` | Further information is requested |
| `good first issue` | Suitable for new contributors |
| `docs` | Documentation only changes |

---

## License

By contributing to HeRoS, you agree that your contributions will be licensed
under the project's MIT License.
