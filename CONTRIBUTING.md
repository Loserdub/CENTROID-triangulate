# Contributing to CENTROID

Thank you for your interest in improving CENTROID. We welcome bug fixes, performance improvements, and documentation enhancements.

The development cycle follows a simple, practical workflow:

$$\text{Change} \longrightarrow \text{Test} \longrightarrow \text{Document} \longrightarrow \text{Submit}$$

---

## 1. Development Principles

- **Local-First & Offline**: Do not add dependencies on external cloud APIs, telemetry, or remote services. CENTROID must operate entirely locally.
- **Reproducible & Deterministic**: Analysis and clustering results must remain deterministic across platforms (`seed=42`, `random_state=42`, MD5 word seeds).
- **Domain Agnostic**: CENTROID is a general-purpose semantic analysis engine. Do not optimize algorithms for a single domain or assume a specific corpus use case.
- **Lightweight Dependencies**: Prefer standard library solutions where possible. Avoid introducing heavy frameworks without measurable necessity.

---

## 2. Coding Standards

- Python 3.11+ compatibility.
- Use explicit type hints for function signatures and public APIs.
- Use `pathlib.Path` for cross-platform filesystem paths (avoid hardcoded path strings or OS-specific separators).
- Standardize on `logging` rather than arbitrary `print()` statements for diagnostic output.
- Wrap user-facing strings injected into HTML templates with `html.escape()` and serialize JSON with `safe_json_dumps()` to prevent XSS.

---

## 3. Testing Requirements

Every change must be validated before submitting:

1. **Run Unit Tests**:
   ```bash
   python -m unittest discover -s tests -p "test_*.py"
   ```
2. **Run Smoke Test Suite**:
   ```bash
   python tests/validate_pipeline.py
   ```
3. **Verify Pipeline Execution**:
   ```bash
   python run.py --dashboard --force
   ```

If you add a new feature or fix a bug, include corresponding unit test cases in `tests/test_centroid.py`.

---

## 4. Submitting Pull Requests

1. Fork the repository and create a feature branch (`git checkout -b feature/my-improvement`).
2. Implement your change adhering to coding standards.
3. Verify that all tests pass cleanly.
4. Update relevant documentation in `README.md`, `docs/ARCHITECTURE.md`, or `docs/DEVELOPMENT.md` if behavior or CLI parameters change.
5. Submit a concise Pull Request describing what changed, why, and confirming that tests pass.

---

## 5. Reporting Issues

When reporting an issue, please include:
- Operating system and Python version (`python --version`).
- The exact CLI command used to invoke CENTROID.
- Relevant log output or terminal tracebacks.
- A minimal reproduction sample or test fixture if applicable.
