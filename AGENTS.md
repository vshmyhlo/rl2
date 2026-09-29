# Development guidelines

- Add type annotations to arguments and return values of new or modified Python functions and methods, including nested functions. Use type aliases for complex shared types.
- Run `uv run ruff format <changed-python-files>` on Python files you change before finishing. Follow the Ruff configuration in `pyproject.toml` and avoid formatting unrelated files.
