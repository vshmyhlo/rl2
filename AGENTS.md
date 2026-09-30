# Development guidelines

- Use Git only for read-only inspection. Do not run any Git operation that writes or mutates state, including staging, committing, amending, checking out or switching branches, restoring, resetting, cleaning, stashing, merging, rebasing, cherry-picking, reverting, fetching, pulling, pushing, or modifying branches, tags, remotes, configuration, or worktrees. Do not bypass this restriction by directly modifying Git metadata or using other tools to perform equivalent Git mutations. Ordinary source-file edits are allowed.
- Add type annotations to arguments and return values of new or modified Python functions and methods, including nested functions. Use type aliases for complex shared types.
- Run `uv run ruff format <changed-python-files>` on Python files you change before finishing. Follow the Ruff configuration in `pyproject.toml` and avoid formatting unrelated files.
- Use pytest for tests: write pytest test functions and run `uv run pytest` (or `uv run pytest <test-path>` for targeted checks).
