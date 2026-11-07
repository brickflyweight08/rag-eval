# 7. Plugin loading puts the current directory on the import path

Status: accepted (2026-11-07)

## Context

Systems are loaded from `module.path:factory`. The demo factories live in
`examples/toy_handbook/systems.py`, which is not part of the installed package. A
console script such as `rag-eval` does not put the current directory on `sys.path`, so
`examples.toy_handbook.systems` would not import from the repo root.

## Decision

- `loading.load_system` inserts the current working directory at the front of
  `sys.path` (if missing) before importing, like `python -m` does.
  `examples/` and `examples/toy_handbook/` have `__init__.py` files.
- Errors distinguish: bad spec format, module not found, a dependency of the module
  missing, attribute missing (listing available names), not callable, factory raised,
  and returned object missing `name` / `retrieve` / `answer`.
- `--system-kwargs` must be a JSON object; it is passed to the factory as keyword
  arguments and recorded in `RunConfig.system_kwargs`. Through the Python API,
  non-JSON values (for example a generator function) can be passed too; they then
  appear in `config.json` as their string form.
- When `run_live` gets a system object instead of a string, `RunConfig.system` is
  `object:<module>.<ClassName>`.

## Consequences

- Run commands from the directory that contains your plugin package (or install it).
- A local module can shadow an installed one with the same name; this is the same
  trade-off `python -m` makes.
