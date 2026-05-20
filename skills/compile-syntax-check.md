# Syntax Verification on PIDSMaker Baseline

When working in environments with restricted permissions or missing library dependencies (e.g. no PyTorch, no PyG, or missing packages), you can verify the syntactic validity of your python changes using python's built-in `py_compile` module.

This module compiles the python files to bytecode without executing them, validating syntax and checking for syntax errors, unresolved indents, etc.

## Verification Command

To compile and check python files:

```bash
/usr/bin/python3 -m py_compile <path-to-python-file1> <path-to-python-file2> ...
```

If the compilation succeeds, the command will complete silently with an exit code of `0`.
If there are any syntax errors, it will print them to stdout/stderr along with the line number.
