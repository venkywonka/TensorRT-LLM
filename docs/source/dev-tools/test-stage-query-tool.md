# Test Stage Query Tool

This document proposes a small utility to identify which Jenkins stage runs a given integration test.

## Goals
- Quickly map a test case to its Jenkins stage name.
- Provide a simple command-line interface.
- Reduce guesswork when using `/bot run --stage-list`.

## Implementation ideas
1. Parse the YAML files in `tests/integration/test_lists/test-db` to collect test entries with their `stage` and `backend` tags.
2. Parse `jenkins/L0_Test.groovy` to map YAML files to stage names and shard counts.
3. Expose a command:
   ```bash
   python tools/query_stage.py triton_server/test_triton.py::test_gpt_ib_ptuning[gpt-ib-ptuning]
   ```
   Which would print:
   ```
   A100X-Triton-Python-[Post-Merge]-1
   A100X-Triton-Python-[Post-Merge]-2
   ```
4. Cache the parsed results so repeated queries are fast.
5. Optionally generate a ready-to-use `/bot run --stage-list` command.

## Future work
- Extend the tool to verify new test entries or list all tests in a given stage.
