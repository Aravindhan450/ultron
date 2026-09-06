#!/bin/bash
export ULTRON_LLAMA_SERVER_CONTEXT_LENGTH=16384
export ULTRON_LLAMA_SERVER_GPU_LAYERS=35

echo "Running 1. Calculator Bug Fix"
.venv/bin/pytest tests/model_in_loop/test_real_coding_agent.py::test_real_model_fixes_calculator_bug -v -s -m mitl > scratch/calc.log 2>&1
echo "Running 2. Slugify Repair"
.venv/bin/pytest tests/model_in_loop/test_real_coding_agent.py::test_real_model_repair_slugify_bug -v -s -m mitl > scratch/slugify.log 2>&1
echo "Running 3. Duration Multi-Case"
.venv/bin/pytest tests/model_in_loop/test_real_coding_agent.py::test_real_model_multi_case_duration -v -s -m mitl > scratch/duration.log 2>&1
echo "Running 4. Multi-File Config"
.venv/bin/pytest tests/model_in_loop/test_real_coding_agent.py::test_real_model_multi_file_config -v -s -m mitl > scratch/config.log 2>&1
echo "Running 5. Syntax/Import Recovery"
.venv/bin/pytest tests/model_in_loop/test_real_coding_agent.py::test_real_model_syntax_import_recovery -v -s -m mitl > scratch/syntax.log 2>&1
echo "Running 6. Regression Prevention"
.venv/bin/pytest tests/model_in_loop/test_real_coding_agent.py::test_real_model_regression_prevention -v -s -m mitl > scratch/regression.log 2>&1

echo "Done"
