"""Compatibility entry point for corrected v2; historical artifacts are read-only.

Use task3_corrected.py for the explicit all/checks/development/finalists/figures stages.
"""
from task3_corrected import main

if __name__ == "__main__":
    main(default_stage='figures')
