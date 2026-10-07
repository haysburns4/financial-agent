"""Setup tooling: the settings spec and .env editing.

Nothing in this package may import src.config, directly or transitively.
Settings() is built at import time and raises when the E-Trade keys are
missing, which is exactly the state a first-run setup starts from.
"""
