import pytest

MARKS = ["feature_data", "model_dev", "ml_infra", "monitoring"]
PTS = dict.fromkeys(MARKS, 25)

def pytest_configure(config):
    for m in MARKS:
        config.addinivalue_line("markers", m)

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    MARKS = ["feature_data", "model_dev", "ml_infra", "monitoring"]
    PTS = dict.fromkeys(MARKS, 25)

    passed = terminalreporter.stats.get("passed", [])
    failed = terminalreporter.stats.get("failed", [])

    # collect pass
    cats_pass = {m for rep in passed  for m in MARKS if m in rep.keywords}
    # collect fail
    cats_fail = {m for rep in failed  for m in MARKS if m in rep.keywords}

    # score: a category gets full score iff **All Tests in Category Pass**
    score = sum(PTS[c] for c in MARKS if c in cats_pass and c not in cats_fail)

    terminalreporter.write_sep("=", f"ML Test Score: {score}/100")
