"""Apply the complete isolated candidate before pytest imports any test module."""
import json
from candidate_bootstrap import provenance


def pytest_report_header(config):
    return "candidate runtime import provenance: " + json.dumps(provenance(), sort_keys=True)


def pytest_sessionfinish(session, exitstatus):
    # User-visible output, not a generated evidence file or production mutation.
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line("CANDIDATE_IMPORT_PROVENANCE=" + json.dumps(provenance(), sort_keys=True))
