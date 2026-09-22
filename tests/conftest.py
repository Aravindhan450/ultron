import os

import pytest

from ultron.core.agents.security import get_security
from ultron.core.tools.paths import set_active_project_dir


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global tool paths and security mode before and after each test."""
    old_ws = os.environ.pop("ULTRON_WORKSPACE", None)
    try:
        os.remove("/tmp/ultron_pt_test.txt")
    except OSError:
        pass
    set_active_project_dir(None)
    get_security().mode = "interactive"
    yield
    set_active_project_dir(None)
    get_security().mode = "interactive"
    try:
        os.remove("/tmp/ultron_pt_test.txt")
    except OSError:
        pass
    if old_ws is not None:
        os.environ["ULTRON_WORKSPACE"] = old_ws

