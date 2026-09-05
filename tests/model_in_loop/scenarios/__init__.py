"""
MITL benchmark test scenarios.
"""

from tests.model_in_loop.scenarios.bug_fix import CalculatorBugFixScenario
from tests.model_in_loop.scenarios.multi_case import DurationFormatterScenario
from tests.model_in_loop.scenarios.multi_file import ConfigServiceScenario
from tests.model_in_loop.scenarios.regression import PriceCalculatorScenario
from tests.model_in_loop.scenarios.repair_scenario import SlugifyRepairScenario
from tests.model_in_loop.scenarios.syntax_import import SyntaxImportRecoveryScenario

__all__ = [
    "CalculatorBugFixScenario",
    "ConfigServiceScenario",
    "DurationFormatterScenario",
    "PriceCalculatorScenario",
    "SlugifyRepairScenario",
    "SyntaxImportRecoveryScenario",
]
