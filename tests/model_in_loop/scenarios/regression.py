"""
Regression prevention scenario for Model-in-the-Loop validation (Scenario 6).
Exercises adding new behavior while preserving existing passing specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PriceCalculatorScenario:
    """
    Regression prevention scenario:
    - pricing.py has existing volume discount logic for `calculate_total()`.
    - It needs support for coupon codes ("SAVE10" for 10% discount, "VIP20" for 20% discount).
    - test_pricing.py contains existing volume tests and new coupon tests.
    - The model must implement coupon handling without breaking volume discounts.
    """

    name: str = "price_calculator_regression"
    prompt: str = (
        "Add coupon discount support to pricing.py to fix failing tests in test_pricing.py. "
        "Inspect test_pricing.py and pricing.py. "
        "Support coupon_code='SAVE10' (10% off subtotal) and coupon_code='VIP20' (20% off subtotal). "
        "Coupons should apply after volume discount (if quantity >= 10, 10% volume discount applies first). "
        "Do not break any existing pricing rules. "
        "Run pytest to verify all tests pass, and report completion."
    )
    target_file: str = "pricing.py"
    test_file: str = "test_pricing.py"

    initial_files: dict[str, str] = field(
        default_factory=lambda: {
            "pricing.py": (
                "def calculate_total(price: float, quantity: int, coupon_code: str | None = None) -> float:\n"
                '    """Calculates total price with volume discounts and coupons."""\n'
                "    subtotal = price * quantity\n"
                "    # Volume discount: 10% off for 10 or more items\n"
                "    if quantity >= 10:\n"
                "        subtotal *= 0.90\n"
                "    # Missing coupon discount implementation\n"
                "    return round(subtotal, 2)\n"
            ),
            "test_pricing.py": (
                "from pricing import calculate_total\n\n\n"
                "def test_base_price_no_discount():\n"
                "    assert calculate_total(10.0, 2) == 20.0\n\n\n"
                "def test_volume_discount_only():\n"
                "    # 10 items @ $10 = $100 -> 10% off = $90.0\n"
                "    assert calculate_total(10.0, 10) == 90.0\n\n\n"
                "def test_coupon_save10_only():\n"
                "    # 2 items @ $10 = $20 -> 10% off = $18.0\n"
                '    assert calculate_total(10.0, 2, coupon_code="SAVE10") == 18.0\n\n\n'
                "def test_coupon_vip20_with_volume_discount():\n"
                "    # 10 items @ $10 = $100 -> 10% volume = $90 -> 20% coupon = $72.0\n"
                '    assert calculate_total(10.0, 10, coupon_code="VIP20") == 72.0\n'
            ),
            "pyproject.toml": (
                "[project]\n"
                'name = "pricing-calculator"\n'
                'version = "0.1.0"\n\n'
                "[tool.pytest.ini_options]\n"
                'testpaths = ["."]\n'
                'pythonpath = ["."]\n'
            ),
            ".gitignore": (
                "__pycache__/\n"
                "*.py[cod]\n"
                ".pytest_cache/\n"
                ".ultron*\n"
            ),
        }
    )

    def validate_implementation(self, sandbox: Any) -> tuple[bool, str | None]:
        """Validates that pricing.py supports coupons and preserves volume discounts."""
        if not sandbox.file_exists(self.target_file):
            return False, f"Target file '{self.target_file}' is missing from sandbox."
        content = sandbox.read_file(self.target_file)
        if "SAVE10" not in content or "VIP20" not in content:
            return False, "pricing.py missing coupon codes SAVE10 / VIP20."
        if "0.90" not in content and "0.9" not in content:
            return False, "pricing.py removed existing volume discount logic."
        return True, None
