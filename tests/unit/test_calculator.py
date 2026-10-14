import math

import pytest

from general_agent.tools.math_tools import CalculatorError, calculator, safe_eval


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("0.15 * 2340", 351),
        ("2 ** 10", 1024),
        ("-5 + 2", -3),
        ("17 // 5", 3),
        ("17 % 5", 2),
        ("sqrt(16)", 4),
        ("round(pi, 2)", 3.14),
        ("max(1, 7, 3)", 7),
        ("log(e)", 1),
    ],
)
def test_safe_eval_valid(expr: str, expected: float) -> None:
    assert math.isclose(safe_eval(expr), expected)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",
        "open('x')",
        "a + 1",
        "[1, 2]",
        "(lambda: 1)()",
        "2 ** 100000",
        "1 if True else 2",
    ],
)
def test_safe_eval_rejects_unsafe(expr: str) -> None:
    with pytest.raises(CalculatorError):
        safe_eval(expr)


def test_division_by_zero_is_reported() -> None:
    with pytest.raises(CalculatorError, match="zero"):
        safe_eval("1 / 0")


def test_tool_returns_error_text_instead_of_raising() -> None:
    out = calculator.invoke({"expression": "1/0"})
    assert out.startswith("Error")
    assert calculator.invoke({"expression": "6*7"}) == "42"
