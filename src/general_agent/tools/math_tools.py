"""Safe math evaluator: parses the expression into an AST and only allows a whitelist of
numeric operations. Never use `eval()` on LLM-generated text.

Every operation is bounded *before* it runs (operand size, exponent, rounding digits,
AST size), because Python's big integers make innocent-looking expressions such as
`((10**999)**999)**999` or `round(5, -10**9)` hang the process.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool

MAX_EXPRESSION_LENGTH = 300
MAX_AST_NODES = 120
MAX_EXPONENT = 1000
MAX_RESULT_DIGITS = 400
MAX_ROUND_DIGITS = 20
MAX_FUNCTION_ARGS = 20

Number = int | float


class CalculatorError(ValueError):
    pass


def _too_large() -> CalculatorError:
    return CalculatorError(f"Result too large (over {MAX_RESULT_DIGITS} digits).")


def _digits(value: Number) -> float:
    """Approximate number of decimal digits in the integer part (cheap, no big-int math)."""
    if isinstance(value, int):
        return value.bit_length() * 0.30103
    return math.log10(abs(value)) + 1 if math.isfinite(value) and value else 0.0


def _check(value: Any) -> Number:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CalculatorError("Only real numbers are supported.")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorError("Result is not a finite number.")
    if _digits(value) > MAX_RESULT_DIGITS:
        raise _too_large()
    return value


def _safe_mul(a: Number, b: Number) -> Number:
    if _digits(a) + _digits(b) > MAX_RESULT_DIGITS + 1:  # checked BEFORE multiplying
        raise _too_large()
    return a * b


def _safe_pow(base: Number, exponent: Number) -> Number:
    if abs(exponent) > MAX_EXPONENT:
        raise CalculatorError(f"Exponent too large (max {MAX_EXPONENT}).")
    if base < 0 and not float(exponent).is_integer():
        raise CalculatorError("Fractional power of a negative number is not a real number.")
    if abs(base) > 1 and exponent > 0 and exponent * math.log10(abs(base)) > MAX_RESULT_DIGITS:
        raise _too_large()
    result: Number = base**exponent
    return result


def _safe_round(value: Number, ndigits: Number | None = None) -> Number:
    if ndigits is None:
        return round(value)
    if not isinstance(ndigits, int) or abs(ndigits) > MAX_ROUND_DIGITS:
        raise CalculatorError(f"round() digits must be an integer within ±{MAX_ROUND_DIGITS}.")
    return round(value, ndigits)


_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: _safe_mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": _safe_round,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}


def _eval(node: ast.AST) -> Number:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return _check(node.value)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        return _check(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _check(_UNARY_OPS[type(node.op)](_eval(node.operand)))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        if len(node.args) > MAX_FUNCTION_ARGS:
            raise CalculatorError("Too many function arguments.")
        return _check(_FUNCTIONS[node.func.id](*(_eval(arg) for arg in node.args)))
    raise CalculatorError(f"Unsupported expression element: {type(node).__name__}")


def safe_eval(expression: str) -> Number:
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError("Expression too long.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid syntax: {exc.msg}") from exc
    except (RecursionError, MemoryError) as exc:
        raise CalculatorError("Expression too deeply nested.") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise CalculatorError("Expression too complex.")
    try:
        return _eval(tree)
    except CalculatorError:
        raise
    except ZeroDivisionError as exc:
        raise CalculatorError("Division by zero.") from exc
    except (OverflowError, ValueError, TypeError, RecursionError) as exc:
        raise CalculatorError(str(exc) or exc.__class__.__name__) from exc


def _format(value: Number) -> str:
    if isinstance(value, int):
        return str(value)  # exact, no float rounding for big integers
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.10g}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression exactly. Supports + - * / // % **, parentheses,
    sqrt, log, log10, exp, sin, cos, tan, abs, round, min, max, floor, ceil, pi, e.
    Write percentages as multiplication, e.g. '15% of 2340' -> '0.15 * 2340'."""
    try:
        return _format(safe_eval(expression))
    except CalculatorError as exc:
        return f"Error: {exc}"
