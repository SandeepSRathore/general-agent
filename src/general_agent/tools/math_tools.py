"""Safe math evaluator: parses the expression into an AST and only allows a whitelist of
numeric operations. Never use `eval()` on LLM-generated text."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool

MAX_EXPRESSION_LENGTH = 300
MAX_EXPONENT = 1000
# Bound every intermediate result: big-int arithmetic on huge numbers (e.g. nested powers
# like ((10**999)**999)**999) would otherwise exhaust CPU/memory.
MAX_RESULT_DIGITS = 400


class CalculatorError(ValueError):
    pass


def _check_size(value: Any) -> Any:
    if isinstance(value, int) and value.bit_length() > MAX_RESULT_DIGITS * 3.33:
        raise CalculatorError(f"Result too large (over {MAX_RESULT_DIGITS} digits).")
    return value


def _safe_pow(base: Any, exponent: Any) -> Any:
    if abs(exponent) > MAX_EXPONENT:
        raise CalculatorError(f"Exponent too large (max {MAX_EXPONENT}).")
    # Estimate digits of the result *before* computing it.
    if abs(base) > 1 and exponent > 0 and exponent * math.log10(abs(base)) > MAX_RESULT_DIGITS:
        raise CalculatorError(f"Result too large (over {MAX_RESULT_DIGITS} digits).")
    return base**exponent


_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
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
    "round": round,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        return _check_size(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        if len(node.args) > 20:
            raise CalculatorError("Too many function arguments.")
        return _check_size(_FUNCTIONS[node.func.id](*(_eval(arg) for arg in node.args)))
    raise CalculatorError(f"Unsupported expression element: {ast.dump(node)[:60]}")


def safe_eval(expression: str) -> float:
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError("Expression too long.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid syntax: {exc.msg}") from exc
    try:
        result = _eval(tree)
    except ZeroDivisionError as exc:
        raise CalculatorError("Division by zero.") from exc
    except (OverflowError, ValueError, TypeError) as exc:
        if isinstance(exc, CalculatorError):
            raise
        raise CalculatorError(str(exc)) from exc
    return float(result) if isinstance(result, int | float) else result


def _format(value: float) -> str:
    if float(value).is_integer():
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
