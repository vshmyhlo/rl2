"""Bounded arithmetic for model-generated calculator requests."""

import ast
import operator

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def calculate(expression: str) -> str:
    """Return arithmetic output or a tool error; never execute Python code."""

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = evaluate(node.operand)
            value = -operand if isinstance(node.op, ast.USub) else operand
        elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or (left < 0 and int(right) != right)):
                raise ValueError("unsupported exponent")
            value = OPERATORS[type(node.op)](left, right)
        else:
            raise ValueError("use numbers, parentheses, and + - * / // % ** only")
        if abs(value) > 1e100:
            raise ValueError("magnitude exceeds 1e100")
        return value

    try:
        if len(expression) > 256:
            raise ValueError("expression exceeds 256 characters")
        tree = ast.parse(expression.strip(), mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise ValueError("expression is too complex")
        result = evaluate(tree.body)
        return str(result).removesuffix(".0")
    except (SyntaxError, ValueError, ArithmeticError) as error:
        return f"Error: {error.msg if isinstance(error, SyntaxError) else error}"
