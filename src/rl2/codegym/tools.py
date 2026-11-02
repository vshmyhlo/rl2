"""Extract public action schemas without importing or executing environment code."""

import ast
from typing import Any

Schema = dict[str, Any]


def annotation_schema(node: ast.expr | None) -> Schema:
    if node is None:
        return {}
    if isinstance(node, ast.Constant):
        if node.value is None:
            return {"type": "null"}
        if isinstance(node.value, str):
            try:
                return annotation_schema(ast.parse(node.value, mode="eval").body)
            except SyntaxError:
                return {}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return {"anyOf": [annotation_schema(node.left), annotation_schema(node.right)]}
    base = node.value if isinstance(node, ast.Subscript) else node
    name = ast.unparse(base).rsplit(".", 1)[-1]
    elements = []
    if isinstance(node, ast.Subscript):
        elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    scalars = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "None": "null"}
    if name in scalars:
        return {"type": scalars[name]}
    if name in {"Optional", "Union"}:
        options = [annotation_schema(item) for item in elements]
        if name == "Optional":
            options.append({"type": "null"})
        return {"anyOf": options}
    if name in {"list", "List", "set", "Set", "Sequence", "tuple", "Tuple"}:
        result: Schema = {"type": "array"}
        if (
            name.lower() == "tuple"
            and elements
            and not (len(elements) == 2 and isinstance(elements[1], ast.Constant) and elements[1].value is Ellipsis)
        ):
            result.update(
                prefixItems=[annotation_schema(item) for item in elements],
                minItems=len(elements),
                maxItems=len(elements),
            )
        else:
            result["items"] = annotation_schema(elements[0]) if elements else {}
        return result
    if name in {"dict", "Dict", "Mapping"}:
        return {
            "type": "object",
            "additionalProperties": annotation_schema(elements[1]) if len(elements) == 2 else True,
        }
    return {}


def parameter_reads(nodes: list[ast.stmt]) -> dict[str, bool]:
    """Collect the JSON keys used by a dispatch branch and whether they are required."""
    reads: dict[str, bool] = {}
    descendants = [node for statement in nodes for node in ast.walk(statement)]
    key_lists = {
        target.id: node.value.elts
        for node in descendants
        if isinstance(node, ast.Assign)
        and isinstance(node.value, (ast.List, ast.Tuple))
        and all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in node.value.elts)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    loop_keys = {
        node.target.id: node.iter.elts if isinstance(node.iter, (ast.List, ast.Tuple)) else key_lists[node.iter.id]
        for node in descendants
        if isinstance(node, ast.comprehension)
        and isinstance(node.target, ast.Name)
        and (
            isinstance(node.iter, (ast.List, ast.Tuple))
            or (isinstance(node.iter, ast.Name) and node.iter.id in key_lists)
        )
    }
    for node in descendants:
        key = None
        required = True
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "params":
            key = node.slice
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "params"
            and node.func.attr == "get"
            and node.args
        ):
            key = node.args[0]
            required = False
        elif (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.In, ast.NotIn))
            and isinstance(node.comparators[0], ast.Name)
            and node.comparators[0].id == "params"
        ):
            key = node.left
        if key is not None:
            keys = loop_keys.get(key.id, []) if isinstance(key, ast.Name) else [key]
            if not keys or not all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in keys):
                raise ValueError("Unsupported dynamic CodeGym parameter key")
            for item in keys:
                reads[item.value] = reads.get(item.value, False) or required
    return reads


def extract_tools(code: str, class_name: str) -> list[Schema]:
    tree = ast.parse(code)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    mappings = [
        node.value
        for node in ast.walk(methods["__init__"])
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "func_mapping"
            for target in node.targets
        )
    ]
    if len(mappings) != 1 or not isinstance(mappings[0], ast.Dict):
        raise ValueError(f"{class_name}: expected a literal func_mapping")
    constants = {
        ast.unparse(target): node.value
        for node in ast.walk(methods["__init__"])
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
    }
    mapping = {
        key.value: value
        for key, value in zip(mappings[0].keys, mappings[0].values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    if len(mapping) != len(mappings[0].keys) or not mapping.keys() <= methods.keys():
        raise ValueError(f"{class_name}: action has no matching method")

    def resolve(node: ast.expr) -> str:
        if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "self.func_mapping":
            node = mapping[ast.literal_eval(node.slice)]
        return ast.dump(constants.get(ast.unparse(node), node))

    branches: dict[str, ast.If] = {}
    for node in ast.walk(methods["step"]):
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "action_code"
        ):
            continue
        selector = resolve(node.test.comparators[0])
        if selector in branches:
            raise ValueError(f"{class_name}: duplicate dispatch branch")
        branches[selector] = node
    missing = [name for name, value in mapping.items() if resolve(value) not in branches]
    fallback: list[ast.stmt] = []
    if len(missing) == 1:
        # A single remaining action may be handled by the final else/elif parameter check.
        tails = [
            branch.orelse
            for branch in branches.values()
            if branch.orelse
            and not any(
                isinstance(node, ast.Name) and node.id == "action_code"
                for statement in branch.orelse
                for node in ast.walk(statement)
            )
        ]
        if len(tails) == 1:
            fallback = tails[0]
    tools: list[Schema] = []
    for name, value in mapping.items():
        method = methods[name]
        branch = branches.get(resolve(value))
        body = branch.body if branch is not None else fallback
        if not body:
            raise ValueError(f"{class_name}.{name}: expected one dispatch branch")
        try:
            reads = parameter_reads(body)
        except ValueError as error:
            raise ValueError(f"{class_name}.{name}: {error}") from error
        args = {
            arg.arg: arg
            for arg in [*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs]
            if arg.arg != "self"
        }
        # Only annotate matching wire keys; renamed arguments remain unrestricted.
        properties = {name: annotation_schema(args[name].annotation) if name in args else {} for name in reads}
        for statement in body:
            for node in ast.walk(statement):
                if (
                    isinstance(node, ast.Call)
                    and ast.unparse(node.func) in {"json.loads", "ast.literal_eval"}
                    and node.args
                    and isinstance(node.args[0], ast.Subscript)
                    and isinstance(node.args[0].value, ast.Name)
                    and node.args[0].value.id == "params"
                ):
                    key = ast.literal_eval(node.args[0].slice)
                    properties[key] = {"type": "string", "description": "Serialized value decoded by the environment."}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": ast.get_docstring(method) or name,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": [name for name, required in reads.items() if required],
                        "additionalProperties": False,
                    },
                },
            }
        )
    if not tools:
        raise ValueError(f"{class_name}: no public actions")
    return tools
