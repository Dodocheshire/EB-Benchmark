"""Decode discrete gripper actions without executing generated expressions."""
import ast
import json
import math


class InvalidManipulationPlan(ValueError):
    pass


def parse_manipulation_plan(output):
    try:
        document = json.loads(output) if isinstance(output, str) else output
        plan = document.get("executable_plan", document.get("properties", {}).get("executable_plan"))
        if isinstance(plan, str):
            plan = ast.literal_eval(plan)
        if not isinstance(plan, (list, tuple)):
            raise ValueError("executable_plan must be a list")
        actions = []
        for item in plan:
            if isinstance(item, (list, tuple)) and len(item) == 1 and isinstance(item[0], dict):
                item = item[0]
            values = item.get("action") if isinstance(item, dict) else item
            if isinstance(values, str):
                values = ast.literal_eval(values)
            if not isinstance(values, (list, tuple)) or len(values) != 7:
                raise ValueError("each action must contain exactly seven numbers")
            if any(type(v) not in (int, float) or not math.isfinite(v) or int(v) != v for v in values):
                raise ValueError("action values must be finite discrete numbers, not expressions or names")
            values = [int(v) for v in values]
            if any(not 0 <= v <= 100 for v in values[:3]):
                raise ValueError("position values must be in [0, 100]")
            if any(not 0 <= v < 120 for v in values[3:6]):
                raise ValueError("rotation values must be in [0, 119]")
            if values[6] not in (0, 1):
                raise ValueError("gripper must be 0 or 1")
            actions.append(values)
        return actions, document
    except (ValueError, TypeError, AttributeError, SyntaxError, OverflowError) as exc:
        raise InvalidManipulationPlan(str(exc)) from exc
