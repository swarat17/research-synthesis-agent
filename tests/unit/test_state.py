import operator
from typing import get_args, get_type_hints

from src.graph.state import ResearchState


def test_errors_field_uses_add_reducer():
    hints = get_type_hints(ResearchState, include_extras=True)
    errors_hint = hints["errors"]
    args = get_args(errors_hint)
    # Annotated[list, operator.add] → args = (list, operator.add)
    assert len(args) == 2, "errors should be Annotated with two args"
    assert args[1] is operator.add, "errors reducer must be operator.add"
