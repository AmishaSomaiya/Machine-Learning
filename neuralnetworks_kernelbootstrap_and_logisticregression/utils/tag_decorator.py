from collections import defaultdict
import re
from typing import Callable, DefaultDict, Set, Tuple


class ProblemDecorator:
    def __init__(self):
        """
        ProblemDecorator allows for tagging of problem example, which later will be stripped of code.
        This enables invoke task to automatically substitute the code with "Your Code Goes Here" message.

        Please do not create new instances of the class but use problem defined below.
        """
        self.functions: DefaultDict[str, Set[Tuple[Callable, int]]] = defaultdict(set)
        self.regex_match_expression: str = "hw[0-9]-[A-B]"
        self.tag_regex: re.Pattern = re.compile(self.regex_match_expression)

    def tag(self, tag: str, start_line: int = 0) -> Callable:
        """Generates decorator that records a function, and passes it through.
        This is later used by invoke tasks to generate problem sets.
        It also allows for passing in start_line argument which keep the function code during generation of assignment zip.

        Args:
            tag (str): Tag of form "hw0", "hw1", ..., "hw9".
                Specifies which homework a problem belongs to.
            start_line (int, optional): Specifies which line of the function problem starts.
                This is useful if there is a starter code.
                Lines after and including `start_line` will be deleted, and substituted with "Your Code Goes Here" message.

        Returns:
            Callable: Decorator that tags a function in `self.functions`, and passes it through unchanged.
        """
        assert self.tag_regex.match(tag), f"Please make sure that tag matches regex {self.regex_match_expression}"

        def decorator(func):
            self.functions[tag].add((func, start_line))
            return func
        return decorator


problem = ProblemDecorator()
