from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker

class RandomStateChecker(BaseChecker):
    __implements__ = IAstroidChecker
    name = 'ml_random_state_checker'

    msgs = {
        'W9001': (
            'Random state not set in ML model',
            'no-random-state',
            'Ensure random_state is set for reproducibility.'
        ),
    }

    def visit_call(self, node):
        try:
            if hasattr(node.func, 'attrname') and 'Classifier' in node.func.attrname:
                if not any(k.arg == 'random_state' for k in node.keywords):
                    self.add_message('no-random-state', node=node)
        except AttributeError:
            pass

def register(linter):
    linter.register_checker(RandomStateChecker(linter))
