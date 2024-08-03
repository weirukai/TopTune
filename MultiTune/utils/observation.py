from openbox.core.base import Observation
from openbox.utils.constants import MAXINT, SUCCESS


class Observation_context(Observation):
    def __init__(self, config, objs, constraints=None, trial_state=SUCCESS, elapsed_time=None, context=None):
        super().__init__(config, objs, constraints, trial_state, elapsed_time)
        self.context = context
