import copy
import sys
import pdb
import numpy as np
import time
import copy

from ConfigSpace import Configuration
from tqdm import tqdm
from openbox.optimizer.parallel_smbo import pSMBO
sys.path.append('..')
from openbox.optimizer.generic_smbo import SMBO
from .bo import BO


class OtterTune(BO):
    def __init__(self, surrogate_type, **kwargs):
        super().__init__(**kwargs)
        self.model = pSMBO(
            self.evaluate,
            self.embedding_space,
            batch_size=4,
            parallel_strategy='sync',
            num_objs=1,
            num_constraints=0,
            surrogate_type=surrogate_type,  # 原来的是context_prf,但在后面没有该模型，于是改为prf
            acq_optimizer_type='local_random',
            initial_runs=self.init_runs,
            init_strategy='random',
            task_id=self.task_id,
            time_limit_per_trial=1000,
            random_state=self.random_state
        )

        if not self.cost_aware:
            self.model.max_iterations = 0
        else:
            self.model.runtime_limit = 0
            self.model.budget_left = 0
        self.config = []

    def Suggest(self):
        if len(self.config) == 0:
            self.config = self.model.config_advisor.get_suggestions()
        config = self.config[0]
        self.config.pop(0)
        real_config = self.input_space_adapter.unproject_point(config)
        new_config = {}
        for k, v in real_config.items():
            if self.arm_type == 'knob1' and k.split('.')[1] in self.db.knob_details_gp.keys():
                new_config[k] = v
            elif self.arm_type == 'knob2' and k.split('.')[1] in self.db.knob_details_smac.keys():
                new_config[k] = v
        real_config = Configuration(self.setup_config_space(self.arm_type, self.db), new_config)
        return config, real_config

    def UpdatePolicy(self, observation):
        self.model.config_advisor.update_observation(observation)
