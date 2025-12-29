# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FlowRL Ray Trainer that extends RayDAPOTrainer with DAPO's advanced features.
"""

from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer


class RayFlowRLTrainer(RayDAPOTrainer):
    """
    FlowRL trainer that inherits from DAPO trainer.

    This trainer uses all of DAPO's features:
    - Group filtering: Filters trajectories by reward variance
    - ReMAX advantage estimation: Generates baseline with greedy sampling
    - Batch balancing: Balances tokens across DP ranks
    - Efficient data handling

    The FlowRL-specific logic (trajectory balance objectives) is implemented
    in the FlowRLActor, which is injected via FlowRLActorRolloutRefWorker.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
