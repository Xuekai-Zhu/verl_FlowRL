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
Simplified FlowRL FSDP Worker - uses standard veRL worker with FlowRLActor.

This is a minimal override that just replaces the actor class.
All FSDP setup, model loading, and optimization is handled by veRL's standard worker.
"""

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from recipe.flowrl.flowrl_actor import FlowRLActor


class FlowRLActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    FlowRL version of ActorRolloutRefWorker.

    This worker only overrides init_model() to use FlowRLActor instead of
    DataParallelPPOActor. Everything else (FSDP setup, checkpointing, etc.)
    is inherited from the standard veRL worker.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Override init_model to use FlowRLActor instead of DataParallelPPOActor."""
        # Call parent's init_model to set up the FSDP model
        super().init_model()

        # Replace the actor with FlowRLActor if this worker is an actor
        if self._is_actor:
            if self.rank == 0:
                print(f"[FlowRL] Replacing DataParallelPPOActor with FlowRLActor")

            # Convert actor config to dataclass
            actor_cfg = omega_conf_to_dataclass(self.config.actor)

            # Create FlowRLActor with trajectory balance loss
            self.actor = FlowRLActor(
                config=actor_cfg,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer
            )
