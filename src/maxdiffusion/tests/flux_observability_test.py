"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import unittest
from unittest.mock import Mock
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
from absl.testing import absltest
from maxdiffusion import pyconfig, max_utils
from maxdiffusion.trainers.flux_trainer import FluxTrainer, FLUX_STATE_KEY
from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformer2DModel
from maxdiffusion.schedulers.scheduling_euler_discrete_flax import FlaxEulerDiscreteScheduler
from flax.training import train_state
import optax

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class FluxObservabilityTest(unittest.TestCase):
  """Testcase to verify and inspect hitesy-* shape and architecture logs without requiring a TPU."""

  def test_flux_compile_train_step_logs(self):
    # 1. Initialize pyconfig with base flux dev config
    pyconfig.initialize([
        None,
        os.path.join(THIS_DIR, "..", "configs", "base_flux_dev.yml"),
        "run_name=flux_observability_test",
        "per_device_batch_size=1",
        "max_sequence_length=256",
    ], unittest=True)
    config = pyconfig.config

    # 2. Instantiate Trainer
    trainer = FluxTrainer(config)
    
    # Use a simple CPU mesh for testing on any environment
    devices_array = np.array([jax.devices()[0]]).reshape((1, 1, 1, 1))
    mesh = Mesh(devices_array, ("data", "fsdp", "context", "tensor"))
    trainer.mesh = mesh

    # 3. Create Mock Pipeline with real FluxTransformer2DModel using a config dictionary (bypasses HF Hub)
    pipeline = Mock()
    pipeline.config = config
    
    model_config = {
        "patch_size": 1,
        "in_channels": 64,
        "num_layers": 2,
        "num_single_layers": 2,
        "attention_head_dim": 128,
        "num_attention_heads": 4,
        "joint_attention_dim": 4096,
        "pooled_projection_dim": 768,
        "guidance_embeds": True,
        "axes_dims_rope": [16, 56, 56],
        "attention_kernel": "dot_product",
        "remat_policy": "MATMUL_WITHOUT_BATCH",
    }
    transformer = FluxTransformer2DModel.from_config(
        model_config,
        mesh=mesh,
    )
    pipeline.flux = transformer
    
    scheduler_config = {
        "_class_name": "FlaxEulerDiscreteScheduler",
        "prediction_type": "epsilon",
        "rescale_zero_terminal_snr": False,
        "timestep_spacing": "trailing"
    }
    noise_scheduler = FlaxEulerDiscreteScheduler.from_config(scheduler_config)
    noise_scheduler_state = noise_scheduler.create_state()
    noise_scheduler_state = noise_scheduler.set_timesteps(
        state=noise_scheduler_state, num_inference_steps=config.num_inference_steps, timestep_spacing="flux"
    )
    pipeline.scheduler = noise_scheduler

    # 4. Create abstract TrainState (0 RAM, instant)
    rng = jax.random.PRNGKey(0)
    with mesh:
      eval_params = transformer.init_weights(
          rngs=rng, max_sequence_length=config.max_sequence_length, eval_only=True
      )
      
      tx = optax.sgd(1e-5)
      abstract_ts = train_state.TrainState.create(apply_fn=transformer.apply, params=eval_params, tx=tx)
      abstract_ts = jax.tree_util.tree_map(lambda x: jnp.asarray(x) if isinstance(x, (int, float)) else x, abstract_ts)

    train_states = {
        FLUX_STATE_KEY: abstract_ts,
        "scheduler": noise_scheduler_state,
    }
    
    state_shardings = {
        "flux_state_shardings": jax.tree_util.tree_map(
            lambda x: jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()), abstract_ts
        )
    }
    data_shardings = trainer.get_data_shardings()

    # 5. Call compile_train_step to trigger all lowering and hitesy-* shape/architecture logs
    p_train_step = trainer.compile_train_step(pipeline, None, train_states, state_shardings, data_shardings)
    self.assertIsNotNone(p_train_step)


if __name__ == "__main__":
  absltest.main()
