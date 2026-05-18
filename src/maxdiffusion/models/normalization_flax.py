"""
Copyright 2024 Google LLC

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

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import nnx


class AdaLayerNormContinuous(nn.Module):
  embedding_dim: int
  elementwise_affine: bool = True
  eps: float = 1e-5
  bias: bool = True
  norm_type: str = "layer_norm"
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  @nn.compact
  def __call__(self, x, conditioning_embedding):
    assert self.norm_type == "layer_norm"
    # Dense output is split immediately into (shift, scale); leaving the output
    # dim unsharded means the split is a free local op rather than forcing an
    # all-gather along the tensor axis (chunk size < per-device shard).
    emb = nn.Dense(
        self.embedding_dim * 2,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", None)),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
        use_bias=self.bias,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )(nn.silu(conditioning_embedding))
    emb = nn.with_logical_constraint(emb, ("activation_batch", None))
    emb = emb[:, None, :]
    shift, scale = jnp.split(emb, 2, axis=-1)
    x = nn.LayerNorm(epsilon=self.eps, use_bias=self.elementwise_affine, use_scale=self.elementwise_affine)(x)
    x = (1 + scale) * x + shift
    return x


class AdaLayerNormZero(nn.Module):
  embedding_dim: int
  norm_type: str = "layer_norm"
  bias: bool = True
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  @nn.compact
  def __call__(self, x, emb):
    emb = nn.silu(emb)
    
    # Project to 6 * dim, split into (shift_msa, scale_msa, gate_msa, shift_mlp,
    # scale_mlp, gate_mlp). Output dim is left unsharded — sharding on "mlp" then
    # splitting into chunks smaller than the per-device shard forces an
    # all-gather along the tensor axis.
    emb = nn.Dense(
        6 * self.embedding_dim,
        use_bias=self.bias,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", None)),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
        name="lin",
    )(emb)
    emb = nn.with_logical_constraint(emb, ("activation_batch", None))
    emb = emb[:, None, :]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(emb, 6, axis=-1)
    
    if self.norm_type == "layer_norm":
      # Fused mathematical reduction loop
      mean = jnp.mean(x, axis=-1, keepdims=True)
      variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
      inv_std = jax.lax.rsqrt(variance + 1e-6)
      
      x = (x - mean) * inv_std * (1.0 + scale_msa) + shift_msa
    else:
      raise ValueError(f"Unsupported `norm_type` ({self.norm_type}) provided.")
      
    return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
  embedding_dim: int
  norm_type: str = "layer_norm"
  bias: bool = True
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  @nn.compact
  def __call__(self, x, emb):
    emb = nn.silu(emb)
    
    # See AdaLayerNormZero — output is split into 3, so leave the output dim
    # unsharded to keep the split as a free local op.
    emb = nn.Dense(
        3 * self.embedding_dim,
        use_bias=self.bias,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", None)),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
        name="lin",
    )(emb)
    emb = nn.with_logical_constraint(emb, ("activation_batch", None))
    emb = emb[:, None, :]
    shift_msa, scale_msa, gate_msa = jnp.split(emb, 3, axis=-1)
    
    if self.norm_type == "layer_norm":
      # Fused optimization math keeping exact pretrained weight compatibility
      mean = jnp.mean(x, axis=-1, keepdims=True)
      variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
      inv_std = jax.lax.rsqrt(variance + 1e-6)
      
      x = (x - mean) * inv_std * (1.0 + scale_msa) + shift_msa
    else:
      raise ValueError(f"Unsupported `norm_type` ({self.norm_type}) provided.")
      
    return x, gate_msa


class FP32LayerNorm(nnx.Module):

  def __init__(self, rngs: nnx.Rngs, dim: int, eps: float, elementwise_affine: bool):
    self.layer_norm = nnx.LayerNorm(
        rngs=rngs,
        num_features=dim,
        epsilon=eps,
        use_bias=elementwise_affine,
        use_scale=elementwise_affine,
        param_dtype=jnp.float32,
        dtype=jnp.float32,
    )

  def __call__(self, inputs: jax.Array) -> jax.Array:
    origin_dtype = inputs.dtype
    return self.layer_norm(inputs.astype(dtype=jnp.float32)).astype(dtype=origin_dtype)
