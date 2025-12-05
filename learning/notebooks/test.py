import os
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2") 
# os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp

print("JAX devices:", jax.devices())

@jax.jit
def f(x):
    return x @ x.T + 1.0

x = jnp.ones((1024, 1024))
y = f(x)
print("done, y shape:", y.shape)
