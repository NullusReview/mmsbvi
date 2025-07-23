# Unscented Kalman Filter (UKF) Usage Guide

This document provides usage examples for the refactored UKF implementation, which supports arbitrary nonlinear systems with GPU optimization and numerical stability guarantees.

## Quick Start

### Basic Import

```python
from src.baselines.ukf import GenericUKF, create_pendulum_ukf, UKFConfig
import jax.numpy as jnp
```

## Example 1: Pendulum System (Convenience Function)

The simplest way to use UKF for a pendulum system:

```python
# Create pendulum UKF using convenience function
pendulum_ukf = create_pendulum_ukf(
    dt=0.05,           # Time step
    g=9.81,           # Gravitational acceleration
    L=1.0,            # Pendulum length
    gamma=0.2,        # Damping coefficient
    process_noise_std=0.1,  # Process noise standard deviation
    obs_noise_std=0.05      # Observation noise standard deviation
)

# Generate some sample observations (angle measurements)
T = 50
observations = jnp.array([[0.5 * jnp.cos(0.1 * t)] for t in range(T)])

# Initial conditions
initial_mean = jnp.array([0.5, 0.0])  # [angle, angular_velocity]
initial_cov = jnp.diag(jnp.array([0.1, 0.5]))

# Run filtering and smoothing
result = pendulum_ukf.filter_and_smooth(observations, initial_mean, initial_cov)

# Extract results
print(f"Total log-likelihood: {result.total_log_likelihood}")
print(f"Runtime: {result.runtime:.4f} seconds")

# Extract state estimates
final_state = result.smoothed_states[-1]
print(f"Final angle estimate: {final_state.mean[0]:.4f}")
print(f"Final angular velocity estimate: {final_state.mean[1]:.4f}")
```

## Example 2: Custom Linear System

Creating a custom linear system:

```python
# Define a 2D linear system: x_{k+1} = A * x_k + w_k
A = jnp.array([[0.9, 0.1], [0.0, 0.8]])
H = jnp.array([[1.0, 0.0]])  # Observe only first state

# Noise covariance matrices
Q = 0.1 * jnp.eye(2)  # Process noise
R = jnp.array([[0.05]])  # Observation noise

# Define system functions
def linear_transition(x):
    return A @ x

def linear_observation(x):
    return H @ x

# Create UKF instance
linear_ukf = GenericUKF(
    state_transition_fn=linear_transition,
    observation_fn=linear_observation,
    process_noise_cov=Q,
    obs_noise_cov=R,
    config=UKFConfig(alpha=0.001, beta=2.0, kappa=0.0)
)

# Generate synthetic data
T = 30
true_states = []
observations = []

x = jnp.array([2.0, -1.0])
for t in range(T):
    x = A @ x + 0.1 * jnp.array([0.1, 0.05])  # Add some process noise
    obs = H @ x + 0.05 * 0.1  # Add observation noise
    
    true_states.append(x)
    observations.append(obs)

observations = jnp.array(observations)

# Run UKF
initial_mean = jnp.array([0.0, 0.0])
initial_cov = jnp.eye(2)

result = linear_ukf.filter_and_smooth(observations, initial_mean, initial_cov)

# Analyze results
estimated_states = jnp.array([state.mean for state in result.smoothed_states])
rmse = jnp.sqrt(jnp.mean((estimated_states - jnp.array(true_states))**2))
print(f"RMSE: {rmse:.4f}")
```

## Example 3: Custom Nonlinear System

Creating a custom nonlinear system with growth dynamics:

```python
def nonlinear_growth_transition(x):
    """Nonlinear growth model: x_{k+1} = x_k + 0.1 * x_k^1.2"""
    return x + 0.1 * jnp.sign(x) * jnp.power(jnp.abs(x), 1.2)

def square_observation(x):
    """Observe squared state"""
    return jnp.array([x[0]**2])

# Noise covariances
Q = jnp.array([[0.01]])  # Small process noise
R = jnp.array([[0.1]])   # Observation noise

# Create UKF
nonlinear_ukf = GenericUKF(
    state_transition_fn=nonlinear_growth_transition,
    observation_fn=square_observation,
    process_noise_cov=Q,
    obs_noise_cov=R
)

# Simulate system
T = 25
x = jnp.array([1.0])
observations = []

for t in range(T):
    x = nonlinear_growth_transition(x) + 0.1 * 0.01  # Add process noise
    obs = square_observation(x) + 0.1 * 0.1  # Add observation noise
    observations.append(obs)

observations = jnp.array(observations)

# Run UKF
initial_mean = jnp.array([0.8])
initial_cov = jnp.array([[0.1]])

result = nonlinear_ukf.filter_and_smooth(observations, initial_mean, initial_cov)
print(f"Nonlinear system log-likelihood: {result.total_log_likelihood:.2f}")
```

## Example 4: Batch Processing

Processing multiple sequences in parallel:

```python
# Create multiple pendulum sequences with different initial conditions
batch_size = 3
T = 20

# Generate batch data
batch_observations = []
batch_initial_means = []
batch_initial_covs = []

for i in range(batch_size):
    # Different initial angles
    initial_angle = 0.2 * (i + 1)
    
    # Generate observations for this sequence
    obs_seq = []
    angle = initial_angle
    for t in range(T):
        angle = angle + 0.05 * (-9.81 * jnp.sin(angle) - 0.2 * 0.0)  # Simple pendulum
        obs_seq.append([angle + 0.05 * 0.1])  # Add noise
    
    batch_observations.append(obs_seq)
    batch_initial_means.append([initial_angle, 0.0])
    batch_initial_covs.append(jnp.eye(2) * 0.1)

# Convert to arrays
batch_observations = jnp.array(batch_observations)
batch_initial_means = jnp.array(batch_initial_means)
batch_initial_covs = jnp.array(batch_initial_covs)

# Process in batch
batch_results = pendulum_ukf.batch_filter_and_smooth(
    batch_observations, batch_initial_means, batch_initial_covs
)

# Analyze batch results
for i, result in enumerate(batch_results):
    print(f"Sequence {i+1}: Log-likelihood = {result.total_log_likelihood:.2f}")
```

## Example 5: Advanced Configuration

Using advanced UKF configuration options:

```python
# Custom UKF configuration for high precision
custom_config = UKFConfig(
    alpha=1e-4,           # Smaller spread of sigma points
    beta=2.0,             # Optimal for Gaussian distributions
    kappa=1.0,            # Secondary scaling parameter
    regularization_eps=1e-10  # Tighter numerical stability
)

# Create UKF with custom configuration
precise_ukf = GenericUKF(
    state_transition_fn=linear_transition,
    observation_fn=linear_observation,
    process_noise_cov=Q,
    obs_noise_cov=R,
    config=custom_config
)

# The rest follows the same pattern as previous examples
```

## Error Handling

The UKF implementation includes comprehensive error checking:

```python
try:
    # This will raise an error due to dimension mismatch
    bad_observations = jnp.ones((10, 3))  # Wrong observation dimension
    result = pendulum_ukf.filter_and_smooth(bad_observations, initial_mean, initial_cov)
except ValueError as e:
    print(f"Caught expected error: {e}")

try:
    # This will raise an error due to non-positive definite initial covariance
    bad_cov = jnp.array([[1.0, 2.0], [2.0, 1.0]])  # Non-positive definite
    result = pendulum_ukf.filter_and_smooth(observations, initial_mean, bad_cov)
except ValueError as e:
    print(f"Caught expected error: {e}")
```

## Performance Tips

1. **JIT Compilation**: The first call to `filter_and_smooth()` will be slow due to JIT compilation. Subsequent calls will be much faster.

2. **Batch Processing**: Use `batch_filter_and_smooth()` for processing multiple sequences to leverage parallelization.

3. **Memory Management**: For very long sequences, consider processing in chunks if memory becomes an issue.

4. **Numerical Stability**: The implementation automatically handles numerical issues, but extremely ill-conditioned systems may still cause problems.

## Migration from Original Implementation

If you were using the original `PendulumUKFSmoother`, you can migrate as follows:

```python
# Old code:
# from src.baselines.ukf import PendulumUKFSmoother
# smoother = PendulumUKFSmoother(dt=0.05, g=9.81, L=1.0, gamma=0.2)
# result = smoother.smooth(observations, initial_mean, initial_cov)

# New code:
from src.baselines.ukf import create_pendulum_ukf
ukf = create_pendulum_ukf(dt=0.05, g=9.81, L=1.0, gamma=0.2)
result = ukf.filter_and_smooth(observations, initial_mean, initial_cov)
```

The API is nearly identical, with improved performance and stability.