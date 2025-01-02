import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
from typing import Any, Callable, Tuple
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from torch.utils import data

# Helper Functions (modified for JAX)
def to_one_hot(labels: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    return jax.nn.one_hot(labels, num_classes)

# CIFAR-10 Dataset
def load_cifar10(batch_size: int, split: str = 'train'):
    ds = tfds.load('cifar10', split=split)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tfds.data.AUTOTUNE)
    ds = tfds.as_numpy(ds)
    return ds

def prepare_batch(batch):
  images = batch["image"].astype(np.float32) / 255.
  return jnp.asarray(images)

# --- MODEL IMPLEMENTATION (Flax Modules) ---
class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.latent_dim, kernel_size=(2, 2), strides=(1, 1))(x)
        return x


class Decoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(features=128, kernel_size=(2, 2), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=3, kernel_size=(4, 4), strides=(2, 2))(x)
        x = jax.nn.sigmoid(x)
        return x


class VectorQuantizer(nn.Module):
    num_embeddings: int
    embedding_dim: int
    commitment_cost: float

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        # Flatten input to (B, H*W, latent_dim)
        flat_inputs = jnp.reshape(inputs, (inputs.shape[0], -1, self.embedding_dim))
        # Initialize embedding vectors at the first run.
        embeddings = self.param('embeddings', jax.random.normal, (self.num_embeddings, self.embedding_dim))

        # Compute distances of flat_inputs to embeddings
        distances = jnp.sum(flat_inputs**2, axis=2, keepdims=True) \
                      - 2 * jnp.matmul(flat_inputs, embeddings.T) \
                      + jnp.sum(embeddings**2, axis=1)
        
        # Get the index of nearest embedding to each flattened input vector
        encoding_indices = jnp.argmin(distances, axis=2)
        
        # One-hot encode the indices
        one_hot_encodings = to_one_hot(encoding_indices, self.num_embeddings)

        # Quantize the encoded output from encoder using codebook. 
        quantized_outputs = jnp.matmul(one_hot_encodings, embeddings)
        
        # Reshape quantized output back to original encoded output shape.
        quantized_outputs = jnp.reshape(quantized_outputs, inputs.shape)
        
        # Commitments
        e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized_outputs) - inputs) ** 2)

        # Commitment cost.
        loss = self.commitment_cost * e_latent_loss
        
        # Using straight-through estimator for better gradient propagation.
        quantized_outputs = inputs + jax.lax.stop_gradient(quantized_outputs - inputs)
        
        return quantized_outputs, loss, encoding_indices

class VQVAE(nn.Module):
  num_embeddings: int
  embedding_dim: int
  commitment_cost: float
  
  @nn.compact
  def __call__(self, x):
    encoder = Encoder(latent_dim=self.embedding_dim)
    decoder = Decoder(latent_dim=self.embedding_dim)
    vector_quantizer = VectorQuantizer(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim, commitment_cost=self.commitment_cost)

    z = encoder(x)
    quantized_z, commitment_loss, encoding_indices  = vector_quantizer(z)
    reconstructions = decoder(quantized_z)

    return reconstructions, commitment_loss, encoding_indices

# --- LOSS FUNCTIONS ---

def reconstruction_loss(recon_x: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((recon_x - x) ** 2)


def vq_vae_loss(
    vqvae_output: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    x: jnp.ndarray,
    beta: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    reconstructions, commitment_loss, _ = vqvae_output
    recon_loss = reconstruction_loss(reconstructions, x)
    loss = recon_loss + beta * commitment_loss

    metrics = {
        'reconstruction_loss': recon_loss,
        'commitment_loss': commitment_loss,
        'loss': loss
    }
    return loss, metrics

# --- TRAINING SETUP ---

# State management
class TrainState(train_state.TrainState):
    vq_vae_params: flax.core.FrozenDict[str, Any]

def create_train_state(rng, vqvae, learning_rate, input_shape):
    params_rng, dropout_rng = jax.random.split(rng)
    init_batch = jnp.zeros((1,) + input_shape)
    variables = vqvae.init(params_rng, init_batch)
    vq_vae_params = variables['params']

    tx = optax.adam(learning_rate)

    return TrainState.create(
        apply_fn=vqvae.apply,
        params = vq_vae_params,
        tx=tx,
        vq_vae_params=vq_vae_params,
        dropout_rng=dropout_rng,
        )


@jax.jit
def train_step(state: TrainState, batch: jnp.ndarray, beta: float):

  def loss_fn(params):
    reconstructions, commitment_loss, _ = state.apply_fn({'params':params}, batch, rngs={'params': state.dropout_rng})
    return vq_vae_loss((reconstructions, commitment_loss, _), batch, beta=beta)
  
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, metrics), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)

  return state, metrics

# Evaluation Function
@jax.jit
def eval_step(state: TrainState, batch: jnp.ndarray, beta:float):
    reconstructions, commitment_loss, _ = state.apply_fn({'params': state.params}, batch, rngs={'params': state.dropout_rng})
    loss, metrics = vq_vae_loss((reconstructions, commitment_loss, _), batch, beta)
    return metrics

def train_model(
    num_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    num_embeddings: int = 512,
    embedding_dim: int = 64,
    commitment_cost: float = 0.25,
    beta: float = 1.0,
    input_shape=(32, 32, 3),
    visualization_epochs: int = 2
):
  # Data Setup
    train_ds = load_cifar10(batch_size=batch_size, split='train')
    test_ds = load_cifar10(batch_size=batch_size, split='test')

    # Model and Optimizer Setup
    key = jax.random.PRNGKey(0)
    key, params_rng = jax.random.split(key)
    vqvae = VQVAE(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)

    state = create_train_state(params_rng, vqvae, learning_rate, input_shape)


    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_ds):
          images = prepare_batch(batch)
          state, metrics = train_step(state, images, beta)

        eval_metrics = {}
        for batch_idx, batch in enumerate(test_ds):
            images = prepare_batch(batch)
            metrics = eval_step(state, images, beta)
            for k, v in metrics.items():
              eval_metrics[k] = eval_metrics.get(k, 0) + v / len(test_ds)

        print(f'Epoch {epoch+1}/{num_epochs} -  Train Loss: {metrics["loss"]:.4f}, Train Reconstruction Loss: {metrics["reconstruction_loss"]:.4f}, Train Commitment Loss: {metrics["commitment_loss"]:.4f}, Evaluation Loss: {eval_metrics["loss"]:.4f}, Evaluation Reconstruction Loss: {eval_metrics["reconstruction_loss"]:.4f}, Evaluation Commitment Loss: {eval_metrics["commitment_loss"]:.4f}')
        
        # Visualize reconstructed images
        if (epoch + 1) % visualization_epochs == 0:
          for batch_idx, batch in enumerate(test_ds):
            images = prepare_batch(batch)
            reconstructions, _, _ = state.apply_fn({'params':state.params}, images, rngs={'params': state.dropout_rng})
            
            # Select a few images and their reconstructions
            num_to_visualize = 4
            images_to_plot = images[:num_to_visualize]
            reconstructions_to_plot = reconstructions[:num_to_visualize]
            
            # Create figure
            fig, axes = plt.subplots(2, num_to_visualize, figsize=(12, 6))
            
            for i in range(num_to_visualize):
              axes[0, i].imshow(images_to_plot[i])
              axes[0, i].set_title("Original")
              axes[0, i].axis('off')
              axes[1, i].imshow(reconstructions_to_plot[i])
              axes[1, i].set_title("Reconstructed")
              axes[1, i].axis('off')

            plt.tight_layout()
            plt.savefig(f"reconstruct/reconstruction_epoch_{epoch+1}.png")
            plt.close(fig)
            break

if __name__ == '__main__':
  train_model()