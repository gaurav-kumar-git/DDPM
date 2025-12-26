import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import os
import matplotlib.pyplot as plt
import math
import numpy as np

class NoiseScheduler():
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model
    
    """
    def __init__(self, num_timesteps=50, type="linear", **kwargs):

        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            self.init_linear_schedule(**kwargs)
        elif type == "cosine":
            self.init_cosine_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented") # change this if you implement additional schedulers


    def init_linear_schedule(self, beta_start, beta_end):
        """
        Precompute whatever quantities are required for training and sampling
        """

        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def init_cosine_schedule(self, s=0.008):
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        factor = (x / self.num_timesteps + s) / (1 + s)
        alphas_cumprod = torch.cos(factor * (math.pi / 2)) ** 2
        alphas_cumprod /= alphas_cumprod[0]
        self.betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def __len__(self):
        return self.num_timesteps

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x):
        return x + self.block(x)

class DDPM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200):
        """
        Noise prediction network for the DDPM

        Args:
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        We have separate learnable modules for `time_embed` and `model`. `time_embed` can be learned or a fixed function as well

        """
        super().__init__()
        self.n_dim = n_dim
        self.n_steps = n_steps

        # Time embedding layers
        self.time_embed = nn.Sequential(
            nn.Embedding(n_steps, 256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # Noise prediction model -> Implements ϵ_θ(x_t, t)
        self.model = nn.Sequential(
            nn.Linear(n_dim + 256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Linear(512, n_dim)
        )

    def forward(self, x, t):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]

        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        t_embed = self.time_embed(t)
        model_input = torch.cat([x, t_embed], dim=1)
        return self.model(model_input)

class ConditionalDDPM(nn.Module):
    def __init__(self, n_classes=2, n_dim=3, n_steps=200):
        """
        Class dependernt noise prediction network for the DDPM

        Args:
            n_classes: number of classes in the dataset
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        We have separate learnable modules for `time_embed` and `model`. `time_embed` can be learned or a fixed function as well

        """
        super().__init__()
        self.n_classes = n_classes
        self.n_dim = n_dim
        self.n_steps = n_steps

        # Time embedding layers
        self.time_embed = nn.Sequential(
            nn.Embedding(n_steps, 256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # Class embedding layers (including null class)
        self.class_embed = nn.Sequential(
            nn.Embedding(n_classes + 1, 256),  # +1 for null class
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # Noise prediction model
        self.model = nn.Sequential(
            nn.Linear(n_dim + 512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Linear(512, n_dim)
        )

    def forward(self, x, t, y):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]
            y: torch.Tensor, the class label tensor [batch_size]
        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        t_embed = self.time_embed(t)
        y_embed = self.class_embed(y)
        combined_embed = torch.cat([t_embed, y_embed], dim=1)
        model_input = torch.cat([x, combined_embed], dim=1)
        return self.model(model_input)

class ClassifierDDPM:
    """
    ClassifierDDPM implements a classification algorithm using the Conditional DDPM model.
    """
    
    def __init__(self, model, noise_scheduler, n_samples=100, device='cuda'):
        """
        Args:
            model: ConditionalDDPM model
            noise_scheduler: NoiseScheduler for DDPM
            n_samples: Number of samples to estimate likelihood
            device: Device for computation
        """
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.n_samples = n_samples
        self.device = device
        self.n_classes = model.n_classes  # Get number of classes from model

    def compute_likelihood(self, x, class_label):
        """
        Estimate log-likelihood p(x | y) using reverse diffusion process.
        """
        batch_size, n_dim = x.shape
        x = x.to(self.device)

        # Generate samples using Classifier-Free Guidance (CFG)
        reconstructed_x, _ = sampleCFG(
            model=self.model, 
            n_samples=batch_size,  # Use batch_size instead of self.n_samples
            noise_scheduler=self.noise_scheduler, 
            guidance_scale=1.0, 
            class_label=class_label
        )  # Shape: [batch_size, n_dim]

        # Compute likelihood as negative reconstruction error
        log_likelihood = -F.mse_loss(reconstructed_x, x, reduction='none').mean(dim=1)      
        return log_likelihood  # Shape: [batch_size]
    
    def predict(self, x):
        """
        Predicts class labels for input x.
        """
        log_likelihoods = torch.stack([self.compute_likelihood(x, c) for c in range(self.n_classes)], dim=1)
        return log_likelihoods.argmax(dim=1)  # Return class with highest likelihood

    def predict_proba(self, x):
        """
        Returns class probabilities using softmax over log-likelihoods.
        """
        log_likelihoods = torch.stack([self.compute_likelihood(x, c) for c in range(self.n_classes)], dim=1)
        return F.softmax(log_likelihoods, dim=1)  # Convert log-likelihoods to probabilities

def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    """
    Train the model and save the model and necessary plots.

    Args:
        model (DDPM): The model to train.
        noise_scheduler (NoiseScheduler): Scheduler for the noise.
        dataloader (torch.utils.data.DataLoader): Dataloader for the dataset.
        optimizer (torch.optim.Optimizer): Optimizer to use.
        epochs (int): Number of epochs to train the model.
        run_name (str): Path to save the model.
    """
    model.train()
    device = next(model.parameters()).device
    alpha_bars = noise_scheduler.alpha_bars.to(device)

    progress_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in progress_bar:
        for x, _ in dataloader:
            x = x.to(device)
            batch_size = x.shape[0]

            # Sample timesteps
            t = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=device).long()

            # Sample noise
            epsilon = torch.randn_like(x)

            # Compute noisy input x_t
            alpha_bars_t = alpha_bars[t]
            sqrt_alpha_bars_t = torch.sqrt(alpha_bars_t)[:, None]
            sqrt_one_minus_alpha_bars_t = torch.sqrt(1 - alpha_bars_t)[:, None]

            x_t = sqrt_alpha_bars_t * x + sqrt_one_minus_alpha_bars_t * epsilon

            # Predict noise and compute loss
            epsilon_theta = model(x_t, t)
            loss = F.mse_loss(epsilon_theta, epsilon)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

        # Save model checkpoint
        torch.save(model.state_dict(), f"{run_name}/model.pth")


@torch.no_grad()
def sample(model, n_samples, noise_scheduler, return_intermediate=False): 
    """
    Sample from the model.

    Args:
        model (DDPM): The trained diffusion model.
        n_samples (int): Number of samples to generate.
        noise_scheduler (NoiseScheduler): Scheduler managing the noise levels.
        return_intermediate (bool, optional): Whether to return intermediate steps for visualization.

    Returns:
        torch.Tensor: Final samples of shape [n_samples, n_dim] if `return_intermediate` is False.
        list[torch.Tensor]: List of intermediate steps, each of shape [n_samples, n_dim], if `return_intermediate` is True.
    """   
    model.eval()
    device = next(model.parameters()).device

    # Precompute scheduler parameters
    betas, alphas, alpha_bars = (
        noise_scheduler.betas.to(device),
        noise_scheduler.alphas.to(device),
        noise_scheduler.alpha_bars.to(device),
    )

    # Initialize noise
    x_t = torch.randn(n_samples, model.n_dim, device=device)
    intermediates = [] if return_intermediate else None

    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, dtype=torch.long, device=device)
        eps_theta = model(x_t, t_tensor)

        beta_t = betas[t]
        sqrt_alpha_t = torch.sqrt(alphas[t])
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bars[t])

        # Compute mean
        mean = (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta) / sqrt_alpha_t

        # Sample noise or set to zero for last step
        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        x_t = mean + torch.sqrt(beta_t) * noise

        if return_intermediate:
            intermediates.append(x_t.cpu())

    return intermediates if return_intermediate else x_t


def sample_albatross(model, noise_scheduler, xT, return_intermediate=False): 
    """
    Deterministic sampling from the trained model using albatross prior samples.

    Args:
        model (DDPM): The trained diffusion model.
        noise_scheduler (NoiseScheduler): Scheduler managing the noise levels.
        xT (torch.Tensor): Predefined noise vectors from `data/albatross_prior_samples.npy`.
        return_intermediate (bool, optional): Whether to return intermediate steps.

    Returns:
        torch.Tensor: Final samples of shape [n_samples, n_dim] if `return_intermediate` is False.
        list[torch.Tensor]: List of intermediate steps, each of shape [n_samples, n_dim], if `return_intermediate` is True.
    """   
    model.eval()
    device = next(model.parameters()).device

    # Precompute scheduler parameters
    betas, alphas, alpha_bars = (
        noise_scheduler.betas.to(device),
        noise_scheduler.alphas.to(device),
        noise_scheduler.alpha_bars.to(device),
    )

    # Use provided xT instead of random initialization
    x_t = xT.to(device)
    intermediates = [] if return_intermediate else None

    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.full((x_t.shape[0],), t, dtype=torch.long, device=device)
        eps_theta = model(x_t, t_tensor)

        beta_t = betas[t]
        sqrt_alpha_t = torch.sqrt(alphas[t])
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bars[t])

        # Compute mean
        mean = (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta) / sqrt_alpha_t

        # Ensure deterministic sampling by setting z = 0 at step t == 0
        noise = torch.zeros_like(x_t) if t == 0 else torch.randn_like(x_t)
        x_t = mean + torch.sqrt(beta_t) * noise

        if return_intermediate:
            intermediates.append(x_t.cpu())

    return intermediates if return_intermediate else x_t


def trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, run_name, p_uncond=0.1):
    """ 
    Train a conditional diffusion model with scheduled noise levels.

    Parameters:
    - model (nn.Module): Diffusion model instance.
    - noise_scheduler (NoiseScheduler): Noise management scheduler.
    - dataloader (DataLoader): Batched training dataset.
    - optimizer (torch.optim.Optimizer): Optimization algorithm.
    - epochs (int): Total number of training iterations.
    - run_name (str): Directory to store the trained model.
    - p_uncond (float, default=0.1): Probability of label nullification.

    Saves:
    - Model checkpoint at "{run_name}/model.pth".
    """

    model.train()
    device = next(model.parameters()).device
    alpha_bars = noise_scheduler.alpha_bars.to(device)
    null_class = model.n_classes  # Index representing the null class

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            # Randomly nullify labels based on probability p_uncond
            y[torch.rand(batch_size, device=device) < p_uncond] = null_class 

            # Sample a random timestep t ~ U({1, ..., T})
            t = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=device)

            # Generate Gaussian noise ε ~ N(0, I)
            epsilon = torch.randn_like(x)

            # Compute noisy sample x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
            alpha_bars_t = alpha_bars[t].view(-1, 1)
            x_t = alpha_bars_t.sqrt() * x + (1 - alpha_bars_t).sqrt() * epsilon

            # Model prediction and loss computation
            epsilon_theta = model(x_t, t, y)
            loss = F.mse_loss(epsilon_theta, epsilon)

            # Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save model checkpoint after every epoch
        torch.save(model.state_dict(), f"{run_name}/model.pth")


@torch.no_grad()
def sampleConditional(model, n_samples, noise_scheduler, class_label):
    

    model.eval()
    device = next(model.parameters()).device

    # Extract precomputed noise scheduler parameters
    betas = noise_scheduler.betas.to(device)
    alphas = noise_scheduler.alphas.to(device)
    alpha_bars = noise_scheduler.alpha_bars.to(device)

    # Step 1: Initialize x_T ~ N(0, I) (starting point of reverse process)
    x_t = torch.randn(n_samples, model.n_dim, device=device)

    # Reverse diffusion loop (iterating backwards in time)
    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)

        # Step 2: Predict the noise ε_θ using the model
        y_tensor = torch.full((n_samples,), class_label, device=device, dtype=torch.long)
        epsilon_theta = model(x_t, t_tensor, y_tensor)

        # Extract noise parameters for the current timestep
        beta_t = betas[t]
        alpha_t = alphas[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)  # √α_t
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bars[t])  # √(1 - ᾱ_t)

        # Step 3: Compute the estimated clean data point at time t
        mean = (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * epsilon_theta) / sqrt_alpha_t

        # Step 4: Add noise if t > 0, otherwise set it to zero
        z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        x_t = mean + torch.sqrt(beta_t) * z  # Compute x_{t-1}

    return x_t,torch.full((n_samples,), class_label, device=device, dtype=torch.long) # Return final generated samples


@torch.no_grad()
def sampleCFG(model, n_samples, noise_scheduler, guidance_scale, class_label):
    """
    Generates samples using Classifier-Free Guidance (CFG).
    
    Implements CFG from Ho & Salimans (2021) alongside the DDPM reverse process.
    
    Args:
        model (nn.Module): Trained diffusion model.
        n_samples (int): Number of samples to generate.
        noise_scheduler (NoiseScheduler): Scheduler with noise parameters.
        guidance_scale (float): Strength of classifier-free guidance.
        class_label (int): Conditioning label for guided generation.

    Returns:
        torch.Tensor: Generated samples of shape [n_samples, model.n_dim].
    """

    model.eval()
    device = next(model.parameters()).device

    # Retrieve precomputed noise parameters from the scheduler
    betas, alphas, alpha_bars = (
        noise_scheduler.betas.to(device),
        noise_scheduler.alphas.to(device),
        noise_scheduler.alpha_bars.to(device),
    )
    null_class = model.n_classes  # Special class index for unconditional generation

    # Step 1: Initialize x_T ~ N(0, I), the starting point for reverse process
    x_t = torch.randn(n_samples, model.n_dim, device=device)

    # Reverse diffusion loop (t = T to 1)
    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)

        # Step 2: Prepare inputs for conditional and unconditional passes
        x_t_repeated = torch.cat([x_t, x_t], dim=0)  # Duplicate x_t
        t_repeated = torch.cat([t_tensor, t_tensor], dim=0)  # Duplicate timesteps
        class_labels = torch.cat([
            torch.full((n_samples,), class_label, device=device, dtype=torch.long),  # Conditional class y
            torch.full((n_samples,), null_class, device=device, dtype=torch.long),   # Unconditional class ∅
        ], dim=0)

        # Step 3: Predict noise ε_θ for both conditional and unconditional passes
        epsilon = model(x_t_repeated, t_repeated, class_labels)
        epsilon_cond, epsilon_uncond = torch.chunk(epsilon, 2, dim=0)  # Split results

        # Step 4: Apply Classifier-Free Guidance (CFG)
        epsilon_theta = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)

        # Step 5: Compute mean for x_{t-1} using guided noise prediction
        beta_t, alpha_t = betas[t], alphas[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bars[t])
        
        mean = (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * epsilon_theta) / sqrt_alpha_t

        # Step 6: Add noise for all steps except the last one
        z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        x_t = mean + torch.sqrt(beta_t) * z  # Compute x_{t-1}

    return x_t,torch.full((n_samples,), class_label, device=device, dtype=torch.long)  # Return generated samples


def sampleSVDD(model, n_samples, noise_scheduler, reward_scale, reward_fn):
    """
    Sample from the SVDD model

    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        reward_scale: float
        reward_fn: callable, takes in a batch of samples torch.Tensor:[n_samples, n_dim] and returns torch.Tensor[n_samples]

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "sample"])
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--lbeta", type=float, default=None)
    parser.add_argument("--ubeta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--n_dim", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--schedule", type=str, default="linear", choices=["linear", "quadratic", "cosine"])  
    parser.add_argument('--type',type=str,default="unconditional",choices=["unconditional",'conditional'])
    
    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    run_name = f'exps/ddpm_{args.type}_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.schedule}_{args.dataset}' # can include more hyperparams
    os.makedirs(run_name, exist_ok=True)

    if args.type == 'unconditional':
        model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps)
        noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta)
        model = model.to(device)

        if args.mode == 'train':
            epochs = args.epochs
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            data_X, data_y = dataset.load_dataset(args.dataset)
            # can split the data into train and test -- for evaluation later
            data_X = data_X.to(device)
            # data_y = data_y.to(device)
            dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X), batch_size=args.batch_size, shuffle=True)
            train(model, noise_scheduler, dataloader, optimizer, epochs, run_name)

        elif args.mode == 'sample':
            model.load_state_dict(torch.load(f'{run_name}/model.pth'))
            samples = sample(model, args.n_samples, noise_scheduler)
            torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')

    if args.type == 'conditional':
        data_X, data_y = dataset.load_dataset(args.dataset)
        model = ConditionalDDPM(n_dim=args.n_dim, n_steps=args.n_steps,n_classes=data_y.unique().numel())
        noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta)
        model = model.to(device)

        if args.mode == 'train':
            epochs = args.epochs
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            # can split the data into train and test -- for evaluation later
            data_X = data_X.to(device)
            data_y = data_y.to(device)
            dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X,data_y), batch_size=args.batch_size, shuffle=True)
            trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, run_name)

        elif args.mode == 'sample':
            model.load_state_dict(torch.load(f'{run_name}/model.pth'))
            
            unique_classes = data_y.unique()
            
            guidance_scales = [0.0, 0.25,0.5,0.75,1]  # Define guidance scales
            n_samples_per_class = args.n_samples // len(unique_classes)  # Divide total samples equally among classes
            
            for guidance_scale in guidance_scales:
                all_samples = []
                all_labels = []

                for class_label in unique_classes:
                    samples, labels = sampleCFG(model, n_samples_per_class, noise_scheduler, guidance_scale, class_label.item())  
                    all_samples.append(samples)  # Append tensor samples
                    all_labels.append(labels)  # Append tensor labels

                # Convert list of tensors to a single tensor
                all_samples = torch.cat(all_samples, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                # Save the generated samples along with their labels
                save_path = f'{run_name}/samples_{args.seed}_{args.n_samples}_guidance_{guidance_scale}.pth'
                torch.save({'samples': all_samples, 'labels': all_labels}, save_path)
                print(f"Saved samples with guidance scale {guidance_scale} to {save_path}")
    else:
        raise ValueError(f"Invalid mode {args.mode}")