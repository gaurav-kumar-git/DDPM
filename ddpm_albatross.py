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
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
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
            nn.Linear(n_dim + 256, 512),
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
    
class ClassifierDDPM():
    """
    ClassifierDDPM implements a classification algorithm using the DDPM model
    """
    
    def __init__(self, model: ConditionalDDPM, noise_scheduler: NoiseScheduler):
        pass

    def __call__(self, x):
        pass

    def predict(self, x):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
        Returns:
            torch.Tensor, the predicted class tensor [batch_size]
        """

        pass

    def predict_proba(self, x):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
        Returns:
            torch.Tensor, the predicted probabilites for each class  [batch_size, n_classes]
        """

        pass


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
        for x_tuple in tqdm(dataloader):
            x = x_tuple[0].to(model.model[0].weight.device)
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

def sample_albatross_on_cpu(model, noise_scheduler, xT, return_intermediate=False):
    """
    Samples using a trained model but runs on CPU to avoid MPS memory issues.

    Args:
        model (DDPM): The trained diffusion model.
        noise_scheduler (NoiseScheduler): Scheduler managing noise levels.
        xT (torch.Tensor): Predefined noise vectors from ⁠ data/albatross_prior_samples.npy ⁠.
        return_intermediate (bool, optional): Whether to return intermediate steps.

    Returns:
        torch.Tensor: Final samples.
    """
    # Move model and tensors to CPU
    device = torch.device("cpu")
    model.to(device)
    xT = xT.to(device)

    model.eval()
    with torch.no_grad():  # Disable gradients to save memory
        return sample_albatross(model, noise_scheduler, xT, return_intermediate)


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
    model.train()
    device = next(model.parameters()).device
    alpha_bars = noise_scheduler.alpha_bars.to(device)
    null_class = model.n_classes  # Null class index

    progress_bar = tqdm(range(epochs), desc="Training Progress")
    for epoch in progress_bar:
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            batch_size = x.size(0)

            # Randomly replace some labels with null class
            mask = torch.rand(batch_size, device=device) < p_uncond
            y[mask] = null_class # ỹ ← ∅ with probability p_uncond

            # Sample timesteps
            # 1. Sample x_0 ~ q(x_0) (data loading)
            # 2. Sample t ~ Uniform({1,...,T})
            t = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=device).long()
            
            # 3. Sample ϵ ~ N(0, I)
            epsilon = torch.randn_like(x)

            # Compute noisy x
            alpha_bars_t = alpha_bars[t]
            # x_t = √ᾱ_t x_0 + √(1-ᾱ_t)ϵ
            x_t = torch.sqrt(alpha_bars_t)[:, None] * x + torch.sqrt(1 - alpha_bars_t)[:, None] * epsilon

            # Predict and compute loss
            epsilon_theta = model(x_t, t, y) # ϵ_θ(x_t, t, ỹ)
            # 4. Take gradient descent step on ∇_θ||ϵ - ϵ_θ(√ᾱ_t x_0 + √(1-ᾱ_t)ϵ, t)||^2
            loss = F.mse_loss(epsilon_theta, epsilon)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

        # Save model
        torch.save(model.state_dict(), f"{run_name}/model.pth")

@torch.no_grad()
def sampleConditional(model, n_samples, noise_scheduler, guidance_scale, class_label):
    model.eval()
    device = next(model.parameters()).device

    betas = noise_scheduler.betas.to(device)
    alphas = noise_scheduler.alphas.to(device)
    alpha_bars = noise_scheduler.alpha_bars.to(device)

    # Initialize with pure noise (x_T ∼ N(0,I))
    x_t = torch.randn(n_samples, model.n_dim, device=device)
    null_class = model.n_classes  # Null class index

    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        # Duplicate inputs for conditional and unconditional
        x_t_repeated = torch.cat([x_t, x_t], dim=0)
        t_repeated = torch.cat([t_tensor, t_tensor], dim=0)
        class_labels = torch.cat([
            torch.full((n_samples,), class_label, device=device, dtype=torch.long),
            torch.full((n_samples,), null_class, device=device, dtype=torch.long),
        ], dim=0)

        # Predict noise for both conditions
        epsilon = model(x_t_repeated, t_repeated, class_labels)
        epsilon_cond, epsilon_uncond = torch.chunk(epsilon, 2, dim=0)

        # Combine using guidance scale
        # ϵ̃ = ϵ_uncond + guidance_scale*(ϵ_cond - ϵ_uncond)
        epsilon_theta = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)

        # Reverse process step
        beta_t = betas[t]
        alpha_t = alphas[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bars[t])

        # Compute x_{t-1} mean
        mean = (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * epsilon_theta) / sqrt_alpha_t
        
        z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        x_prev = mean + torch.sqrt(beta_t) * z

        x_t = x_prev

    return x_t

def sampleCFG(model, n_samples, noise_scheduler, guidance_scale, class_label):
    """
    Sample from the conditional model
    
    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        guidance_scale: float
        class_label: int

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass

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
    parser.add_argument("--mode", choices=['train', 'sample','sample_albatross'], default='sample')
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--lbeta", type=float, default=None)
    parser.add_argument("--ubeta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataset", type=str, default = None)
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--n_dim", type=int, default = None)

    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using device: {device}")
    run_name = f'exps/ddpm_{args.dataset}_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}' # can include more hyperparams
    os.makedirs(run_name, exist_ok=True)

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
        model.load_state_dict(torch.load(f'{run_name}/model.pth', weights_only=True))
        samples = sample(model, args.n_samples, noise_scheduler)
        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')

    elif args.mode == 'sample_albatross':
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        prior_samples_path= "data/albatross_prior_samples.npy"
        xT = torch.tensor(np.load(prior_samples_path), dtype=torch.float32).to(device)
        samples = sample_albatross_on_cpu(model, noise_scheduler, xT)
        torch.save(samples, f'{run_name}/albatross_samples.npy')
        
    else:
        raise ValueError(f"Invalid mode {args.mode}")