import torch
import numpy as np
import argparse
import os
from ddpm_albatross import DDPM, NoiseScheduler, sample_albatross_on_cpu

def reproduce_samples(model_path, output_file, prior_samples_path,lbeta,ubeta):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = DDPM(n_dim=64, n_steps=200).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    noise_scheduler = NoiseScheduler(num_timesteps=200,beta_start=lbeta,beta_end=ubeta)

    xT = torch.tensor(np.load(prior_samples_path), dtype=torch.float32).to(device)

    samples = sample_albatross_on_cpu(model, noise_scheduler, xT=xT)

    np.save(output_file, samples.cpu().detach().numpy())
    print(f"Reproduced samples saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="exps/ddpm_albatross_64_200_0.0001_0.02/model.pth", help="Path to the trained model")
    parser.add_argument("--output_file", type=str, default="albatross_samples_reproduce.npy", help="Output file for reproduced samples")
    parser.add_argument("--prior_samples", type=str, default="data/albatross_prior_samples.npy", help="Path to prior samples for deterministic sampling")
    parser.add_argument("--lbeta", type=float, default=0.0001, help="lbeta")
    parser.add_argument("--ubeta", type=float, default=0.02, help="ubeta")
    
    args = parser.parse_args()
    
    reproduce_samples(args.model_path, args.output_file, args.prior_samples,args.lbeta,args.ubeta)
