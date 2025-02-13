import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 
import copy 
from torch.nn import init
from torch.nn import functional as F
import math 
import time

# Dataset Generation 
def generate_dataset(batch_size=64, num_samples=1000, seq_len=250, input_duration=5, noise_std=0.01):
    """
    Generate a sataset for the ring task.
    :param batch_size: Number of samples in a batch.
    :param num_samples: Total number of samples.
    :param seq_len: Length of each sequence (time steps).
    :param input_duration: Duration (in time steps) for which the input is provided.
    :param noise_std: Standard deviation of the Gaussian noise. 
    :return: input tensor, target tensor
    """
    inputs = []
    targets = []
    original_degrees = []

    for _ in range(num_samples):
        # Random angle theta (original angle value)
        theta = np.random.uniform(-np.pi, np.pi)
        theta_degree = np.degrees(theta)
        sin_cos = np.array([np.sin(theta), np.cos(theta)])
        
        # Input: sin_cos for 'input_duration', zeros for the rest ??? Some questions 
        input_sequence = np.zeros((seq_len, 2))
        input_sequence[0] = sin_cos

        # Add Gaussian noise to the input
        noise = np.random.normal(0, noise_std, (seq_len, 2))
        input_sequence += noise 
        
        # Target: Maintain the same sin_cos for the entire sequence
        target_sequence = np.tile(sin_cos, (seq_len, 1))

        inputs.append(input_sequence)
        targets.append(target_sequence)
        original_degrees.append(theta_degree)
    
    return (
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(original_degrees, dtype=torch.float32),
    )

# Continuous-time RNN
class CTRNN(nn.Module):
    """Continuous-time RNN.
    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms.
            If None, dt equals time constant tau

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
             if None, hidden is initialized through self.init_hidden()
    
    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=10, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        # print("the batch size is", batch_size)
        return torch.zeros(batch_size, self.hidden_size)
    
    def recurrence(self, input, hidden):
        """Run network for one time step.

        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)

        Outputs: 
             h_new: tensor of shape (batch, hidden_size).
                network activity at the next time step
        """
        # print("the size of input before h_new is", input.shape)
        # print("the hidden size current is", hidden.shape)
        h_new = torch.relu(self.input2h(input) + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new 

    def forward(self, input, hidden=None):
        """Propogate input through the network."""

        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)
        
        # Loop through time
        output = []
        steps = range(input.size(0))
        #print("the shape of input is !!!!", input.shape)
        for i in steps:

            #print("the size of the input[i] is", input[i].shape)
            #print("the shape of hidden is", hidden.shape)

            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)
        
        # Stack together output from all time steps
        output = torch.stack(output, dim=0) # (seq_len, batch, hidden_size)
        return output, hidden 

class RNNNet(nn.Module):
    """Recurrent neiwork model.

    Parameters:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size

    Inputs:
        x: tensor of shape (Seq Len, Batch, Input size)

    Outputs:
        out: tensor of shape (Seq Len, Batch, Output size)
        rnn_output: tensor of shape (Seq Len, Batch, Hidden size)
    """
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, **kwargs)

        # Add an output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        out = self.fc(rnn_output)
        return out, rnn_output


"""
# RNN Model
class VanillaRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=300, output_size=2, recurrent_noise_std=0.01):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        # self.recurrent_noise_std = recurrent_noise_std 
        self.rnn_cell = nn.RNNCell(input_size, hidden_size, nonlinearity="relu")
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []
        for t in range(seq_len):
            h = self.rnn_cell(x[:, t], h)
            # Add Gaussian noise to the hidden state if specified
            #if self.recurrent_noise_std > 0:
            #   noise = torch.randn_like(h) * self.recurrent_noise_std
            #   h = h + noise 
            outputs.append(self.output_layer(h))
        return torch.stack(outputs, dim=1)
"""

# Function to add Gaussian noise to model weights
def add_weight_noise(model, noise_st):
    
    #Add Gaussian noise to the model weights.
    #:param model: The model whose weights will be perturbed
    #:param noise_std: Standard deviation of the Gaussian noise
    
    # in pytorch, a model's weights (parameters) are stored as torch.Tensor objects
    # we can access all the parameters using model.parameters(), which includes the weights of the RNN cells and the weights of the output layer or biases 
    # Gaussian noise is generated using torch.randn_like, whcih creates a tensor of teh same shape as the input tensor
    # The noise is added to the model weights using param.add_() 
    # breakpoint() 
    with torch.no_grad(): 
        for param in model.parameters():
            noise = torch.randn_like(param) * noise_st
            param.add_(noise)

# Training Function
def train_model(model, dataset, batch_size, num_epchos, learning_rate=1e-3, record_interval=10, eval_dataset=None, original_degrees=None): 
    inputs, targets = dataset 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
    criterion = nn.MSELoss() 
    eval_records = []
    
    start_time = time.time()
    for epcho in range(num_epchos): 
        model.train()  
        permutation = torch.randperm(inputs.size(0))  
        inputs = inputs[permutation]
        targets = targets[permutation]
        epoch_loss = 0
        for i in range(0, inputs.size(0), batch_size):
            x_batch = inputs[i : i + batch_size].to(device)
            y_batch = targets[i : i + batch_size].to(device)
            # breakpoint() 
            # print("the shape of targets is", y_batch.shape) 
            optimizer.zero_grad()
            x_batch = torch.transpose(x_batch, 0, 1)
            # print("shape of the x_batch is", x_batch.shape)
            outputs, _ = model(x_batch)
            outputs = torch.transpose(outputs, 0, 1)
            # print("the shape of the outputs is", outputs.shape)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # avg_loss = epoch_loss / (inputs.size(0) // batch_size)
        print(f"Epoch {epcho + 1}/{num_epchos}, Loss: {epoch_loss:.6f}, Time: {(time.time() - start_time):.1f}")

        # Evaluate every record_interval epochs if eval_dataset is provided
        if eval_dataset is not None and (epcho + 1) % record_interval == 0:
            # Evaluate for each noise condition separately
            eval_results = evaluate_model_with_noise(model, eval_dataset, batch_size, noise_std_list=[0, weight_noise_std], original_degree=original_degrees)
            for noise_std, stats in eval_results.items():
                eval_records.append({
                    'epoch': epcho + 1,
                    'noise_std': noise_std,
                    'mean_angle_error': stats['mean'],
                    'std_angle_error': stats['std']
                })
    
    return eval_records

"""
def evaluate_model_with_noise(model, dataset, batch_size, noise_std_list=[0, 0.003]):
    
    # Evaluate the model after adding Gaussian noise to its weights.
    # :param model: The trained model.
    # :param dataset: Tuple of (inputs, targets).
    # :param batch_size: Batch size for evaluation.
    # :param noise_std: Standard deviation of the Gaussian noise added to weights
    
    inputs, targets = dataset
    criterion = nn.MSELoss() # Used to compute the loss value
    results = {} 
    
    for noise_std in noise_std_list:

        # Add Gaussian noise to the modek weights 
        if noise_std > 0:
           add_weight_noise(model, noise_st=noise_std)
    
        # Evaluate the model
        model.eval()
        total_loss = 0
        total_angle_error = 0
        with torch.no_grad():
            for i in range(0, inputs.size(0), batch_size):
                x_batch = inputs[i : i + batch_size].to(device)
                y_batch = targets[i : i + batch_size].to(device)

                outputs = model(x_batch)

                # Compute the angle from the model's output, every the last one from the sequence of all the samples in the batch 
                output_angles = torch.atan2(outputs[:, -1, 0], outputs[:, -1, 1]).cpu().numpy()
                output_angles_deg = np.degrees(output_angles)

                # Compute the target angle, every the last one from the sequence of all the samples in the batch 
                target_angles = torch.atan2(y_batch[:, -1, 0], y_batch[:, -1, 1]).cpu().numpy()
                target_angles_deg = np.degrees(target_angles)

                # Compute the absolute angle error in degrees
                angle_error = np.abs(output_angles_deg - target_angles_deg)
                print("the angle degree error is", angle_error)
                # angle_error = np.min(angle_error, 360 - angle_error)

                total_angle_error += np.sum(angle_error)

                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
    
        avg_loss = total_loss / (inputs.size(0) // batch_size)
        avg_angle_error = total_angle_error / inputs.size(0) 
        results[noise_std] = avg_angle_error

        print(f"Average Angle Error (in degrees, std={noise_std}): {avg_angle_error:.6f}")
    
    return results

    #print(f"Evaluation Loss (with weight noise std={noise_std}): {avg_loss:.6f}")
    #print(f"Average Angle Error (in degrees): {avg_angle_error:.6f}")
"""

# Evaluation function: For each noise_std, sample 10 different perturbations, evaluate separately, and compute mean and std.
def evaluate_model_with_perturbations(model, dataset, batch_size, noise_std, n_iter=10, original_degree=None):
    """
    Evaluate the model after adding Gaussian noise to its weights.
    For a given noise_std, sample n_iter different perturbations, evaluate each separately, 
    and return the mean and std of the angle error.
    :param model: The trained model.
    :param dataset: Tuple of (inputs, targets).
    :param batch_size: Batch size for evaluation.
    :param noise_std: The standard deviation of the weight noise.
    :param n_iter: Number of perturbation samples.
    :return: (mean_angle_error, std_angle_error)
    """
    inputs, targets = dataset
    criterion = nn.MSELoss()
    angle_errors = []

    # Save the original state dict to restore later
    original_state = copy.deepcopy(model.state_dict())

    for _ in range(n_iter):
        # Restore original weights before each perturbation
        model.load_state_dict(original_state)
        # Add noise perturbation (sample once per evaluation)
        if noise_std > 0:
            add_weight_noise(model, noise_st=noise_std)
        
        model.eval()
        total_angle_error = 0
        count = 0
        with torch.no_grad():
            for i in range(0, inputs.size(0), batch_size):
                x_batch = inputs[i : i + batch_size].to(device)
                y_batch = targets[i : i + batch_size].to(device)
                x_batch = torch.transpose(x_batch, 0, 1)
                original_deg_batch = original_degree[i : i + batch_size].to(device).cpu().numpy()
                outputs, _ = model(x_batch)
                outputs = torch.transpose(outputs, 0, 1)
                # Use the final time step for output and target angles, after training for specific epochs
                output_angles = torch.atan2(outputs[:, -1, 0], outputs[:, -1, 1]).cpu().numpy() 
                output_angles_deg = np.degrees(output_angles) 
                # Compute circular absolute difference
                angle_error = np.abs(output_angles_deg - original_deg_batch) 
                angle_error = np.minimum(angle_error, 360 - angle_error)
                total_angle_error += np.sum(angle_error)
                count += x_batch.size(0)
        avg_angle_error = total_angle_error / count
        angle_errors.append(avg_angle_error)
    
    # Restore the original model state
    model.load_state_dict(original_state)
    
    return np.mean(angle_errors), np.std(angle_errors)

# Evaluation function over a list of noise conditions, returning a dictionary with results.
def evaluate_model_with_noise(model, dataset, batch_size, noise_std_list=[0, 0.003], n_iter=10, original_degree=None):
    results = {}
    for noise_std in noise_std_list:
        mean_error, std_error = evaluate_model_with_perturbations(model, dataset, batch_size, noise_std, n_iter=n_iter, original_degree=original_degree)
        results[noise_std] = {'mean': mean_error, 'std': std_error}
        print(f"Noise std {noise_std}: Mean Angle Error = {mean_error:.6f}, Std = {std_error:.6f}")
    return results

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def plot_outputs_vs_targets(model, dataset, example_idx=0):
    inputs, targets = dataset
    model.eval()
    x_example = inputs[example_idx:example_idx+1].to(device)
    y_example = targets[example_idx:example_idx+1].to(device)
    x_example = torch.transpose(x_example, 0, 1)
    with torch.no_grad():
        outputs, _= model(x_example)
    outputs = torch.transpose(outputs, 0, 1)
    x_example = x_example.cpu().numpy().squeeze()
    y_example = y_example.cpu().numpy().squeeze()
    outputs = outputs.cpu().numpy().squeeze()
    time_steps = np.arange(x_example.shape[0])
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, x_example[:, 0], label='Input sin(theta)', linestyle='--', alpha=0.7)
    plt.plot(time_steps, y_example[:, 0], label='Target sin(theta)', linestyle='-', linewidth=2)
    plt.plot(time_steps, outputs[:, 0], label='Predicted sin(theta)', linestyle='-', linewidth=2)
    plt.xlabel('Time Steps')
    plt.ylabel('sin(theta)')
    plt.title('sin(theta) over Time')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, x_example[:, 1], label='Input cos(theta)', linestyle='--', alpha=0.7)
    plt.plot(time_steps, y_example[:, 1], label='Target cos(theta)', linestyle='-', linewidth=2)
    plt.plot(time_steps, outputs[:, 1], label='Predicted cos(theta)', linestyle='-', linewidth=2)
    plt.xlabel('Time Steps')
    plt.ylabel('cos(theta)')
    plt.title('cos(theta) over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs_vs_targets_with_input_noise.png", dpi=300, bbox_inches="tight") 
    plt.show()


# Hyperparameters and Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
num_samples = 10000
seq_len = 125
input_duration = 5
noise_std = 0.01   # Input noise std
weight_noise_std = 0.003   # Weight noise std for evaluation

# Prepare training dataset
inputs, targets, _ = generate_dataset(batch_size=batch_size, num_samples=num_samples, seq_len=seq_len, input_duration=input_duration, noise_std=noise_std)
dataset = (inputs, targets)

# Prepare evaluation dataset (using a different seq_len if desired)
inputs_eval, targets_eval, original_degrees_eval = generate_dataset(batch_size=batch_size, num_samples=1000, seq_len=250, input_duration=input_duration, noise_std=noise_std)
dataset_eval = (inputs_eval, targets_eval)

# Initialize model
model = RNNNet(input_size=2, hidden_size=300, output_size=2).to(device)

# Train the model and record evaluation metrics every 50 epochs
num_epochs = 200
eval_records = train_model(model=model, dataset=dataset, batch_size=batch_size, num_epchos=num_epochs, learning_rate=1e-3, record_interval=10, eval_dataset=dataset_eval, original_degrees=original_degrees_eval)

# Plot model outputs vs targets for a single example
plot_outputs_vs_targets(model, dataset_eval, example_idx=0)

# breakpoint()
# Convert evaluation records to DataFrame and plot with seaborn
df = pd.DataFrame(eval_records)
print(df.head())

sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
# Plot mean angle error vs epoch, with error bars showing the standard deviation.
ax = sns.lineplot(data=df, x='epoch', y='mean_angle_error', hue='noise_std', marker='o')
ax.errorbar(df['epoch'], df['mean_angle_error'], yerr=df['std_angle_error'], fmt='none', capsize=5, color='black')
ax.set_title("Angle Error vs Training Epochs")
ax.set_xlabel("Epoch")
ax.set_ylabel("Average Angle Error (degrees)")
plt.savefig("angle_error_plot_with_input_noise.png", dpi=300, bbox_inches="tight")
plt.show()

save_model(model, "trained_model_with_input_noise_200epochs.pth")

"""
# Specify the cpu 
device = "cpu"

# Load the trained checkpoint 
checkpoint = torch.load('trained_model_with_input_noise_200epochs.pth')

# Initialize the model
model = RNNNet(input_size=2, hidden_size=300, output_size=2).to(device)

# Load the checkpoints from the saved file
model.load_state_dict(checkpoint)

# Access all parameters
for name, param in model.named_parameters():
    if param.requires_grad:  # Only look at trainable parameters
        print(f"Parameter name: {name}")
        print(f"Weight shape: {param.shape}")
        print(f"Weight values:\n{param.data}\n")
"""



