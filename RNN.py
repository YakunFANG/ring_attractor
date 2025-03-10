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
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean

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
        # noise = np.random.normal(0, noise_std, (seq_len, 2))
        # input_sequence += noise 
        
        # Target: Maintain the same sin_cos for the entire sequence
        target_sequence = np.tile(sin_cos, (seq_len, 1))
        target_sequence[0] = [0, 0]

        inputs.append(input_sequence)
        targets.append(target_sequence)
        original_degrees.append(theta_degree)
    
    return (
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(original_degrees, dtype=torch.float32),
    )

class LowRankCTRNN(nn.Module):
    """COntinuous-time RNN with low-rank recurrent weights.
    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        rank: Rank of the recurrent weight matrix
        dt: discretization time step in ms
    """

    def __init__(self, input_size, hidden_size, rank, rnn_std, dt=10, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.recurrent_noise_std = rnn_std
        self.tau = 100
        self.alpha = dt / self.tau if dt is not None else 1

        # Input to hidden layer
        self.input2h = nn.Linear(input_size, hidden_size)

        # Low-rank recurrent weights
        self.U = nn.Parameter(torch.empty(hidden_size, rank))  # U: hidden_size x rank
        self.V = nn.Parameter(torch.empty(rank, hidden_size))  # V: rank x hidden_size

        # Initialize U and V using Kaiming (He) normal initialization
        nn.init.kaiming_normal_(self.U, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.V, mode='fan_in', nonlinearity='relu')

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)
    
    def recurrence(self, input, hidden):
        """Run network for one time step.
        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)
        Outputs:
            h_new: tensor of shape (batch, hidden_size)
        """
        # Compute recurrent dynamics using low-rank RNNs
        recurrent_term = (self.U @ (self.V @ hidden.T)).T
        h_new = torch.relu(self.input2h(input) + recurrent_term)
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new
    
    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)
        
        output = []
        for i in range(input.size(0)):
            if self.recurrent_noise_std > 0:
                noise = np.sqrt(2*(1/self.alpha)*self.recurrent_noise_std**2)*torch.normal(mean=0.0, std=1, size=hidden.shape).to(hidden.device)
                hidden += noise
            
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)
        
        output = torch.stack(output, dim=0)
        return output, hidden 

class LowRankRNNNet(nn.Module):
    """Recurrent network model with low-rank RNN.
    Parameters:
        input_size: int, input size
        hidden_size: int, hidden size
        rank: int, rank of the recurrent weight matrix
        output_size: int, output size
    """

    def __init__(self, input_size, hidden_size, rank, output_size, rnn_std, **kwargs):
        super().__init__()
        self.rnn = LowRankCTRNN(input_size, hidden_size, rank, rnn_std, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        out = self.fc(rnn_output)
        return out, rnn_output 


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

    def __init__(self, input_size, hidden_size, rnn_std, dt=10, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size 
        # Add rnn noise
        self.recurrent_noise_std = rnn_std
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
            if self.recurrent_noise_std > 0:
                # Add rnn noise
                noise = np.sqrt(2*(1/self.alpha)*self.recurrent_noise_std**2)*torch.normal(mean=0.0, std=1, size=hidden.shape).to(hidden.device)
                hidden = hidden + noise 

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
    def __init__(self, input_size, hidden_size, output_size, rnn_std, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, rnn_std, **kwargs)

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
            noise = torch.normal(mean=0.0, std=noise_st, size=param.shape, device=param.device)
            param.add_(noise)

def add_weight_noise_rnn_weights(model, noise_st):
    """
    Add Gaussian noise to the recurrent weights of the model.
    :param model: The model whose recurrent weights will be perturbed.
    :param noise_st: Standard deviation of the Gaussian noise.
    """
    with torch.no_grad():
        # Perturb only the recurrent weights (h2h layer)
        for name, param in model.named_parameters():
            if 'V' or 'U' in name:  # Only perturb parameters in the h2h layer
                noise = torch.normal(mean=0.0, std=noise_st, size=param.shape, device=param.device)
                param.add_(noise)

# Training Function
def train_model(model, dataset, batch_size, learning_rate=1e-3, record_interval=20, eval_dataset=None, original_degrees=None, rnn_std=None, L1_lambda=1e-3): 
    # inputs, targets = dataset 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
    criterion = nn.MSELoss() 
    eval_records = []
    
    start_time = time.time()
    loss_container = []
    model_container = []
    
    whether_complete = 0
    epcho = 0
    max_iteration = 50000
    
    while whether_complete == 0 and (epcho+1) <= max_iteration: 
        inputs, targets, _ = generate_dataset(batch_size=batch_size, num_samples=num_samples, seq_len=seq_len, input_duration=input_duration, noise_std=0.01)
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
            outputs, hiddens = model(x_batch)
            outputs = torch.transpose(outputs, 0, 1)
            # print("the shape of the outputs is", outputs.shape)
            loss = criterion(outputs, y_batch)
            norm = (1/(batch_size*300))
            L1_penalty = norm * hiddens.abs().sum() 
            # L2_lambda = 1e-3
            loss = loss + L1_lambda * L1_penalty
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # avg_loss = epoch_loss / (inputs.size(0) // batch_size)
        print(f"Epoch {epcho + 1}, Loss: {epoch_loss:.6f}, Time: {(time.time() - start_time):.1f}")

        # Evaluate every record_interval epochs if eval_dataset is provided
        if eval_dataset is not None and (epcho + 1) % 10 == 0:
            # Evaluate for each noise condition separately
            eval_results = evaluate_model_with_noise(model, eval_dataset, batch_size, noise_std_list=[0, weight_noise_std], original_degree=original_degrees)
            #print("the eval_results.items() are", eval_results.items())
            for noise_std, stats in eval_results.items():
                eval_records.append({
                    'epoch': epcho + 1,
                    'noise_std': noise_std,
                    'mean_angle_error': stats['mean'],
                    'std_angle_error': stats['std']
                })
            
            stats_need = eval_results[0]
            loss_container.append(stats_need['loss'])
            model_now = model
            model_container.append(model_now)

        if len(loss_container) == 200: 
            whether_complete = 1
            criteria = loss_container[0]
            for loss_value in loss_container:
                if loss_value < criteria:
                    whether_complete = 0
                    print("the current 200 loss values are", loss_container)
                    loss_container.pop(0)
                    model_container.pop(0)
                    break 
        
        epcho += 1
    
    return eval_records, model_container

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

    total_loss = []

    for _ in range(n_iter):
        current_total_loss = 0 
        # Restore original weights before each perturbation
        model.load_state_dict(original_state)
        # Add noise perturbation (sample once per evaluation)
        if noise_std > 0:
            add_weight_noise_rnn_weights(model, noise_st=noise_std)
        
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

                loss = criterion(outputs, y_batch)
                current_total_loss += loss

                # Use the final time step for output and target angles, after training for specific epochs
                output_angles = torch.atan2(outputs[:, -1, 0], outputs[:, -1, 1]).cpu().numpy() 
                output_angles_deg = np.degrees(output_angles) 
                # Compute circular absolute difference
                angle_error = np.abs(output_angles_deg - original_deg_batch) 
                angle_error = np.minimum(angle_error, 360 - angle_error)
                total_angle_error += np.sum(angle_error)
                count += x_batch.size(0)
        
        total_loss.append(current_total_loss)
        avg_angle_error = total_angle_error / count
        angle_errors.append(avg_angle_error)
    
    # Restore the original model state
    model.load_state_dict(original_state)
    total_loss_cpu = [item.cpu() if isinstance(item, torch.Tensor) else item for item in total_loss]
    mean_loss = np.mean(total_loss_cpu)  
    
    return np.mean(angle_errors), np.std(angle_errors), mean_loss

# Evaluation function over a list of noise conditions, returning a dictionary with results.
def evaluate_model_with_noise(model, dataset, batch_size, noise_std_list=[0, 0.003], n_iter=10, original_degree=None):
    results = {}
    #breakpoint()
    for noise_std in noise_std_list:
        mean_error, std_error, loss_eval = evaluate_model_with_perturbations(model, dataset, batch_size, noise_std, n_iter=n_iter, original_degree=original_degree)
        results[noise_std] = {'mean': mean_error, 'std': std_error, 'loss': loss_eval}
        print(f"Noise std {noise_std}: Mean Angle Error = {mean_error:.6f}, Std = {std_error:.6f}")
    #print("the current results are:", results)
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
    plt.savefig("outputs_vs_targets_low_rank_input_noise_rnn_noise_std_0.2.png", dpi=300, bbox_inches="tight") 
    plt.show()

def generate_representation_matrix(model, network_name, num_angles=360, seq_len=40000):
    """
    Generate a representation matrix for the given model.
    :param model: The trained model. 
    :param num_angles: Number of angles to sample.
    :param seq_len: Length of each sequence (time steps).
    :return Representation matrix of shape [N, M], where N is the number of neurons and M is the number of angles. 
    """
    model.eval()
    # angles = np.linspace(0, 2 * np.pi, num_angles)
    # representation_matrix = []
    # count = 0
    
    # Change the codes to make the input can be put in only one batch
    # sin_cos = np.array([np.sin(theta), np.cos(theta)])
    # input_sequence = np.zeros((seq_len, 2))
    # input_sequence[0] = sin_cos
    # input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(1).to(device)
    input_tensor, target_tensor, _ = generate_dataset(batch_size=360, num_samples=num_angles, seq_len=seq_len, input_duration=input_duration, noise_std=noise_std)
    
    input_tensor = torch.transpose(input_tensor, 0, 1) 
    
    with torch.no_grad():
        out, rnn_output = model(input_tensor)
        # print("the shape of rnn_output is", rnn_output.shape)
        final_activity = rnn_output[-1].cpu().numpy()
        final_output = out[-1].cpu().numpy()
        # print("the current final activity is", len(final_activity[0])) 
        # representation_matrix.append(final_activity)
    # print(f"Processing model: {network_name}, angle: {theta}, index: {count}")
    
    # print("the shape of representation matrix is", representation_matrix.size) 
    representation_matrix = np.array(final_activity).T 
    final_activity_ouput = np.array(final_output) 
    
    # print("the shape of it is", representation_matrix.shape)

    return representation_matrix, final_activity_ouput

# Capture the activity of the RNN at each time step for each angle.
def generate_representation_matrix_evaluate(model, network_name, num_angles=360, seq_len=40000):
    """
    Generate a representation matrix for the given model. 
    :param model: The trained model. 
    :param num_angles: Number of angles to sample. 
    :param seq_len: Length of each sequence (time steps). 
    :return Representation matrix of shape [N, M], where N is the number of neurons and M is the number of angles. 
    """
    model.eval()
    input_tensor, target_tensor, _ = generate_dataset(batch_size=num_angles, num_samples=num_angles, seq_len=seq_len, input_duration=input_duration, noise_std=noise_std)

    input_tensor = torch.transpose(input_tensor, 0, 1)

    with torch.no_grad():
        _, rnn_output = model(input_tensor)
        # rnn_output shape: (seq_len, batch_size, hidden_size)
        # We want to capture the activity over time for each angle
        # Shape: (seq_len, num_angles, hidden_size) 
        activity_over_time = rnn_output.cpu().numpy()

    return activity_over_time 

# Visualize the activity over time for each angle. 
# We will create a new function to visualize the activity OVER TIME for each angle using PCA. 
def visualize_activity_over_time(activity_over_time, angle_idx, title):
    """
    Visualize the activity over time for a specific angle using PCA.
    :param activity_over_time: Activity over time for all angles, shape (seq_len, num_angles, hidden_size)
    :param angle_idx: Index of the angles to visualize.
    :param title: Title for the plot.
    """
    from sklearn.decomposition import PCA

    # Extract the activity for the specific angle
    # Shape: (seq_len, hidden_size)
    angle_activity = activity_over_time[:, angle_idx, :]

    # Apply PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    # Shape: (seq_len, 3)
    reduced_activity = pca.fit_transform(angle_activity)

    # Create a time axis (assuming time steps are sequential)
    time_steps = np.arange(reduced_activity.shape[0])

    # Plot the three principal components over time
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, reduced_activity[:, 0], label="PC1", color="b", alpha=0.8)
    plt.plot(time_steps, reduced_activity[:, 1], label="PC2", color="g", alpha=0.8)
    plt.plot(time_steps, reduced_activity[:, 2], label="PC3", color="r", alpha=0.8)

    plt.xlabel("Time Steps")
    plt.ylabel("PCA Component Value")
    plt.title(f"{title} (Angle {angle_idx}) - 2D PCA Line Plot")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"activity_over_time_angle_{angle_idx}_{title}_2D.png", dpi=300, bbox_inches="tight")
    plt.show()


def participation_ratio(representation_matrix):
    """
    Compute the participation ratio of the representation matrix.
    :param representation_matrix: Representation matrix of shape [N, M]
    :param return Participation ratio
    """
    cov_matrix = np.cov(representation_matrix)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    participation_ratio = np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2)
    return participation_ratio

def visualize_representation_matrices(representation_matrices, title, network_names):
    """
    Visulaize the representation matrix using PCA.
    :param representation_matrix: Representation matrix of shape [N, M].
    :param param title: Title for the plot.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    reduced_matrices = [pca.fit_transform(matrix.T) for matrix in representation_matrices]


    colors = ['r', 'b', 'y', 'g']

    for i, reduced_matrix in enumerate(reduced_matrices):
       fig = plt.figure(figsize=(12, 10))
       ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

       # Create a 3D scatter plot
       ax.scatter(
           reduced_matrix[:, 0],  # x-coordinates (PC1)
           reduced_matrix[:, 1],  # y-coordinates (PC2)
           reduced_matrix[:, 2],  # z-coordinates (PC3)
           color=colors[i],       # Color for this network
           label=network_names[i],  # Label for this network
           alpha=0.7              # Transparency
        )

       # Add title and labels
       ax.set_title(title)
       ax.set_xlabel("PC1")
       ax.set_ylabel("PC2")
       ax.set_zlabel("PC3")  # Label for the z-axis

       # Add a legend
       ax.legend()

       # Save the figure
       plt.savefig(f"neural_representation_{network_names[i]}.png", dpi=300, bbox_inches="tight")
       plt.show()

from sklearn.cluster import KMeans, DBSCAN

def find_distinct_attractors(final_states, method='dbscan', n_clusters=None, eps=None, min_samples=5):
    """
    Cluster the final states to identify distinct attractors.
    :param final_states: Final states of shape (num_angles, hidden_size), (N, M) in this case (360, 300)
    :param method: Clustering method ('kmeans' or 'dbscan')  
    :param n_clusters: Number of clusters for K-Means (if method='kmeans') 
    :param eps: Maximum distance between two samples for DBSCAN (if method='dbscan'), if None, it will be estimated
    :param min_samples: Minimum number of samples in a neighborhood for DBSCAN (if method='dbscan')
    :return: Cluster labels and number of distinct attractors
    """
    
    # Estimate circumference by summing pairwise distances between consecutive points
    circumference = sum(euclidean(final_states[i], final_states[i + 1]) for i in range(len(final_states) - 1))
    circumference += euclidean(final_states[-1], final_states[0])  # Close the loop
    
    # Scale `eps` if not provided
    if eps is None:
        eps = circumference / len(final_states)  # Use average pairwise distance

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(final_states)
    
    # Count the number of distinct attractors (excluding noise points labeled as -1)
    n_attractors = len(set(labels)) - (1 if -1 in labels else 0)
    
    return labels, n_attractors

def visualize_output(output_weight_matrix, representation_matrix, model_name):
    """
    Visualize the 2D projection of the representation matrix using the output weight matrix.
    :param output_weight_matrix: The output weight matrix (N x 2) from the model's output layer.
    :param representation_matrix: The representation matrix (N x hidden_size).
    :param model_name: Name of the model for the plot title.
    """
    # Project the representation matrix using the output weight matrix
    projection = np.dot(output_weight_matrix, representation_matrix)
    projection = projection.T

    plt.figure(figsize=(8, 6))
    plt.scatter(projection[:, 0], projection[:, 1], alpha=0.6)
    plt.title(f"2D Projection of Representation Matrix ({model_name})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()
    plt.savefig(f"2D_Projection_of_Representation_Matrix_{model_name}.png")


def visualize_all_receptive_fields(representation_matrices):
    """
    Visualize the receptive fields of all neurons in a heatmap.

    :param representation_matrices: List of neural representation matrices, 
                                    each of shape (num_neurons, num_angles) -> (300, 360)
    """

    network_names = ['Low_rank_std_0', 'Low_rank_std_0.15']
    
    for idx, representation_matrix in enumerate(representation_matrices):
        
        num_neurons, num_angles = representation_matrix.shape
        angles = np.linspace(0, 360, num_angles, endpoint=False)  # X-axis: angles

        # Sort neurons based on their peak response angle for better visualization
        sorted_indices = np.argsort(np.argmax(representation_matrix, axis=1))
        sorted_representation = representation_matrix[sorted_indices]

        # Create the heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(sorted_representation, aspect='auto', cmap='gray_r', 
                   extent=[0, 360, 0, num_neurons])  # ✅ Corrected Y-Axis Range
        plt.colorbar(label="Activity")
        plt.xlabel("Input Angle (degree)")
        plt.ylabel("Neuron (sorted)")
        plt.title(f"Receptive Fields of All Neurons - {network_names[idx]}")

        # Save the figure
        plt.savefig(f"Receptive_Fields_{network_names[idx]}.png", dpi=300, bbox_inches="tight")
        plt.show()

def compute_pca_variance(activity_over_time):
    """
    Compute the number of principal components needed to explain 80% of the variance.
    :param activity_over_time: Activity over time for all angles, shape (seq_len, num_angles, hidden_size)
    :return: NUmber of PCs needed to explain 80% of the variance
    """
    from sklearn.decomposition import PCA

    # Reshape the activity to (num_samples, hidden_size)
    num_samples = activity_over_time.shape[0] * activity_over_time.shape[1]
    hidden_size = activity_over_time.shape[2]
    activity_reshaped = activity_over_time.reshape(num_samples, hidden_size)
    
    # Perform PCA
    pca = PCA()
    pca.fit(activity_reshaped)
 
    # Compute the cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the number of components needed to explain 80% of the variance
    n_components = np.argmax(cumulative_variance >= 0.8) + 1

    return n_components, cumulative_variance


# Hyperparameters and Device
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print("the current device is", device)


batch_size = 64
num_samples = 10000
seq_len = 125
input_duration = 5
noise_std = 0.01   # Input noise std
weight_noise_std = 0.003   # Weight noise std for evaluation


""" 
# Prepare training dataset
inputs, targets, _ = generate_dataset(batch_size=batch_size, num_samples=num_samples, seq_len=seq_len, input_duration=input_duration, noise_std=noise_std)
# inputs.to(device)
# targets.to(device)
dataset = (inputs, targets)
dataset_gpu = tuple(tensor.to(device) for tensor in dataset)
# breakpoint()
#print("device is", dataset_gpu[0].device)
#print("device is", dataset_gpu[1].device)

# Prepare evaluation dataset (using a different seq_len if desired)
inputs_eval, targets_eval, original_degrees_eval = generate_dataset(batch_size=batch_size, num_samples=1000, seq_len=250, input_duration=input_duration, noise_std=noise_std)
original_degrees_eval.to(device)
dataset_eval = (inputs_eval, targets_eval)
dataset_eval_gpu = tuple(tensor.to(device) for tensor in dataset_eval)

# Initialize model
model = LowRankRNNNet(input_size=2, hidden_size=300, rank=3, output_size=2, rnn_std=0.2).to(device)

# Train the model and record evaluation metrics every 50 epochs
eval_records, model_container = train_model(model=model, dataset=dataset_gpu, batch_size=batch_size, learning_rate=1e-3, record_interval=10, eval_dataset=dataset_eval_gpu, original_degrees=original_degrees_eval, rnn_std=0.2)

model = model_container[0] 
save_model(model_container[0], "trained_low_rank_model_input_noise_rnn_noise_std_0.2.pth")
# Plot model outputs vs targets for a single example
plot_outputs_vs_targets(model, dataset_eval, example_idx=0) 

# breakpoint()
# Convert evaluation records to DataFrame and plot with seaborn
df = pd.DataFrame(eval_records)
print(df.head())

sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
# Plot mean angle error vs epoch, with error bars showing the standard deviation.
ax = sns.lineplot(data=df, x='epoch', y='mean_angle_error', hue='noise_std', marker='o')
ax.errorbar(df['epoch'], df['mean_angle_error'], yerr=df['std_angle_error'], fmt='none', capsize=5, color='black')
ax.set_title("Angle Error vs Training Epochs")
ax.set_xlabel("Epoch")
ax.set_ylabel("Average Angle Error (degrees)")
plt.savefig("angle_error_plot_low_rank_input_noise_rnn_noise_std_0.2.png", dpi=300, bbox_inches="tight")
plt.show()
""" 


network_names=['low_rank_std_0', 'low_rank_std_0.01', 'low_rank_std_0.1', 'low_rank_std_0.15', 'low_rank_std_0.2']
print("the program is executed as expected")
# Task 1: Evaluate the Performance with Random Weight Noise
# Generate the test datasets
inputs_test, targets_test, original_degrees_test = generate_dataset(batch_size = batch_size, num_samples=1000, seq_len=250, input_duration=input_duration, noise_std=noise_std)
dataset_test = (inputs_test, targets_test)

# Load the trained checkpoint 
checkpoint1 = torch.load('trained_low_rank_model_input_noise_rnn_noise_std_0.pth')
#checkpoint2 = torch.load('trained_low_rank_model_input_noise_rnn_noise_std_0.01.pth')
#checkpoint3 = torch.load('trained_low_rank_model_input_noise_rnn_noise_std_0.1.pth')
checkpoint4 = torch.load('trained_low_rank_model_input_noise_rnn_noise_std_0.15.pth')
#checkpoint5 = torch.load('trained_low_rank_model_input_noise_rnn_noise_std_0.2.pth')

# Initialize the model
model1 = LowRankRNNNet(input_size=2, hidden_size=300, rank=3, output_size=2, rnn_std=0).to(device)
#model2 = LowRankRNNNet(input_size=2, hidden_size=300, rank=3, output_size=2, rnn_std=0).to(device)
#model3 = LowRankRNNNet(input_size=2, hidden_size=300, rank=3, output_size=2, rnn_std=0).to(device)
model4 = LowRankRNNNet(input_size=2, hidden_size=300, rank=3, output_size=2, rnn_std=0).to(device)
#model5 = LowRankRNNNet(input_size=2, hidden_size=300, rank=3, output_size=2, rnn_std=0).to(device)


# Load the checkpoints from the saved file
model1.load_state_dict(checkpoint1)
#model2.load_state_dict(checkpoint2)
#model3.load_state_dict(checkpoint3)
model4.load_state_dict(checkpoint4)
#model5.load_state_dict(checkpoint5)

"""
# Evaluate the model with varying weight noise
noise_std_list = [0, 0.01, 0.03, 0.1]
test_results_1 = evaluate_model_with_noise(model1, dataset_test, batch_size, noise_std_list, n_iter=10, original_degree=original_degrees_test)
#test_results_2 = evaluate_model_with_noise(model2, dataset_test, batch_size, noise_std_list, n_iter=10, original_degree=original_degrees_test)
#test_results_3 = evaluate_model_with_noise(model3, dataset_test, batch_size, noise_std_list, n_iter=10, original_degree=original_degrees_test)
test_results_4 = evaluate_model_with_noise(model4, dataset_test, batch_size, noise_std_list, n_iter=10, original_degree=original_degrees_test)
#test_results_5 = evaluate_model_with_noise(model5, dataset_test, batch_size, noise_std_list, n_iter=10, original_degree=original_degrees_test)

# Access all parameters
# for name, param in model.named_parameters():
#    if param.requires_grad:  # Only look at trainable parameters
#        print(f"Parameter name: {name}")
#        print(f"Weight shape: {param.shape}")
#        print(f"Weight values:\n{param.data}\n")


# Convert test results to DataFrame and plot with seaborn
# breakpoint()
df_test1 = pd.DataFrame(test_results_1).T.reset_index()
df_test1.columns = ['noise_std', 'mean_angle_error', 'std_angle_error', 'loss']
#df_test2 = pd.DataFrame(test_results_2).T.reset_index()
#df_test2.columns = ['noise_std', 'mean_angle_error', 'std_angle_error', 'loss']
#df_test3 = pd.DataFrame(test_results_3).T.reset_index()
#df_test3.columns = ['noise_std', 'mean_angle_error', 'std_angle_error', 'loss']
df_test4 = pd.DataFrame(test_results_4).T.reset_index()
df_test4.columns = ['noise_std', 'mean_angle_error', 'std_angle_error', 'loss']
#df_test5 = pd.DataFrame(test_results_5).T.reset_index()
#df_test5.columns = ['noise_std', 'mean_angle_error', 'std_angle_error', 'loss']
#print(df_test.shape)
#print(df_test['mean_angle_error'].shape)
#print(df_test['std_angle_error'].shape)
#print(df_test)
import pandas as pd

# Add a 'network' column to each DataFrame
df_test1['network'] = 'Low_rank_activity_std_0'
#df_test2['network'] = 'Low_rank_activity_std_0.01'
#df_test3['network'] = 'Low_rank_activity_std_0.1'
df_test4['network'] = 'Low_rank_activity_std_0.15'
#df_test5['network'] = 'Low_rank_activity_std_0.2'

# Combine the DataFrames
df_combined = pd.concat([df_test1, df_test4], ignore_index=True)

# Print the combined DataFrame
print(df_combined)

# Define colors for each network
colors = ['skyblue', 'lightgreen'] # 'salmon', 'gold', 'red']
networks = df_combined['network'].unique()

print("the networks are", networks) 

# Create the plot
plt.figure(figsize=(20, 15))

# Plot bars for each network
x = np.arange(len(df_test1))  # X-axis positions
width = 0.2  # Width of each bar

for i, network in enumerate(networks):
    df_network = df_combined[df_combined['network'] == network]
    plt.bar(x + i * width, df_network['mean_angle_error'], width, yerr=df_network['std_angle_error'], 
            capsize=5, label=network, color=colors[i])


# Customize the plot
plt.xticks(x + width, df_test1['noise_std'])
plt.title("Angle Error vs Weight Noise Std (Comparison of Three Networks)")
plt.xlabel("Weight Noise Std")
plt.ylabel("Average Angle Error (degrees)")
plt.legend(title='Network')
plt.savefig("angle_error_vs_weight_noise_low_rank_combined.png", dpi=300, bbox_inches="tight")
plt.show()
""" 

#representation_matrix1 = generate_representation_matrix_evaluate(model1, network_names[0])
#np.save('representation_matrix1_low_rank_over_time.npy', representation_matrix1)
#representation_matrix4 = generate_representation_matrix_evaluate(model4, network_names[3])
#np.save('representation_matrix4_low_rank_over_time.npy', representation_matrix4)
#print(f"Shape of representation_matrix1: {representation_matrix1.shape}")
#print(f"Shape of representation_matrix4: {representation_matrix4.shape}")

#participation_ratio_1=participation_ratio(representation_matrix1)
#participation_ratio_4=participation_ratio(representation_matrix4)

#print(f"Participation Ratio for std_0: {participation_ratio_1}")
#print(f"Participation Ratio for std_0.15: {participation_ratio_4}")

representation_matrix1 = np.load('representation_matrix1_low_rank.npy')
representation_matrix4 = np.load('representation_matrix4_low_rank.npy')
print("the shape of representation_matrix1", representation_matrix1.shape)
print("the shape of representation_matrix4", representation_matrix4.shape)

visualize_all_receptive_fields([representation_matrix1, representation_matrix4])

""" 
angles_to_visualize = [0, 90, 180, 270]
for angle_idx in angles_to_visualize:
   visualize_activity_over_time(representation_matrix1, angle_idx, title="low_rank_std_0")
#    visualize_activity_over_time(representation_matrix2, angle_idx, title="Input_Noise")
#    visualize_activity_over_time(representation_matrix3, angle_idx, title="Input_Noise_Rnn_Noise")
   visualize_activity_over_time(representation_matrix4, angle_idx, title="low_rank_std_0.15")
"""
   
"""
n_components1, cumulative_variance1 = compute_pca_variance(representation_matrix1)
n_components4, cumulative_variance4 = compute_pca_variance(representation_matrix4)


print(f"Number of PCs needed to explain 80% of the variance for {network_names[0]}: {n_components1}")
print(f"Number of PCs needed to explain 80% of the variance for {network_names[3]}: {n_components4}")
"""


""" 
network_names=['Original', 'Input_noise', 'Input_noise_Rnn_noise', 'Input_noise_Rnn_noise_std_0.05']
# print("executing task 2")
# Task 2: generate representation matrix for the trained model
#representation_matrix1, output1 = generate_representation_matrix(model1, network_names[0])
#np.save('representation_matrix1.npy', representation_matrix1)
#np.save('output1.npy', output1)

# breakpoint()
# print("shape of it", len(representation_matrix1[0]))
#representation_matrix2, output2 = generate_representation_matrix(model2, network_names[1])
#np.save('representation_matrix2.npy', representation_matrix2)
#np.save('output2.npy', output2)
#representation_matrix3, output3 = generate_representation_matrix(model3, network_names[2])
#np.save('representation_matrix3.npy', representation_matrix3)
#np.save('output3.npy', output3)
#representation_matrix4, output4 = generate_representation_matrix(model4, network_names[3])
#np.save('representation_matrix4.npy', representation_matrix4)
#np.save('output4.npy', output4)

# Load representation matrices
representation_matrix1 = np.load('representation_matrix1.npy')
representation_matrix2 = np.load('representation_matrix2.npy')
representation_matrix3 = np.load('representation_matrix3.npy')
representation_matrix4 = np.load('representation_matrix4.npy')

# Extract output weight matrices from the models
output_weight_matrix1 = model1.fc.weight.detach().cpu().numpy()
output_weight_matrix2 = model2.fc.weight.detach().cpu().numpy()
output_weight_matrix3 = model3.fc.weight.detach().cpu().numpy()
output_weight_matrix4 = model4.fc.weight.detach().cpu().numpy()

participation_ratio_1=participation_ratio(representation_matrix1)
participation_ratio_2=participation_ratio(representation_matrix2)
participation_ratio_3=participation_ratio(representation_matrix3)
participation_ratio_4=participation_ratio(representation_matrix4)


print(f"Participation Ratio for original: {participation_ratio_1}")
print(f"Participation Ratio for input_error: {participation_ratio_2}")
print(f"Participation Ratio for input_error&rnn_error: {participation_ratio_3}")
print(f"Participation Ratio for input_error&rnn_error_std_0.05: {participation_ratio_4}")


# Visualize the representation matrices in 3D
visualize_representation_matrices (
    [representation_matrix1, representation_matrix2, representation_matrix3, representation_matrix4], title="Representation matrices in 3D",
    network_names=[f"Original_PR_{participation_ratio_1}", f"Input_noise_PR_{participation_ratio_2}", f"Input_noise_Rnn_noise_std_0.003_PR_{participation_ratio_3}", f"Input_noise_Rnn_noise_std_0.05_PR_{participation_ratio_4}"]
)


visualize_output(output_weight_matrix1, representation_matrix1, "Original")
visualize_output(output_weight_matrix2, representation_matrix2, "Input_noise")
visualize_output(output_weight_matrix3, representation_matrix3, "Input_noise_RNN_noise_0.003")
visualize_output(output_weight_matrix4, representation_matrix4, "Input_noise_RNN_noise_0.05")
"""

""" 
network_names=['Original', 'Input_noise', 'Input_noise_Rnn_noise_std_0.003', 'Input_noise_Rnn_noise_std_0.05']
# Task 2: generate representation matrix for the trained model
#representation_matrix1 = generate_representation_matrix_evaluate(model1, network_names[0])
#np.save('representation_matrix1_over_time.npy', representation_matrix1)
#representation_matrix2 = generate_representation_matrix_evaluate(model2, network_names[1])
#np.save('representation_matrix2_over_time.npy', representation_matrix2)
#representation_matrix3 = generate_representation_matrix_evaluate(model3, network_names[2])
#np.save('representation_matrix3_over_time.npy', representation_matrix3)
#representation_matrix4 = generate_representation_matrix_evaluate(model4, network_names[3])
#np.save('representation_matrix4_over_time.npy', representation_matrix4)


representation_matrix1 = np.load('representation_matrix1.npy')
representation_matrix2 = np.load('representation_matrix2.npy')
representation_matrix3 = np.load('representation_matrix3.npy')
representation_matrix4 = np.load('representation_matrix4.npy')

for idx, matrix in enumerate([representation_matrix1, representation_matrix2, representation_matrix3, representation_matrix4]):
    print(f"the number of non-zero neuron for the model_{network_names[idx]} ", np.count_nonzero(np.any(matrix != 0, axis=1)))
"""
    
"""
# Visualize activity over time for a few angles (this is a list of angle indecies)
angles_to_visualize = [0, 90, 180, 270]

#for angle_idx in angles_to_visualize:
#    visualize_activity_over_time(representation_matrix1, angle_idx, title="Original")
#    visualize_activity_over_time(representation_matrix2, angle_idx, title="Input_Noise")
#    visualize_activity_over_time(representation_matrix3, angle_idx, title="Input_Noise_Rnn_Noise")
#    visualize_activity_over_time(representation_matrix4, angle_idx, title="Input_Noise_Rnn_Noise_Std_0.05")
print("the shape of the representation matrix is", representation_matrix1.shape)

# Find distinct attractors using DBSCAN
labels1, n_attractors1 = find_distinct_attractors(representation_matrix1.T, eps=0.005, min_samples=1)
labels2, n_attractors2 = find_distinct_attractors(representation_matrix2.T, eps=0.005, min_samples=1)
labels3, n_attractors3 = find_distinct_attractors(representation_matrix3.T, eps=0.005, min_samples=1)
labels4, n_attractors4 = find_distinct_attractors(representation_matrix4.T, eps=0.005, min_samples=1)

visualize_tuning_curves([representation_matrix1, representation_matrix2, representation_matrix3, representation_matrix4])

# Print the number of distinct attractors for each model
print(f"Number of distinct attractors for Original Network: {n_attractors1}")
print(f"Number of distinct attractors for Input Noise Network: {n_attractors2}")
print(f"Number of distinct attractors for Input Noise & RNN Noise Network_0.003: {n_attractors3}")
print(f"Number of distinct attractors for Input Noise & RNN Noise_std_0.05 Network: {n_attractors4}")
"""

# Access all parameters
#for name, param in model1.named_parameters():
#   if param.requires_grad:  # Only look at trainable parameters
#       print(f"Parameter name: {name}")
#       print(f"Weight shape: {param.shape}")
#       print(f"Weight values:\n{param.data}\n")
