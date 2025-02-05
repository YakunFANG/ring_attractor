import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 

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

    for _ in range(num_samples):
        # Random angle theta (original angle value)
        theta = np.random.uniform(0, 2 * np.pi)
        theta_degrees = np.degrees(theta)
        sin_cos = np.array([np.sin(theta), np.cos(theta)])
        
        # Input: sin_cos for 'input_duration', zeros for the rest ??? Some questions 
        input_sequence = np.zeros((seq_len, 2))
        input_sequence[:input_duration] = sin_cos

        # Add Gaussian noise to the input
        noise = np.random.normal(0, noise_std, input_sequence.shape)
        input_sequence += noise
        
        # Target: Maintain the same sin_cos for the entire sequence
        target_sequence = np.tile(sin_cos, (seq_len, 1))

        inputs.append(input_sequence)
        targets.append(target_sequence)
    
    return (
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        theta_degrees,
    )

# RNN Model
class VanillaRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=300, output_size=2, recurrent_noise_std=0.01):
        # hidden_size 300: requested by there are in total 300 neurons in the RNN 
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.recurrent_noise_std = recurrent_noise_std

        # Recurrent weights; since the RNN is constructed by multiple cells
        # nn.RNNCell: a single recurrent step for a vanilla RNN, taking the current input and the previous hidden state as inputs, 
        # and computes the next hidden state 
        # specify the activation function to be relu 
        self.rnn_cell = nn.RNNCell(input_size, hidden_size, nonlinearity="relu")

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            h = self.rnn_cell(x[:, t], h)

            #cAdd Gaussian noise to the hidden state
            if self.recurrent_noise_std > 0:
                noise = torch.randn_like(h) * self.recurrent_noise_std
                h = h + noise 
            
            outputs.append(self.output_layer(h))

        return torch.stack(outputs, dim=1)
    
# Function to add Gaussian noise to model weights
def add_weight_noise(model, noise_st):
    """
    Add Gaussian noise to the model weights.
    :param model: The model whose weights will be perturbed
    :param noise_std: Standard deviation of the Gaussian noise
    """
    # in pytorch, a model's weights (parameters) are stored as torch.Tensor objects
    # we can access all the parameters using model.parameters(), which includes the weights of the RNN cells and the weights of the output layer or biases 
    # Gaussian noise is generated using torch.randn_like, whcih creates a tensor of teh same shape as the input tensor
    # The noise is added to the model weights using param.add_() 
    with torch.no_grad(): # Disable the gradient tracking
        for param in model.parameters():
            noise_total = 0
            # Perturb the error for 10 times 
            noise_total = torch.randn_like(param) * noise_st
            for i in range(9):
                noise_total += torch.randn_like(param) * noise_st
            noise_average = noise_total/10
            param.add_(noise_average)

# Training Function
def train_model(model, dataset, batch_size, num_epchos, learning_rate=1e-3, record_interval=50, eval_dataset=[]):
    """
    Train the RNN model on the dataset.
    :param model: The RNN model.
    :param dataset: Tuple of (inputs, targets).
    :param batch_size: Tuple of (inputs, targets).
    :param num_epochs: Number of epchos for training
    :param learning_rate: Learning rate for optimizer.
    :param eval_dataset: dataset used for evaluation process 
    """
    inputs, targets = dataset
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    eval_records = []

    for epcho in range(num_epchos):
        model.train()

        # Shuffle the datatset for each epoch ???
        permutation = torch.randperm(inputs.size(0))
        inputs = inputs[permutation]
        targets = targets[permutation]

        epoch_loss = 0
        for i in range(0, inputs.size(0), batch_size):
            x_batch = inputs[i : i + batch_size].to(device)
            y_batch = targets[i : i + batch_size].to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epcho + 1}/{num_epchos}, Loss: {epoch_loss / (inputs.size(0) // batch_size):.6f}")
        
        if (epcho + 1) % record_interval == 0:
            eval_results = evaluate_model_with_noise(model, eval_dataset, batch_size)
            for noise_std, angle_error in eval_results.items():
                eval_records.append({
                    'epoch': epcho + 1,
                    'noise_std': noise_std,
                    'angle_error': angle_error
                })
    
    return eval_records

# Evaluation Function
def evaluate_model_with_noise(model, dataset, batch_size, noise_std_list=[0, 0.003]):
    """
    Evaluate the model after adding Gaussian noise to its weights.
    :param model: The trained model.
    :param dataset: Tuple of (inputs, targets).
    :param batch_size: Batch size for evaluation.
    :param noise_std: Standard deviation of the Gaussian noise added to weights
    """
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
                y_batch = inputs[i : i + batch_size].to(device)

                outputs = model(x_batch)

                # Compute the angle from the model's output
                output_angles = torch.atan2(outputs[:, -1, 0], outputs[:, -1, 1]).cpu().numpy()
                output_angles_deg = np.degrees(output_angles)

                # Compute the input angle (from the first time step)
                input_angles = torch.atan2(x_batch[:, 0, 0], x_batch[:, 0, 1]).cpu().numpy()
                input_angles_deg = np.degrees(input_angles)

                # Compute the absolute angle error in degrees
                angle_error = np.abs(output_angles_deg - input_angles_deg)
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

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
num_samples = 10000
seq_len = 125
input_duration = 5
noise_std = 0.01 # Standard deviation of Gaussian noise for inputs
weight_noise_std = 0.003 # Standard deviation of Gaussian noise for weights 

# Prepare dataset
inputs, targets, original_degree_train = generate_dataset(batch_size=batch_size, num_samples=num_samples, seq_len=seq_len, input_duration=input_duration)
dataset = (inputs, targets)

# Initialize model
model = VanillaRNN().to(device)

inputs, targets, original_degree_eval = generate_dataset(batch_size=batch_size, num_samples=1000, seq_len=250, input_duration=input_duration)
dataset_eval = (inputs, targets)

# Train model
eval_records = train_model(model=model, dataset=dataset, batch_size=batch_size, num_epchos=200, learning_rate=1e-3, eval_dataset=dataset_eval)  

df = pd.DataFrame(eval_records)
print(df.head())

sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.lineplot(data=df, x='epoch', y='angle_error', hue='noise_std', marker='o')
ax.set_title("Angle Error vs Training Epochs")
ax.set_xlabel("Epoch")
ax.set_ylabel("Average Angle Error (degrees)")
plt.savefig("angle_error_plot_with_input_error_rnn_error.png", dpi=300, bbox_inches="tight")
plt.show()


# inputs, targets = generate_dataset(batch_size=batch_size, num_samples=1000, seq_len=500, input_duration=input_duration)
# dataset_eval = (inputs, targets)
# eva_ seq_len = 250+ 
# num_samples 1000 
# Evaluate model with weight noise
# evaluate_model_with_noise(model, dataset, batch_size=batch_size, noise_std=weight_noise_std) 

save_model(model, "/Users/yakunfang/Desktop/Ring_attractor/trained_model_with_input_error_rnn_error_200epchos.pth")
    

