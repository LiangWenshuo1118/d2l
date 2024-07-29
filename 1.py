import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import dpdata
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Assuming 10 outputs to represent forces (3x3) and energy (1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

def compute_descriptors(coordinates, Rc=1.5, eta=1.0, Rs=0.0, lambda_angle=-1.0, zeta=1.0):
    O, H1, H2 = coordinates[:, 0, :], coordinates[:, 1, :], coordinates[:, 2, :]
    d_OH1 = torch.norm(O - H1, dim=1)
    d_OH2 = torch.norm(O - H2, dim=1)
    d_H1H2 = torch.norm(H1 - H2, dim=1)

    v_OH1 = H1 - O
    v_OH2 = H2 - O
    cos_theta = torch.sum(v_OH1 * v_OH2, dim=1) / (torch.norm(v_OH1, dim=1) * torch.norm(v_OH2, dim=1))
    theta_H1OH2 = torch.acos(cos_theta)

    def cutoff(r, Rc):
        return 0.5 * (torch.cos(np.pi * r / Rc) + 1) * (r < Rc).float()

    exp_term_OH1 = torch.exp(-eta * (d_OH1 - Rs)**2)
    exp_term_OH2 = torch.exp(-eta * (d_OH2 - Rs)**2)
    G1_O = exp_term_OH1 * cutoff(d_OH1, Rc) + exp_term_OH2 * cutoff(d_OH2, Rc)

    G2_O = (1 + lambda_angle * cos_theta)**zeta * torch.exp(-eta * (d_OH1**2 + d_OH2**2 + d_H1H2**2)) * \
        cutoff(d_OH1, Rc) * cutoff(d_OH2, Rc) * cutoff(d_H1H2, Rc)

    return torch.stack((G1_O, G2_O), dim=1)

# Initialize and load your data system
data = dpdata.LabeledSystem("OUTCAR", fmt="vasp/outcar")
coordinates = torch.tensor(data['coords'], dtype=torch.float32)
forces = torch.tensor(data['forces'], dtype=torch.float32)
energies = torch.tensor(data['energies'], dtype=torch.float32)

# Compute descriptors
descriptors = compute_descriptors(coordinates)

# Split dataset
train_dataset = TensorDataset(descriptors[:500], forces[:500], energies[:500])
test_dataset = TensorDataset(descriptors[500:], forces[500:], energies[500:])

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10)

model = NeuralNetwork(input_size=2)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training function adjusted for 1000 epochs and loss printout
def train(dataloader, model, loss_fn, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X, y_force, y_energy in dataloader:
            pred = model(X)
            pred_force = pred[:, :9]  # first 9 elements are forces
            pred_energy = pred[:, 9]  # last element is energy
            loss_force = loss_fn(pred_force, y_force.reshape(-1,9))
            loss_energy = loss_fn(pred_energy.unsqueeze(1), y_energy.unsqueeze(1))
            loss = loss_force + loss_energy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

train(train_loader, model, loss_fn, optimizer)


def predict_and_return_values(model, dataloader):
    model.eval()
    actual_forces = []
    actual_energies = []
    predictions_forces = []
    predictions_energies = []

    with torch.no_grad():
        for X, y_force, y_energy in dataloader:
            pred = model(X)
            pred_forces = pred[:, :9]  # first 9 elements are forces
            pred_energies = pred[:, 9]  # last element is energy

            predictions_forces.append(pred_forces.numpy())
            predictions_energies.append(pred_energies.numpy())
            actual_forces.append(y_force.numpy())
            actual_energies.append(y_energy.numpy())

    # Convert lists to numpy arrays
    predictions_forces = np.vstack(predictions_forces)
    predictions_energies = np.concatenate(predictions_energies)
    actual_forces = np.vstack(actual_forces)
    actual_energies = np.concatenate(actual_energies)

    return actual_energies, predictions_energies

# Obtain actual and predicted energies
actual_energies, predicted_energies = predict_and_return_values(model, test_loader)

# Plotting the diagonal plot
plt.figure(figsize=(8, 6))
plt.scatter(actual_energies, predicted_energies, color='blue', alpha=0.5)
plt.plot([actual_energies.min(), actual_energies.max()], [actual_energies.min(), actual_energies.max()], 'r--')
plt.xlabel('Actual Energies')
plt.ylabel('Predicted Energies')
plt.title('Energy Prediction Diagonal Plot')
plt.grid(True)
plt.show()
