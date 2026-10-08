# %% [code]
# %% [code]
# Import required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import random

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration class
@dataclass
class FlightBERTConfig:
    bit_lengths: Dict[str, int]
    scaling_factors: Dict[str, float]
    sequence_length: int = 10
    prediction_horizon: int = 1
    batch_size: int = 32
    num_epochs: int = 10
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 10
    feedforward_dim: int = 512
    dropout: float = 0.1
    learning_rate: float = 1e-4
    scheduler_patience: int = 4
    scheduler_factor: float = 0.7
    scheduler_min_lr: float = 1e-7
    scheduler_threshold: float = 1e-4

# Binary Encoder
class BinaryEncoder:
    def __init__(self, bit_lengths, scaling_factors):
        self.bit_lengths = bit_lengths
        self.scaling_factors = scaling_factors
        self.total_bits = sum(bit_lengths.values())
        
    def encode(self, values_dict):
        binary_vectors = []
        for feature, value in values_dict.items():
            scaled_value = abs(float(value)) * self.scaling_factors[feature]
            int_value = int(round(scaled_value))
            binary = format(int_value, f'0{self.bit_lengths[feature]}b')
            binary_vector = [int(b) for b in binary]
            binary_vectors.extend(binary_vector)
        return torch.tensor(binary_vectors, dtype=torch.float32)

    def decode(self, binary_tensor):
        start_idx = 0
        decoded_values = {}
        for feature, bits in self.bit_lengths.items():
            feature_bits = binary_tensor[start_idx:start_idx + bits]
            decimal = sum(bit * (2 ** idx) for idx, bit in enumerate(reversed(feature_bits)))
            decoded_values[feature] = decimal / self.scaling_factors[feature]
            start_idx += bits
        return decoded_values

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# FlightBERT Model
class FlightBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.total_bits = sum(config.bit_lengths.values())
        
        self.input_embed = nn.Linear(self.total_bits, config.embed_dim)
        self.pos_encoder = PositionalEncoding(config.embed_dim, config.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.output_proj = nn.Linear(config.embed_dim, self.total_bits)
        
    def forward(self, x):
        x = self.input_embed(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1:, :]
        x = self.output_proj(x)
        return torch.sigmoid(x)

# Dataset class
class FlightTrajectoryDataset(Dataset):
    def __init__(self, df, config: FlightBERTConfig, train=True):
        self.config = config
        self.sequence_length = config.sequence_length
        self.prediction_horizon = config.prediction_horizon
        self.train = train
        self.flight_sequences = []
        
        for flight_id, flight_data in df.groupby('aircraft_registration'):
            flight_data = flight_data.reset_index(drop=True)
            
            if len(flight_data) >= (self.sequence_length + self.prediction_horizon):
                for i in range(len(flight_data) - self.sequence_length - self.prediction_horizon + 1):
                    self.flight_sequences.append({
                        'flight_data': flight_data,
                        'start_idx': i,
                    })
        
        print(f"Created dataset with {len(self.flight_sequences)} sequences")
        
    def _encode_features(self, row):
        features = {
            'longitude': float(row['longitude']),
            'latitude': float(row['latitude']),
            'altitude': float(row['altitude']),
            'ground_speed': float(row['ground_speed']),
            'heading': float(row['heading'])
        }
        return self.config.encoder.encode(features)
    
    def __len__(self):
        return len(self.flight_sequences)
    
    def __getitem__(self, idx):
        sequence = self.flight_sequences[idx]
        flight_data = sequence['flight_data']
        start_idx = sequence['start_idx']
        
        seq_start = start_idx
        seq_end = start_idx + self.sequence_length
        target_idx = seq_end
        
        sequence_data = flight_data.iloc[seq_start:seq_end]
        target_data = flight_data.iloc[target_idx:target_idx + 1]
        
        sequence_tensors = [self._encode_features(row) for _, row in sequence_data.iterrows()]
        target_tensor = self._encode_features(target_data.iloc[0])
        
        sequence_encoded = torch.stack(sequence_tensors)
        target_encoded = target_tensor.unsqueeze(0)
        
        return sequence_encoded, target_encoded

# Loss function
class HybridLoss(nn.Module):
    def __init__(self, encoder, bce_weight=1.0, mse_weight=1.0):
        super().__init__()
        self.encoder = encoder
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        
        with torch.no_grad():
            pred_flat = predictions.view(-1, predictions.size(-1))
            target_flat = targets.view(-1, targets.size(-1))
            
            pred_decoded_list = [self.encoder.decode(p) for p in pred_flat.detach().cpu()]
            target_decoded_list = [self.encoder.decode(t) for t in target_flat.cpu()]
            
            pred_decoded = torch.tensor([
                [d['longitude'], d['latitude'], d['altitude'], 
                 d['ground_speed'], d['heading']] 
                for d in pred_decoded_list
            ], device=predictions.device)
            
            target_decoded = torch.tensor([
                [d['longitude'], d['latitude'], d['altitude'], 
                 d['ground_speed'], d['heading']] 
                for d in target_decoded_list
            ], device=targets.device)
            
            mse_loss = self.mse(pred_decoded, target_decoded)
        
        return self.bce_weight * bce_loss + self.mse_weight * mse_loss

def load_flight_data(csv_path, skip_range=6):
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, header=None)
    df.columns = [
        'mode-s', 'latitude', 'longitude', 'heading', 'altitude', 'ground_speed',
        'transponder', 'radar', 'aircraft_type', 'aircraft_registration',
        'timestamp', 'origin', 'dest', 'flight_number', 'status', 'vertical_speed',
        'flight_number2', 'end', 'last_airport'
    ]
    
    needed_columns = ['latitude', 'longitude', 'altitude', 'ground_speed', 'heading', 
                     'aircraft_registration', 'timestamp']
    df = df[needed_columns].dropna()
    print(f"Loaded {len(df)} points after removing NaN")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['altitude'] = df['altitude'] * 0.3048
    
    df = df[(df['ground_speed'] != 0) & (df['altitude'] != 0)]
    df = df.sort_values(['aircraft_registration', 'timestamp'])
    df = df.groupby('aircraft_registration').apply(
        lambda x: x.iloc[::skip_range]).reset_index(drop=True)
    
    min_points = 30
    df = df.groupby('aircraft_registration').filter(
        lambda x: len(x) >= min_points).reset_index(drop=True)
    
    print(f"Final dataset: {len(df)} points from {len(df['aircraft_registration'].unique())} flights")
    return df

def create_data_loaders(csv_path, config):
    df = load_flight_data(csv_path)
    
    unique_flights = df['aircraft_registration'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_flights)
    
    train_split = int(0.7 * len(unique_flights))
    val_split = int(0.85 * len(unique_flights))
    
    train_flights = unique_flights[:train_split]
    val_flights = unique_flights[train_split:val_split]
    test_flights = unique_flights[val_split:]
    
    train_df = df[df['aircraft_registration'].isin(train_flights)].copy().reset_index(drop=True)
    val_df = df[df['aircraft_registration'].isin(val_flights)].copy().reset_index(drop=True)
    test_df = df[df['aircraft_registration'].isin(test_flights)].copy().reset_index(drop=True)
    
    print(f"\nData split stats:")
    print(f"Train: {len(train_df)} points from {len(train_flights)} flights")
    print(f"Val: {len(val_df)} points from {len(val_flights)} flights")
    print(f"Test: {len(test_df)} points from {len(test_flights)} flights")
    
    train_dataset = FlightTrajectoryDataset(train_df, config, train=True)
    val_dataset = FlightTrajectoryDataset(val_df, config, train=False)
    test_dataset = FlightTrajectoryDataset(test_df, config, train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, print_interval=100):
    model.train()
    total_loss = 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                desc=f'Epoch {epoch}')
    
    for batch_idx, (inputs, targets) in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % print_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def make_single_prediction(model, test_loader, encoder, device='cpu'):
    model.eval()
    
    test_iter = iter(test_loader)
    sequence_data, target_data = next(test_iter)
    
    batch_size = sequence_data.size(0)
    sample_idx = random.randint(0, batch_size - 1)
    
    sequence_data = sequence_data[sample_idx:sample_idx+1].to(device)
    target_data = target_data[sample_idx:sample_idx+1].to(device)
    
    with torch.no_grad():
        prediction = model(sequence_data)
        
    pred_decoded = encoder.decode(prediction.squeeze().cpu().numpy())
    actual_decoded = encoder.decode(target_data.squeeze().cpu().numpy())
    
    errors = {
        key: abs(pred_decoded[key] - actual_decoded[key]) 
        for key in pred_decoded.keys()
    }
    
    return {
        'predicted': pred_decoded,
        'actual': actual_decoded,
        'errors': errors
    }


# %% [code]
def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model from checkpoint and continue training"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# %% [code]
# Now initialize everything and run training
config = FlightBERTConfig(
    bit_lengths={
        'longitude': 13,
        'latitude': 13,
        'altitude': 16,
        'ground_speed': 10,
        'heading': 9
    },
    scaling_factors={
        'longitude': 1e2,
        'latitude': 1e2,
        'altitude': 1,
        'ground_speed': 1,
        'heading': 1
    },
    sequence_length=9,
    prediction_horizon=10,
    batch_size=512,
    num_epochs=8
)
checkpoint_path = "/kaggle/input/lastmodel/checkpoint_epoch_6.pt"
# Create encoder
config.encoder = BinaryEncoder(config.bit_lengths, config.scaling_factors)

# Load data
csv_path = "/kaggle/input/dataset-voli/Italy_24-04-03.csv"
train_loader, val_loader, test_loader = create_data_loaders(csv_path, config)

# Initialize model and training components
model = FlightBERT(config).to(device)
criterion = HybridLoss(config.encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Initialize scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=config.scheduler_factor,
    patience=config.scheduler_patience,
    threshold=config.scheduler_threshold,
    min_lr=config.scheduler_min_lr,
    verbose=True
)
if checkpoint_path:
    load_checkpoint(model, optimizer, scheduler, checkpoint_path)
# Training loop
print("\nStarting training...")
train_losses = []
val_losses = []

try:
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Training phase
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation phase
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch} Summary:")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Save checkpoint every epoch
        if (epoch + 1) % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'checkpoint_epoch_{epoch+1}.pt')
        
except KeyboardInterrupt:
    print("\nTraining interrupted by user!")
except Exception as e:
    print(f"\nError during training: {str(e)}")
    raise e

print("\nTraining completed!")

# %% [code]
# Make a prediction
results = make_single_prediction(model, test_loader, config.encoder, device)

# Print results
print("\nPrediction Results:")
print("\nActual values:")
for key, value in results['actual'].items():
    print(f"{key}: {value:.3f}")

print("\nPredicted values:")
for key, value in results['predicted'].items():
    print(f"{key}: {value:.3f}")

print("\nAbsolute errors:")
for key, value in results['errors'].items():
    print(f"{key}: {value:.3f}")