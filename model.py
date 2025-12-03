"""
Climate to Real Estate Price Prediction Model!!!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ClimateDataset(Dataset):
    """ dataset for climate and price data"""
    def __init__(self, features, prices):
        self.features = torch.FloatTensor(features) #from numpy to tensor
        self.prices = torch.FloatTensor(prices)
    
    def __len__(self):
        return len(self.prices) #sample size
    
    def __getitem__(self, idx): #get item at index
        return self.features[idx], self.prices[idx]


class PriceModel(nn.Module):
    """neural net: 8 inputs -> 64 -> 32 -> 1 output"""
    # 2 linear hidden layers for now
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def train_model(train_loader, val_loader, epochs=50):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PriceModel().to(device) # move the model to gpu if can
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    loss_funtion = nn.MSELoss()
    
    # loss for each epoch
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # training per epoch
        model.train()
        train_loss = 0
        for features, targets in train_loader: # loop through batches
            features, targets = features.to(device), targets.to(device)
            
            predictions = model(features).squeeze()
            loss = loss_funtion(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # validation per epoch
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                predictions = model(features).squeeze()
                val_loss += loss_funtion(predictions, targets).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    return model, train_losses, val_losses


def evaluate_model(model, val_loader):
    """Evaluate model and calculate metrics"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    
    predictions, actuals = [], []
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            preds = model(features).squeeze().cpu().numpy()
            predictions.extend(preds)
            actuals.extend(targets.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # calc the errors
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    r2 = 1 - (np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))
    
    print(f"\nEvaluation Results:")
    print(f"MSE:  {mse:.5f}")
    print(f"RMSE: {rmse:.5f}")
    print(f"MAE:  {mae:.5f}")
    print(f"R²:   {r2:.5f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def plot_results(train_losses, val_losses, predictions, actuals):
    """plots for better visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Predictions vs Actual
    axes[1].scatter(actuals, predictions, alpha=0.5)
    axes[1].plot([actuals.min(), actuals.max()], 
                 [actuals.min(), actuals.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Price Change (%)')
    axes[1].set_ylabel('Predicted Price Change (%)')
    axes[1].set_title('Predictions vs Actual')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    print("\nPlots saved to results.png")



def main():
    """ training pipeline"""
    print("Climate -> Real Estate Price Prediction Model\n")
    
    # load the processed data
    print("loading data...")
    df = pd.read_csv('processed_data.csv')
    print(f"loaded {len(df)} rows")
    
    # filter empty rows
    df = df.dropna(subset=['price_change'])
    print(f"After removing missing targets: {len(df)} rows")
    
    # feature columns,,,,,,check if the the input order is in the same format
    feature_cols = [
        'avg_temp_f',           
        'max_temp_f',           
        'min_temp_f',           
        'precip_inches',        
        'sea_level_change',     #placeholder
        'total_events',         
        'drought_severity',     
        'flood_risk_score'      
    ]
    
    # fill the missing values in features with median
    for col in feature_cols[:-1]:  # skip sea_level_change since we're creating it
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # placeholder for sea_level_change
    df['sea_level_change'] = 0.0
    
    # X and y
    climate_data = df[feature_cols].values
    price_changes = df['price_change'].values
    
    print(f"Features shape: {climate_data.shape}")
    print(f"Target shape: {price_changes.shape}\n")
    
    # split and normalize
    X_train, X_val, y_train, y_val = train_test_split(
        climate_data, price_changes, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Create dataloaders
    train_dataset = ClimateDataset(X_train, y_train)
    val_dataset = ClimateDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")
    
    # train
    model, train_losses, val_losses = train_model(train_loader, val_loader, epochs=50)
    
    # eval
    metrics = evaluate_model(model, val_loader)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    with torch.no_grad():
        val_features = torch.FloatTensor(X_val).to(device)
        predictions = model(val_features).squeeze().cpu().numpy()   # get the preds for plotting
    
    # plot!!!
    plot_results(train_losses, val_losses, predictions, y_val)
    
    print("\nTraining complete! Model saved to best_model.pth")

if __name__ == "__main__":
    main()