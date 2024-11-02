import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from loader import get_dataloader
from Anomaly import AnomalyDetectionModel

def main():
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MVTec Dataset
    dataset_path = "/home/phd/dataset/MVTECDATASET/"
    dataloader = get_dataloader(dataset_path)

    # Define model parameters for ViT and KAN
    vit_params = {
        "img_size": 224,
        "patch_size": 16,
        "num_classes": 2,
        "dim": 768,
        "depth": 12,
        "heads": 12,
        "mlp_dim": 3072,
    }
    
    kan_params = {
        "input_dim": 768,
        "output_dim": 1,
        "poly_degree": 3,
    }

    # Initialize the anomaly detection model and move it to the device (GPU/CPU)
    model = AnomalyDetectionModel(vit_params=vit_params, kan_params=kan_params).to(device)

    # Define loss function and optimizer
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    main()
