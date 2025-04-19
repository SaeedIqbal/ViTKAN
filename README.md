# ViTKAN: Vision Transformer with Kolmogorov - Arnold Network for Anomaly Detection

## 1. Description
This repository (`ViTKAN`) focuses on anomaly detection, combining a Vision Transformer (ViT) and a Kolmogorov - Arnold Network (KAN). The Vision Transformer extracts features from images, while the KAN further processes these features to generate anomaly scores. 

### Key Components
- **Vision Transformer (ViT)**: Divides images into patches, applies patch embedding, positional encoding, and uses a Transformer encoder with self - attention mechanism. It also includes a multichannel autoencoder for spatial - temporal feature extraction.
- **Kolmogorov - Arnold Network (KAN)**: Utilizes Chebyshev and Legendre polynomial layers to process the features extracted by the ViT.
- **Anomaly Detection Model**: Integrates the ViT and KAN to generate anomaly scores for input images.

## 2. Usage

### Prerequisites
- Python 3.x
- PyTorch
- Other necessary libraries (can be installed via `pip` or `conda` based on the imports in the code)

### Installation
Clone the repository to your local machine:
```bash
git clone <repository_url>
cd ViTKAN
```

### Training
The main training script is `main.py`. Before running the training, make sure to set the correct dataset path in the script.

```bash
python main.py
```

### Parameters
- **ViT Parameters**: You can modify the parameters for the Vision Transformer in the `vit_params` dictionary in `main.py`. Key parameters include `img_size`, `patch_size`, `num_classes`, `dim`, `depth`, `heads`, and `mlp_dim`.
- **KAN Parameters**: The parameters for the Kolmogorov - Arnold Network can be adjusted in the `kan_params` dictionary in `main.py`. The main parameters are `input_dim`, `output_dim`, and `poly_degree`.

### Model Initialization
The anomaly detection model is initialized in `main.py` using the following code:
```python
model = AnomalyDetectionModel(vit_params=vit_params, kan_params=kan_params).to(device)
```

### Training Loop
The training loop in `main.py` iterates over a specified number of epochs (`num_epochs`). For each epoch, it computes the loss, performs backpropagation, and updates the model's parameters.

## 3. Datasets

### Training Dataset
The model is trained on the MVTec Dataset. To use this dataset, you need to set the correct dataset path in the `main.py` script. By default, the path is set to `/home/phd/dataset/MVTECDATASET/`.

### Validation Datasets
- **Real - IAD**: This dataset is used for validation purposes. You need to integrate it into the validation process by modifying the code to load and evaluate the model on this dataset.
- **Open Images V7**: Similar to Real - IAD, Open Images V7 is used for validation. You need to adjust the code to load and validate the model using this dataset.

## 4. Code Structure
- `ViT.py`: Contains the implementation of the Vision Transformer.
- `Legendre - Chebyshev.py`: Defines the Kolmogorov - Arnold Network with Chebyshev and Legendre polynomial functions.
- `Anomaly.py`: Implements the anomaly detection model that combines the ViT and KAN.
- `main.py`: The main training script that initializes the model, defines the loss function and optimizer, and runs the training loop.
- `ChebyshevPolynomialLayer.py` and `LegendrePolynomialLayer.py`: Implement the Chebyshev and Legendre polynomial layers respectively.
- `kan.py`: Contains the implementation of the KANLinear layer and the KAN model.

## 5. Future Work
The current README states that the developers are working to finalize and remove the issues of DACL with KAN. Future improvements may include:
- Further optimization of the model architecture.
- Integration of more advanced data augmentation techniques.
- Improved handling of the validation datasets for more accurate evaluation.
