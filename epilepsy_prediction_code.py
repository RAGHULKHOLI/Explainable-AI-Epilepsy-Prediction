import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import transforms, datasets 
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights 
from torch.utils.data import DataLoader, random_split, Dataset 
import os 
import zipfile 
import shutil 
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 
import cv2 # Used for image resizing in Grad-CAM visualization 
from matplotlib.cm import get_cmap # Import get_cmap 
import pydicom # For DICOM file handling 
 
# --- 0. Google Colab Setup --- 
# Ensure you have a GPU runtime enabled in Colab: 
# Runtime -> Change runtime type -> Hardware accelerator -> GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Using device: {device}") 
 
# --- Helper for DICOM Loading --- 
def load_dicom_image(dicom_path): 
    """ 
    Loads a DICOM file, extracts pixel data, and converts it to a PIL Image. 
    Handles 16-bit pixel data by normalizing to 0-255. 
    """ 
    try: 
        dicom_data = pydicom.dcmread(dicom_path) 
        pixel_array = dicom_data.pixel_array 
 
        # Normalize pixel array to 0-255 range if it's not already 
        if pixel_array.dtype != np.uint8: 
            # Apply windowing if available in DICOM metadata 
            if 'WindowWidth' in dicom_data and 'WindowCenter' in dicom_data: 
                window_width = dicom_data.WindowWidth 
                window_center = dicom_data.WindowCenter 
                min_val = window_center - window_width / 2 

                max_val = window_center + window_width / 2 
                pixel_array = np.clip(pixel_array, min_val, max_val) 
                pixel_array = ((pixel_array - min_val) / (max_val - min_val)) * 255.0 
            else: 
                # Simple min-max normalization if no windowing info 
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255.0 
            pixel_array = pixel_array.astype(np.uint8) 
 
        # Convert to PIL Image (ensure it's grayscale 'L' mode if single channel) 
        if pixel_array.ndim == 2: 
            return Image.fromarray(pixel_array, mode='L') 
        elif pixel_array.ndim == 3 and pixel_array.shape[2] == 3: 
            return Image.fromarray(pixel_array, mode='RGB') 
        else: 
            raise ValueError(f"Unsupported pixel array dimensions: {pixel_array.shape}") 
 
    except Exception as e: 
        print(f"Error loading DICOM file {dicom_path}: {e}") 
        return None 
 
# --- Custom Dataset to handle multiple image types (JPG, JPEG, PNG, DICOM) --- 
class MultiFormatMRIDataset(Dataset): 
    def __init__(self, root_dir, transform=None): 
        self.root_dir = root_dir 
        self.transform = transform 
        self.image_paths = [] 
        self.labels = [] 
        self.class_to_idx = {} 
 
        # Collect image paths and labels 
        for class_name in os.listdir(root_dir): 
            class_path = os.path.join(root_dir, class_name) 
            if os.path.isdir(class_path): 
                if class_name not in self.class_to_idx: 
                    self.class_to_idx[class_name] = len(self.class_to_idx) 
 
                for filename in os.listdir(class_path): 
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.dcm')): 
                        self.image_paths.append(os.path.join(class_path, filename)) 
                        self.labels.append(self.class_to_idx[class_name]) 

        self.classes = sorted(self.class_to_idx.keys(), key=lambda x: self.class_to_idx[x]) 
        print(f"Dataset initialized with {len(self.image_paths)} images. Classes: {self.classes}") 
 
    def __len__(self): 
        return len(self.image_paths) 
 
    def __getitem__(self, idx): 
        img_path = self.image_paths[idx] 
        label = self.labels[idx] 
 
        if img_path.lower().endswith('.dcm'): 
            image = load_dicom_image(img_path) 
        else: 
            # Open as RGB to ensure 3 channels for ImageNet pre-trained models 
            # Even if original is grayscale, PIL's convert('RGB') will repeat channels. 
            image = Image.open(img_path).convert('RGB') 
 
        if image is None: 
            # Fallback for corrupted/unreadable images: return a dummy image 
            print(f"Warning: Could not load {img_path}. Returning a dummy image.") 
            dummy_img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) 
            image = Image.fromarray(dummy_img_array, mode='RGB') 
            # In a real application, you might want to log this and potentially skip the sample. 
 
        if self.transform: 
            image = self.transform(image) 
 
        return image, label 
 
 
# --- 1. Data Preparation and Loading --- 
 
# Define the path to your uploaded zip file 
zip_path = '/content/MRI DATASET file 1.zip' 
extract_path = '/content/MRI_DATASET_EXTRACTED' 
 
# Check if the zip file exists and extract it 
if os.path.exists(zip_path): 
    print(f"Extracting {zip_path}...") 

    with zipfile.ZipFile(zip_path, 'r') as zip_ref: 
        zip_ref.extractall(extract_path) 
    print("Extraction complete.") 
else: 
    print(f"'{zip_path}' not found. Creating a dummy dataset for demonstration.") 
    # Create dummy data if the zip file is not found, to make the code runnable 
    os.makedirs(os.path.join(extract_path, 'train', 'healthy'), exist_ok=True) 
    os.makedirs(os.path.join(extract_path, 'train', 'epilepsy'), exist_ok=True) 
    os.makedirs(os.path.join(extract_path, 'test', 'healthy'), exist_ok=True) 
    os.makedirs(os.path.join(extract_path, 'test', 'epilepsy'), exist_ok=True) 
 
    # Generate dummy images 
    for i in range(50): # 50 healthy train 
        img_array = np.random.randint(0, 256, (256, 256), dtype=np.uint8) 
        Image.fromarray(img_array).save(os.path.join(extract_path, 'train', 'healthy', f'healthy_{i}.jpg')) 
    for i in range(50): # 50 epilepsy train 
        img_array = np.random.randint(0, 256, (256, 256), dtype=np.uint8) 
        # Add a simple "abnormality" for visual distinction in dummy data 
        img_array[50:100, 50:100] = 255 
        Image.fromarray(img_array).save(os.path.join(extract_path, 'train', 'epilepsy', f'epilepsy_{i}.jpg')) 
    for i in range(10): # 10 healthy test 
        img_array = np.random.randint(0, 256, (256, 256), dtype=np.uint8) 
        Image.fromarray(img_array).save(os.path.join(extract_path, 'test', 'healthy', f'healthy_test_{i}.jpg')) 
    for i in range(10): # 10 epilepsy test 
        img_array = np.random.randint(0, 256, (256, 256), dtype=np.uint8) 
        img_array[50:100, 50:100] = 255 
        Image.fromarray(img_array).save(os.path.join(extract_path, 'test', 'epilepsy', f'epilepsy_test_{i}.jpg')) 
    print("Dummy dataset created.") 
 
 
# Define image transformations for EfficientNetB0 
# EfficientNetB0 was pre-trained on ImageNet, which consists of 3-channel (RGB) images. 
# MRI scans are typically grayscale. We convert them to 3 channels by repeating the 
# single channel, and then apply ImageNet's standard normalization. 
data_transforms = { 
    'train': transforms.Compose([ 
        transforms.Resize((224, 224)),         # Resize to EfficientNetB0 input size 
        # Grayscale to 3 channels is handled by PIL.Image.convert('RGB') in the dataset 
        transforms.RandomHorizontalFlip(),      # Data augmentation: horizontal flip 
        transforms.RandomRotation(10),          # Data augmentation: slight rotation 

        transforms.ToTensor(),                  # Convert PIL Image to PyTorch Tensor (scales to [0,1]) 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization 
    ]), 
    'val': transforms.Compose([ 
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ]), 
    'test': transforms.Compose([ 
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ]), 
    'inference': transforms.Compose([ # For single image inference, no augmentation 
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ]) 
} 
 
# Load datasets using MultiFormatMRIDataset 
try: 
    image_datasets = { 
        'train': MultiFormatMRIDataset(os.path.join(extract_path, 'train'), data_transforms['train']), 
        'test': MultiFormatMRIDataset(os.path.join(extract_path, 'test'), data_transforms['test']) 
    } 
 
    # Split the training dataset into training and validation sets 
    # Changed from 0.9 to 0.7 for training 
    train_size = int(0.7 * len(image_datasets['train'])) 
    val_size = len(image_datasets['train']) - train_size 
    train_dataset, val_dataset = random_split(image_datasets['train'], [train_size, val_size]) 
 
    # Create DataLoaders 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2) 
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2) 
    test_loader = DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=2) 
 
    class_names = image_datasets['train'].classes 
    num_classes = len(class_names) 

    print(f"Classes found: {class_names}") 
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: 
{len(image_datasets['test'])}") 
 
except Exception as e: 
    print(f"Error loading dataset: {e}") 
    print("Please ensure your 'MRI DATASET.zip' contains 'train' and 'test' folders,") 
    print("each with 'healthy' and 'epilepsy' subfolders containing images.") 
    print("The dummy dataset will be used if the zip was not found or was malformed.") 
    # Fallback to a simpler dummy dataset if ImageFolder fails 
    # This ensures the code can still run for demonstration 
    class SimpleDummyDataset(Dataset): 
        def __init__(self, num_samples=100, transform=None, label_ratio=0.5): 
            self.num_samples = num_samples 
            self.transform = transform 
            self.images = [] 
            self.labels = [] 
            for i in range(num_samples): 
                img_array = np.random.randint(0, 256, (256, 256), dtype=np.uint8) 
                label = 0 if np.random.rand() < label_ratio else 1 
                if label == 1: # Add abnormality for epilepsy class 
                    img_array[50:100, 50:100] = 255 
                self.images.append(Image.fromarray(img_array, mode='L').convert('RGB')) # Ensure RGB for 
dummy 
                self.labels.append(label) 
            self.classes = ['healthy', 'epilepsy'] # Define classes for dummy 
        def __len__(self): return self.num_samples 
        def __getitem__(self, idx): 
            img = self.images[idx] 
            label = self.labels[idx] 
            if self.transform: img = self.transform(img) 
            return img, label 
 
    print("Using simple dummy dataset for demonstration.") 
    train_dataset = SimpleDummyDataset(num_samples=100, transform=data_transforms['train']) 
    val_dataset = SimpleDummyDataset(num_samples=20, transform=data_transforms['val']) 
    test_dataset = SimpleDummyDataset(num_samples=30, transform=data_transforms['test']) 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2) 
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2) 

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2) 
    class_names = ['healthy', 'epilepsy'] 
    num_classes = 2 
 
 
# --- 2. Model Definition (EfficientNetB0 with Transfer Learning) --- 
print("Loading pre-trained EfficientNetB0 model...") 
# Load EfficientNetB0 with pre-trained ImageNet weights 
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1) 
 
# Modify the classifier head for our specific binary classification task 
# EfficientNetB0's classifier is a sequential block. The last layer is at index 1. 
num_ftrs = model.classifier[1].in_features 
model.classifier[1] = nn.Linear(num_ftrs, num_classes) # Replace with a new linear layer for 2 classes 
 
# Move the model to the specified device (GPU if available) 
model = model.to(device) 
print(f"Model loaded and adapted for {num_classes} classes.") 
 
# --- 3. Loss Function and Optimizer --- 
# CrossEntropyLoss is suitable for classification tasks. 
# Adam optimizer is a robust choice for deep learning. 
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001) # Learning rate can be tuned 
 
# --- 4. Training Loop --- 
num_epochs = 10 # Adjust as needed based on dataset size and convergence 
print(f"Starting training for {num_epochs} epochs...") 
 
for epoch in range(num_epochs): 
    # Training phase 
    model.train() # Set model to training mode 
    running_loss = 0.0 
    correct_train = 0 
    total_train = 0 
 
    for inputs, labels in train_loader: 
        inputs, labels = inputs.to(device), labels.to(device) 
 
        optimizer.zero_grad() # Zero the parameter gradients 

        outputs = model(inputs) # Forward pass 
        loss = criterion(outputs, labels) # Calculate loss 
        loss.backward() # Backward pass: compute gradients 
        optimizer.step() # Update model parameters 
 
        running_loss += loss.item() * inputs.size(0) 
        _, predicted = torch.max(outputs.data, 1) 
        total_train += labels.size(0) 
        correct_train += (predicted == labels).sum().item() 
 
    epoch_train_loss = running_loss / len(train_dataset) 
    epoch_train_accuracy = 100 * correct_train / total_train 
 
    # Validation phase 
    model.eval() # Set model to evaluation mode 
    val_loss = 0.0 
    correct_val = 0 
    total_val = 0 
    with torch.no_grad(): # Disable gradient calculations for validation 
        for inputs, labels in val_loader: 
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs) 
            loss = criterion(outputs, labels) 
            val_loss += loss.item() * inputs.size(0) 
            _, predicted = torch.max(outputs.data, 1) 
            total_val += labels.size(0) 
            correct_val += (predicted == labels).sum().item() 
 
    epoch_val_loss = val_loss / len(val_dataset) 
    epoch_val_accuracy = 100 * correct_val / total_val 
 
    print(f"Epoch {epoch+1}/{num_epochs}: " 
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}% | " 
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}%") 
 
print("Training complete.") 
 
# --- 5. Final Evaluation on Test Set --- 
print("\nEvaluating model performance on the test set...") 

model.eval() # Set model to evaluation mode 
correct_test = 0 
total_test = 0 
all_preds = [] 
all_labels = [] 
 
with torch.no_grad(): 
    for inputs, labels in test_loader: 
        inputs, labels = inputs.to(device), labels.to(device) 
        outputs = model(inputs) 
        _, predicted = torch.max(outputs.data, 1) 
        total_test += labels.size(0) 
        correct_test += (predicted == labels).sum().item() 
        all_preds.extend(predicted.cpu().numpy()) 
        all_labels.extend(labels.cpu().numpy()) 
 
test_accuracy = 100 * correct_test / total_test 
print(f"Final Test Accuracy: {test_accuracy:.2f}%") 
 
# Optional: More detailed evaluation metrics 
from sklearn.metrics import classification_report, confusion_matrix 
print("\nClassification Report:") 
print(classification_report(all_labels, all_preds, target_names=class_names)) 
 
print("\nConfusion Matrix:") 
print(confusion_matrix(all_labels, all_preds)) 
 
# --- 6. Grad-CAM Implementation for Visualization and Severity Scaling --- 
 
def calculate_severity_score(heatmap, model_confidence): 
    """ 
    Calculates a conceptual severity score based on Grad-CAM heatmap properties 
    and model confidence. 
    """ 
    # Define a threshold for high activation in the heatmap (e.g., top 30% intensity) 
    # Using percentile makes it adaptive to different heatmap distributions 
    heatmap_threshold_value = np.percentile(heatmap, 70) if heatmap.size > 0 else 0.0 
 
    activated_pixels = np.sum(heatmap > heatmap_threshold_value) 
    total_pixels = heatmap.size 

    area_ratio = activated_pixels / total_pixels if total_pixels > 0 else 0.0 
 
    # Calculate mean intensity only for activated pixels 
    if activated_pixels > 0: 
        mean_intensity = np.mean(heatmap[heatmap > heatmap_threshold_value]) 
    else: 
        mean_intensity = 0.0 
 
    # Peak intensity is the maximum value in the heatmap 
    peak_intensity = np.max(heatmap) if heatmap.size > 0 else 0.0 
 
    # Combine metrics into a single score (weights can be tuned) 
    # This is a conceptual score. Real severity requires extensive clinical correlation and validation. 
    severity_score = (area_ratio * 0.4) + (mean_intensity * 0.3) + (model_confidence * 0.3) 
 
    # Categorize severity based on the combined score 
    if severity_score < 0.3: 
        severity_category = "Mild" 
    elif 0.3 <= severity_score < 0.7: 
        severity_category = "Moderate" 
    else: 
        severity_category = "Severe" 
 
    return severity_score, severity_category 
 
 
def visualize_cam(model, img_tensor, target_class_idx, class_names, device): 
    """ 
    Generates and visualizes a Grad-CAM heatmap on the input image. 
    Also calculates and displays a severity score if the prediction is 'epilepsy'. 
 
    Args: 
        model (torch.nn.Module): The trained deep learning model. 
        img_tensor (torch.Tensor): The preprocessed input image tensor. 
        target_class_idx (int): The index of the target class for Grad-CAM (e.g., 'epilepsy' class index). 
        class_names (list): List of class names (e.g., ['healthy', 'epilepsy']). 
        device (torch.device): The device the model is on (cuda or cpu). 
    """ 
    model.eval() # Set model to evaluation mode 

    # Store gradients and activations 
    activations = None 
    gradients = None 
 
    # Define hooks to capture forward activations and backward gradients 
    def save_activations(module, input, output): 
        nonlocal activations 
        activations = output 
 
    def save_gradients(module, grad_input, grad_output): 
        nonlocal gradients 
        gradients = grad_output[0] 
 
    # Register hooks to the last convolutional layer of EfficientNetB0 
    # The 'features' block is a Sequential module, and features[-1] is the last block. 
    target_layer = model.features[-1] 
    hook_handle_fwd = target_layer.register_forward_hook(save_activations) 
    hook_handle_bwd = target_layer.register_full_backward_hook(save_gradients) 
 
    # Perform forward pass 
    img_input = img_tensor.unsqueeze(0).to(device) # Add batch dimension 
    output = model(img_input) 
    probabilities = torch.softmax(output, dim=1) 
    predicted_class = torch.argmax(probabilities, dim=1).item() 
    predicted_class_name = class_names[predicted_class] 
    model_confidence = probabilities[0, predicted_class].item() # Confidence for the predicted class 
 
    # If the predicted class is healthy, just display the image and state it's normal. 
    if predicted_class_name == 'healthy': 
        print(f"\nPrediction for this image: {predicted_class_name} scan (Confidence: {model_confidence:.2f}).") 
        original_img_np = img_tensor.cpu().numpy().transpose(1, 2, 0) 
        # Denormalize for display (reverse ImageNet normalization) 
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([0.229, 0.224, 0.225]) 
        original_img_np = original_img_np * std + mean 
        original_img_np = np.clip(original_img_np, 0, 1) # Clip to ensure valid RGB range 
 
        plt.figure(figsize=(6, 6)) 
        plt.imshow(original_img_np) 

        plt.title(f"Predicted: {predicted_class_name} (Confidence: {model_confidence:.2f})") 
        plt.axis('off') 
        plt.show() 
        # Remove hooks after use 
        hook_handle_fwd.remove() 
        hook_handle_bwd.remove() 
        return 
 
    # If predicted class is 'epilepsy', proceed with Grad-CAM and severity calculation 
    print(f"\nPrediction for this image: {predicted_class_name} scan (Confidence: {model_confidence:.2f}). 
Highlighting potential epileptic areas...") 
 
    # Zero gradients and perform backward pass for the target class (epilepsy) 
    model.zero_grad() 
    # Select the score for the predicted class to backpropagate through 
    target_output = output[0, predicted_class] # Use predicted_class for Grad-CAM 
    target_output.backward() 
 
    # Get the gradients and activations 
    if gradients is None or activations is None: 
        print("Error: Could not retrieve gradients or activations. Grad-CAM skipped.") 
        hook_handle_fwd.remove() 
        hook_handle_bwd.remove() 
        return 
 
    gradients = gradients.cpu().data.numpy()[0] 
    activations = activations.cpu().data.numpy()[0] 
 
    # Compute weights (global average pooling of gradients) 
    weights = np.mean(gradients, axis=(1, 2)) 
 
    # Create the CAM heatmap 
    cam = np.zeros(activations.shape[1:], dtype=np.float32) 
    for i, w in enumerate(weights): 
        cam += w * activations[i] 
 
    # Apply ReLU to the CAM (only positive contributions) 
    cam = np.maximum(cam, 0) 
 
    # Normalize heatmap to [0, 1] 

    if np.max(cam) > 0: 
        cam = cam / np.max(cam) 
    else: 
        print("Warning: CAM is all zeros. Cannot normalize.") 
        cam = np.zeros(cam.shape, dtype=np.float32) # Keep as zeros 
 
    # Calculate severity score using the generated heatmap and model's confidence 
    severity_score, severity_category = calculate_severity_score(cam, model_confidence) 
    print(f"Calculated Severity: {severity_category} (Score: {severity_score:.2f})") 
 
    # Resize CAM to original image size for overlay 
    heatmap_resized = cv2.resize(cam, (img_tensor.shape[2], img_tensor.shape[1])) 
    # Use a colormap for the heatmap 
    cmap = get_cmap('jet') 
    heatmap_colored = cmap(heatmap_resized)[:,:,:3] # Get RGB channels from colormap 
 
    # Convert original image tensor to numpy array for display 
    original_img_np = img_tensor.cpu().numpy().transpose(1, 2, 0) 
    # Denormalize for display (reverse ImageNet normalization) 
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    original_img_np = original_img_np * std + mean 
    original_img_np = np.clip(original_img_np, 0, 1) # Clip to ensure valid RGB range 
 
    # Blend the heatmap with the original image 
    superimposed_img = original_img_np * 0.6 + heatmap_colored * 0.4 # Adjust blending ratio as needed 
    superimposed_img = np.clip(superimposed_img, 0, 1) # Clip to ensure valid RGB range 
 
    # Display results with colorbar 
    fig, axes = plt.subplots(1, 2, figsize=(14, 7)) # Increased figure size for better visualization 
 
    axes[0].imshow(original_img_np) 
    axes[0].set_title(f"Original Image\nPredicted: {predicted_class_name} (Conf: {model_confidence:.2f})") 
    axes[0].axis('off') 
 
    im = axes[1].imshow(superimposed_img) 
    axes[1].set_title(f"Epilepsy Area Highlighted (Grad-CAM)\nSeverity: {severity_category} (Score: 
{severity_score:.2f})") 
    axes[1].axis('off') 

    # Add colorbar for heatmap intensity 
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=axes[1], orientation='vertical', fraction=0.046, 
pad=0.04) 
    cbar.set_label('Grad-CAM Intensity') 
 
    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels 
    plt.show() 
 
    # Remove hooks after use 
    hook_handle_fwd.remove() 
    hook_handle_bwd.remove() 
 
# --- 7. Demonstration of Epilepsy Area Detection and Severity --- 
 
# Find the index for the 'epilepsy' class 
try: 
    epilepsy_class_idx = class_names.index('epilepsy') 
    healthy_class_idx = class_names.index('healthy') 
except ValueError: 
    print("Error: 'epilepsy' or 'healthy' class not found in dataset. Cannot proceed with visualization.") 
    epilepsy_class_idx = 1 # Default to 1 if not found (assuming binary classification) 
    healthy_class_idx = 0 
 
 
# --- Function to predict and visualize for a single image file --- 
def predict_and_visualize_single_image(model, image_path, class_names, device, epilepsy_class_idx): 
    """ 
    Loads a single image from a given path, preprocesses it, performs inference, 
    and visualizes the Grad-CAM heatmap with severity if epilepsy is detected. 
    """ 
    print(f"\n--- Processing image: {image_path} ---") 
 
    # Load image based on file type 
    img = None 
    if image_path.lower().endswith('.dcm'): 
        img = load_dicom_image(image_path) 
    elif image_path.lower().endswith(('.jpg', '.jpeg', '.png')): 
        try: 
            img = Image.open(image_path).convert('RGB') # Ensure 3 channels for consistency 
        except Exception as e: 

            print(f"Error opening image {image_path}: {e}") 
            return 
    else: 
        print(f"Unsupported file format for {image_path}. Skipping.") 
        return 
 
    if img is None: 
        print(f"Could not load image from {image_path}.") 
        return 
 
    # Apply inference transforms 
    img_tensor = data_transforms['inference'](img) 
 
    # Perform visualization and severity calculation 
    visualize_cam(model, img_tensor, epilepsy_class_idx, class_names, device) 
 
# --- Demonstration with uploaded images --- 
print("\n--- Demonstrating Epilepsy Area Detection and Severity on Provided Images ---") 
 
# List of uploaded image paths (ensure these files are present in the Colab environment) 
uploaded_image_paths = [ 
    'Brain Tumor.jpg', 
    'Focal Cortical Dysplasia.jpg', 
    'Left_Hippocampal_Sclerosis_on_MRI.jpg', 
    'Stroke.png' 
] 
 
for img_path in uploaded_image_paths: 
    # Check if the file exists in the current directory (Colab uploads usually put them here) 
    if os.path.exists(img_path): 
        predict_and_visualize_single_image(model, img_path, class_names, device, epilepsy_class_idx) 
    else: 
        print(f"File not found: {img_path}. Please ensure it's uploaded to the Colab environment.") 
# --- 8. Clean up extracted data (optional) --- 
# Uncomment the following lines if you want to remove the extracted dataset 
# after the script finishes. 
# if os.path.exists(extract_path): 
#     print(f"Cleaning up extracted directory: {extract_path}") 
#     shutil.rmtree(extract_path) 
#     print("Cleanup complete.")
