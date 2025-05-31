# I cheated :(, don't know how else I'm going to read all the images with different names
import os
import matplotlib.pyplot as plt

from PIL import Image
from typing import List, Tuple
from utils.torch_lite import Tensor

from utils.random_lite import SimpleRandom
from utils.module import *
from utils.optimizer import SGD
from utils.functional import relu, cross_entropy

rng = SimpleRandom(seed=11)

def read_config_from_txt(path: str) -> dict:
    config = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
            if key in ['batch_size', 'epochs', 'seed']:
                config[key] = int(value)
            elif key in ['learning_rate']:
                config[key] = float(value)
            else:
                config[key] = value
    return config

def get_optimizer(optimizer_type, model, lr):
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == "sgd":
        return SGD(model, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def get_loss_function(loss_name):
    loss_name = loss_name.lower()
    if loss_name == "cross_entropy":
        return cross_entropy
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

def log_metrics(log_file, epoch, train_loss, val_loss, val_accuracy, batch_idx=None):
    """Log training metrics to file"""
    
    with open(log_file, 'a') as f:
        if batch_idx is not None:
            f.write(f"Epoch: {epoch}, Batch: {batch_idx}, Train Loss: {train_loss:.6f}\n")
        else:
            f.write(f"Epoch: {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.2f}%\n")

def plot_training_curves(train_losses, val_losses, val_acc, save_path="training_curves.png"):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot validation accuracy
    ax2.plot(epochs, val_acc, 'g-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved to {save_path}")

def evaluate_model(model, dataloader, batch_size, criteria, num_batches=10):
    """Evaluate model and return average loss and accuracy"""
    total_loss = 0.0
    correct = 0
    total = 0
    
    for val_batch in range(num_batches):
        val_images, val_labels = dataloader.get_batch(batch_size=batch_size, split='val')
        val_logits = model(val_images)
        val_loss = criteria(val_logits, val_labels)
        total_loss += val_loss.data
        
        # Calculate accuracy
        for i, sample_logits in enumerate(val_logits.data):
            pred = 0
            max_val = sample_logits[0]
            for j, val in enumerate(sample_logits):
                if val > max_val:
                    max_val = val
                    pred = j
            
            if pred == val_labels.data[i]:
                correct += 1
            total += 1
    
    avg_loss = total_loss / num_batches
    accuracy = (correct / total) * 100
    return avg_loss, accuracy

class MNISTPNGLoader:
    """MNIST dataloader that reads from organized directories"""
    
    def __init__(self, data_dir: str, seed: int = 11, image_type: str = "png", split: float = 0.9):
        self.data_dir = data_dir
        self.rng = SimpleRandom(seed=seed)
        self.image_type = image_type.lower()
        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []
        self.test_images = []
        self.test_labels = []
        self.split = split
        
    def load_mnist_data(self):
        temp_train_images = []
        temp_train_labels = []
        
        for digit in range(10):
            digit_path = f"{self.data_dir}/train/{digit}"
            count = self._load_images_from_digit_folder(digit_path, digit, temp_train_images, temp_train_labels)
            print(f"Loaded digit {digit}: {count} training images")
        

        self._split_train_val(temp_train_images, temp_train_labels)
        
        for digit in range(10):
            digit_path = f"{self.data_dir}/test/{digit}"
            count = self._load_images_from_digit_folder(digit_path, digit, self.test_images, self.test_labels)
            print(f"Loaded digit {digit}: {count} test images")
        
        print(f"Final split:")
        print(f"+ Training: {len(self.train_images)} samples")
        print(f"+ Validation: {len(self.val_images)} samples") 
        print(f"+ Test: {len(self.test_images)} samples")
        if self.train_images:
            print(f"Image shape: {len(self.train_images[0])}x{len(self.train_images[0][0])}")
    
    def _load_images_from_digit_folder(self, digit_path: str, digit: int, image_list: List, label_list: List):
        all_files = os.listdir(digit_path)

        # For my code, reduced is jpg and png is original
        image_files = [f for f in all_files if f.lower().endswith('.' + self.image_type)]
        
        loaded_count = 0
        for img_filename in image_files:
            img_path = f"{digit_path}/{img_filename}"
            
            img = Image.open(img_path)

            if img.mode != 'L':
                img = img.convert('L')
            
            img_array = list(img.getdata())
            
            # Reshape to 28x28 and normalize to [0,1]
            img_2d = []
            for row in range(28):
                img_row = []
                for col in range(28):
                    pixel_val = img_array[row * 28 + col] / 255.0
                    img_row.append(pixel_val)
                img_2d.append(img_row)
            
            image_list.append(img_2d)
            label_list.append(digit)
            loaded_count += 1
                            
        return loaded_count
    
    def _split_train_val(self, temp_images: List, temp_labels: List):

        combined = list(zip(temp_images, temp_labels))
        
        # Shuffle data for randomness
        for i in range(len(combined)):
            j = rng.rand() % len(combined)
            combined[i], combined[j] = combined[j], combined[i]
        
        split_idx = int(self.split * len(combined))
        
        train_data = combined[:split_idx]
        val_data = combined[split_idx:]
        
        self.train_images = [img for img, label in train_data]
        self.train_labels = [label for img, label in train_data]
        
        self.val_images = [img for img, label in val_data]
        self.val_labels = [label for img, label in val_data]
    
    def get_batch(self, batch_size: int, split: str = 'train') -> Tuple[Tensor, Tensor]:
        """Get a batch of data from specified split"""
        if split == 'train':
            images = self.train_images
            labels = self.train_labels
        elif split == 'val':
            images = self.val_images
            labels = self.val_labels
        elif split == 'test':
            images = self.test_images
            labels = self.test_labels
        
        batch_images = []
        batch_labels = []
        
        for _ in range(batch_size):
            idx = rng.rand() % len(images)
            img = [images[idx]]
            batch_images.append(img)
            batch_labels.append(labels[idx])
        
        return Tensor(batch_images, requires_grad=False), Tensor(batch_labels, requires_grad=False)


class CNN(Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = Convol2D(1, 6, kernel_size=5, padding=2)
        self.pool1 = MaxPooling(2, 2)
        self.conv2 = Convol2D(6, 16, kernel_size=5, padding=0)
        self.pool2 = MaxPooling(2, 2)
        self.conv3 = Convol2D(16, 120, kernel_size=5, padding=0)
        self.flatten = Flatten()
        self.fc1 = Linear(120, 84)
        self.fc2 = Linear(84, 10)
    
    def forward(self, x):
        x = self.conv1(x)      
        x = relu(x)            
        x = self.pool1(x)      
        
        # Second conv block  
        x = self.conv2(x)      
        x = relu(x)          
        x = self.pool2(x)     
        
        x = self.conv3(x)     
        x = relu(x)          
        
        x = self.flatten(x)   
        
        x = self.fc1(x)       
        x = relu(x)           
        x = self.fc2(x) 

        logits = x 
        
        return logits


class CNN_Lite(Module):
    def __init__(self):
        super().__init__()
        
        # 3 conv, 1 maxpool, 1 dense
        self.conv1 = Convol2D(1, 3, kernel_size=3, padding=1)
        self.conv2 = Convol2D(3, 6, kernel_size=3, padding=1)
        self.conv3 = Convol2D(6, 9, kernel_size=3, padding=1)
        self.pool = MaxPooling(2, 2)
        self.flatten = Flatten()
        self.fc = Linear(9 * 14 * 14, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.conv3(x)
        x = relu(x)
        x = self.pool(x)
        x = self.flatten(x)  
        x = self.fc(x)   

        logits = x   

        return logits


def training_pipeline(config):
    data_dir = config["data_dir"]
    seed = config.get("seed", 11)

    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 10)

    lr = config.get("learning_rate", 0.001)
    image_type = config.get("image_type", "png")

    save_dir = config.get("save_dir", "results")
    train_val_split = config.get("train_val_split", 0.9)
    train_val_split = float(train_val_split)
    
    # create if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # list for track
    train_losses = []
    val_losses = []
    val_acc = []
    
    log_file = config.get("log_file", "training_log.txt")
    if os.path.exists(log_file):
        os.remove(log_file)
    
    with open(log_file, 'w') as f:
        f.write("=== Training Log ===\n")
        f.write(f"Configuration: {config}\n")
        f.write("=" * 50 + "\n\n")

    dataloader = MNISTPNGLoader(data_dir, seed=seed, image_type=image_type, split=train_val_split)
    dataloader.load_mnist_data()

    model = CNN_Lite()
    optimizer = SGD(model, lr=lr)

    criteria = get_loss_function(config["loss_function"])

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        train_loss = 0.0
        num_batches = len(dataloader.train_images) // batch_size

        print(f"Number of iteratios per epoch: {num_batches}")

        for batch in range(num_batches):
            images, labels = dataloader.get_batch(batch_size=batch_size, split='train')
            
            # zero grad before each batch
            optimizer.zero_grad()
            
            # forward
            logits = model(images)
            loss = criteria(logits, labels)
            train_loss += loss.data
            
            # backward
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                print(f"Epoch: {epoch+1}, Iter: {batch + 1}/{num_batches}, Loss: {loss.data:.4f}")

            log_metrics(log_file, epoch+1, loss.data, None, None, batch + 1)

        avg_train_loss = train_loss / num_batches
        
        avg_val_loss, val_accuracy = evaluate_model(model, dataloader, 32, criteria, num_batches=10)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_acc.append(val_accuracy)
        
        # Print epoch summary
        print(f"Epoch {epoch+1} Summary:")
        print(f"+ Average Train Loss: {avg_train_loss:.4f}")
        print(f"+ Average Val Loss: {avg_val_loss:.4f}")
        print(f"+ Validation Accuracy: {val_accuracy:.2f}%")
        
        # Log epoch summary
        log_metrics(log_file, epoch+1, avg_train_loss, avg_val_loss, val_accuracy)

    test_correct = 0
    test_total = 0
    test_batches = 10 # 320 samples
    
    for test_batch in range(test_batches):
        test_images, test_labels = dataloader.get_batch(batch_size=32, split='test')
        test_logits = model(test_images)
        
        for i, sample_logits in enumerate(test_logits.data):
            pred = 0
            max_val = sample_logits[0]
            for j, val in enumerate(sample_logits):
                if val > max_val:
                    max_val = val
                    pred = j
            
            if pred == test_labels.data[i]:
                test_correct += 1
            test_total += 1
    
    test_accuracy = (test_correct / test_total) * 100
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    
    # Log final results
    with open(log_file, 'a') as f:
        f.write(f"\n=== Training Complete ===\n")
        f.write(f"Final Test Accuracy: {test_accuracy:.2f}%\n")

    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(train_losses, val_losses, val_acc, save_path=f"{save_dir}/training_curves.png")

    # Save the parameters values to a text file
    print("Training completed!")
    print("Saving model parameters to model_parameters.txt")
    if os.path.exists(f"{save_dir}/model_parameters.txt"):
        os.remove(f"{save_dir}/model_parameters.txt")
    with open(f"{save_dir}/model_parameters.txt", "w") as f:
        f.write(model.parameters())


if __name__ == "__main__":
    config = read_config_from_txt("config.txt")
    print(f"Config: {config}")
    training_pipeline(config)