# I cheated :(, don't know how else I'm going to read all the images with different names
import os

from PIL import Image
from typing import List, Tuple
from torch_lite import Tensor

from random_lite import SimpleRandom
from module import *
from optimizer import SGD
from functional import relu, cross_entropy

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


class MNISTPNGLoader:
    """MNIST dataloader that reads from organized directories"""
    
    def __init__(self, data_dir: str, seed: int = 11, image_type: str = "png"):
        self.data_dir = data_dir
        self.rng = SimpleRandom(seed=seed)
        self.image_type = image_type.lower()
        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []
        self.test_images = []
        self.test_labels = []
        
    def load_mnist_data(self):
        temp_train_images = []
        temp_train_labels = []
        
        for digit in range(10):
            digit_path = f"{self.data_dir}/train/{digit}"
            count = self._load_images_from_digit_folder(digit_path, digit, temp_train_images, temp_train_labels)
            print(f"Loaded digit {digit}: {count} training images")
        
        # Split 90% train, 10% validation
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
        
        # Split: 90% train, 10% validation
        split_idx = int(0.9 * len(combined))
        
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

    dataloader = MNISTPNGLoader(data_dir, seed=seed, image_type=image_type)
    dataloader.load_mnist_data()

    model = CNN_Lite()
    optimizer = SGD(model, lr=lr)

    criteria = get_loss_function(config["loss_function"])

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        
        # Training
        train_loss = 0.0
        num_batches = len(dataloader.train_images) // batch_size

        print(f"Number of iteration per epoch: {num_batches}")

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

            if (batch) % 10 == 0:
                print(f"Epoch: {epoch+1}, Iter: {batch + 1}/{num_batches}, Loss: {loss.data:.4f}")

        avg_loss = train_loss / num_batches
        print(f"Average Loss: {avg_loss:.4f}")

        val_correct = 0
        val_total = 0
        val_batches = 10 
        
        for val_batch in range(val_batches):
            val_images, val_labels = dataloader.get_batch(batch_size=32, split='val')
            val_logits = model(val_images)
            
            for i, sample_logits in enumerate(val_logits.data):
                pred = 0
                max_val = sample_logits[0]
                for j, val in enumerate(sample_logits):
                    if val > max_val:
                        max_val = val
                        pred = j
                
                if pred == val_labels.data[i]:
                    val_correct += 1
                val_total += 1
        
        val_accuracy = (val_correct / val_total) * 100
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Save the parameters values to a text file
    print("Training completed!")

    print("Saving model parameters to model_parameters.txt")
    if os.path.exists("model_parameters.txt"):
        os.remove("model_parameters.txt")
    with open("model_parameters.txt", "w") as f:
        f.write(model.parameters())


# def training_pipeline_overfit():
#     data_dir = "reduced_mnist_png"
#     dataloader = MNISTPNGLoader(data_dir)
#     dataloader.load_mnist_data()

#     model = CNN_Lite()

#     lr = 0.01

#     optimizer = SGD(model, lr=lr)
#     images, labels = dataloader.get_batch(batch_size=32, split='train')


#     for epoch in range(50):

#         print(model.fc.biases.grad)

#         optimizer.zero_grad()
#         logits = model(images)
#         loss = cross_entropy(logits, labels)
#         loss.backward()
#         optimizer.step()

#         preds = []
#         for sample_logits in logits.data:
#             pred = 0
#             max_val = sample_logits[0]
#             for j, val in enumerate(sample_logits):
#                 if val > max_val:
#                     max_val = val
#                     pred = j
#             preds.append(pred)
#         correct = sum([p == l for p, l in zip(preds, labels.data)])
#         acc = correct / len(labels.data) * 100

#         print(f"Epoch {epoch+1}: Loss={loss.data:.4f}, Accuracy={acc:.2f}%")

#         # Print loss for each iteration (since each epoch is one iter here)
#         print(f"Iter {epoch+1}: Loss={loss.data:.4f}")


if __name__ == "__main__":
    config = read_config_from_txt("config.txt")
    print(f"Config: {config}")
    training_pipeline(config)
