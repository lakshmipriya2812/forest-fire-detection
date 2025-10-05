"""
Forest Fire Detection - Training Script
Simple and easy to use!
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import argparse

# ============================================
# MODEL CREATION
# ============================================

def create_simple_cnn():
    """Simple CNN Model for Beginners"""
    print("Creating Simple CNN model...")
    
    model = models.Sequential([
        # Layer 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Layer 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Layer 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Layer 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


def create_transfer_learning():
    """Transfer Learning Model (Better Accuracy)"""
    print("Creating Transfer Learning model...")
    
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


# ============================================
# DATA PREPARATION
# ============================================

def prepare_data(train_dir, val_dir, batch_size=32):
    """Load and prepare training data"""
    print(f"\nLoading data from:")
    print(f"  Train: {train_dir}")
    print(f"  Validation: {val_dir}")
    
    # Training data with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation data (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    # Load validation data
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    print(f"\n✓ Training samples: {train_generator.samples}")
    print(f"✓ Validation samples: {val_generator.samples}")
    print(f"✓ Classes: {train_generator.class_indices}")
    
    return train_generator, val_generator


# ============================================
# TRAINING
# ============================================

def train_model(model, train_gen, val_gen, epochs, model_name):
    """Train the model"""
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create models folder if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f'models/{model_name}_best.h5',
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "="*60)
    print("🔥 STARTING TRAINING")
    print("="*60 + "\n")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history


# ============================================
# VISUALIZATION
# ============================================

def plot_history(history, model_name):
    """Plot training results"""
    
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'models/{model_name}_history.png')
    print(f"\n✓ Training plot saved: models/{model_name}_history.png")


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='transfer', choices=['cnn', 'transfer'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--val_dir', type=str, default='data/validation')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🔥 FOREST FIRE DETECTION - TRAINING")
    print("="*60)
    print(f"Model: {args.model.upper()}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print("="*60)
    
    # Check if data exists
    if not os.path.exists(args.train_dir):
        print(f"\n❌ ERROR: Directory '{args.train_dir}' not found!")
        print("\nPlease create this structure:")
        print("data/")
        print("├── train/")
        print("│   ├── fire/")
        print("│   └── no_fire/")
        print("└── validation/")
        print("    ├── fire/")
        print("    └── no_fire/")
        return
    
    # Create model
    if args.model == 'cnn':
        model = create_simple_cnn()
        model_name = 'simple_cnn'
    else:
        model = create_transfer_learning()
        model_name = 'transfer_learning'
    
    print(f"\n✓ Model created!")
    print(f"✓ Total parameters: {model.count_params():,}")
    
    # Prepare data
    train_gen, val_gen = prepare_data(args.train_dir, args.val_dir, args.batch_size)
    
    # Train
    history = train_model(model, train_gen, val_gen, args.epochs, model_name)
    
    # Plot results
    plot_history(history, model_name)
    
    # Final results
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    print(f"\n💾 Model saved: models/{model_name}_best.h5")
    print("\n🎯 Ready to detect fires!")
    print(f"Run: python src/detect.py --model models/{model_name}_best.h5 --image test.jpg")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()