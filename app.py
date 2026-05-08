import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# VGG16
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Collect all data into one dataframe
def create_df(dataset):
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"Dataset path not found: {dataset}")
    
    image_paths, labels = [], []

    for dirpath, dirnames, filenames in os.walk(dataset):
        for filename in filenames:

            image = os.path.join(dirpath, filename)
            image_paths.append(image)
            if dirpath[-3:] == 'all':
                labels.append('all')
            else:
                labels.append('hem')
    
    if not image_paths:
        raise ValueError(f"No images found in {dataset}")
                
    df = pd.DataFrame({'Image Path': image_paths, 
                           'Label': labels}) 
    
    return df


train_dir = os.path.join("C-NMC_Leukemia", "training_data")
df =  create_df(train_dir)

train_df, remaining_df = train_test_split(df, train_size=0.7, shuffle=True, random_state=31, stratify=df['Label'])
valid_df, test_df= train_test_split(remaining_df, train_size=0.5, shuffle=True, random_state=31, stratify=remaining_df['Label'])

print("Number of training samples: %d" % len(train_df.index))
print("Number of test samples: %d" % len(test_df.index))
print("Number of validation samples: %d" % len(valid_df.index))

def show_history_plot(history):

    training_accuracy = history['accuracy']
    epochs = range(1, len(training_accuracy) + 1)

    # Creating subplots for accuracy and loss
    plt.figure(figsize=(15, 5))

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    plt.plot(epochs, history['accuracy'], 'b', label='Training accuracy', marker='o')
    plt.plot(epochs, history['val_accuracy'], 'c', label='Validation accuracy', marker='o')
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Plotting training and validation loss
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    plt.plot(epochs, history['loss'], 'b', label='Training loss', marker='o')
    plt.plot(epochs, history['val_loss'], 'c', label='Validation loss', marker='o')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Improve layout and displaying the plot
    plt.tight_layout()
    plt.show()


def show_conf_matrix(model):
    test_gen.reset()  # Reset the generator to be sure it's at the start of the dataset
    y_pred = model.predict(test_gen, steps=test_gen.n // test_gen.batch_size+1, verbose=0)

    label_dict = test_gen.class_indices
    classes = list(label_dict.keys())

    # Convert predictions to labels
    pred_labels = np.argmax(y_pred, axis=1)  
    y_true = test_gen.classes  
    
    # Generate the confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, pred_labels)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=['ALL', 'HEM'])

    # Plot the confusion matrix 
    cmap = plt.cm.Blues
    cm_display.plot(cmap=cmap, colorbar=False)  
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.figure(figsize=(7, 7)) 
    plt.show()


def extract_features(model, generator, steps):
    """Extract features from VGG16 base model"""
    generator.reset()  # Ensure we start from beginning
    features = []
    labels = []
    
    samples_processed = 0
    for i in range(steps):
        try:
            x_batch, y_batch = next(generator)
            feature_batch = model.predict(x_batch, verbose=0)
            features.append(feature_batch)
            labels.append(y_batch)
            samples_processed += len(x_batch)
        except StopIteration:
            break
    
    if not features:
        raise ValueError("No features extracted! Check generator.")
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    print(f"  → Processed {samples_processed} samples")
    
    # Flatten features
    features = features.reshape(features.shape[0], -1)
    # Convert one-hot labels to class indices
    labels = np.argmax(labels, axis=1)
    
    return features, labels


def train_ml_classifiers(X_train, y_train, X_val, y_val):
    """Train SVM and Random Forest classifiers"""
    classifiers = {}
    
    # SVM Classifier
    print("\nTraining SVM Classifier...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=31)
    svm_model.fit(X_train, y_train)
    classifiers['SVM'] = svm_model
    
    # Random Forest Classifier
    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=31, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    classifiers['Random Forest'] = rf_model
    
    return classifiers


def evaluate_ml_classifiers(classifiers, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evaluate and compare ML classifiers"""
    results = []
    
    for name, model in classifiers.items():
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        test_acc = model.score(X_test, y_test)
        
        results.append({
            'Model': name,
            'Train Accuracy': train_acc,
            'Validation Accuracy': val_acc,
            'Test Accuracy': test_acc
        })
    
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("CLASSIFIER COMPARISON")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    
    return results_df


def show_ml_confusion_matrix(classifiers, X_test, y_test):
    """Show confusion matrices for ML classifiers"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (name, model) in enumerate(classifiers.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'{name} - Confusion Matrix', fontsize=14)
        
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['ALL', 'HEM'])
        ax.set_yticklabels(['ALL', 'HEM'])
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.show()


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap"""
    # Create a model that maps input to activations and output
    grad_model = Model(
        inputs=[model.input],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute gradient of predicted class with respect to feature map
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Gradient of output with respect to output feature map
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Mean intensity of gradient over specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel by importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def visualize_gradcam(image_path, model, last_conv_layer_name='block5_conv3'):
    """Visualize Grad-CAM for a given image"""
    # Load and preprocess image
    img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Get prediction
    preds = model.predict(img_array, verbose=0)
    pred_class = np.argmax(preds[0])
    confidence = preds[0][pred_class]
    
    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_class)
    
    # Rescale heatmap to range 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap to colorize heatmap
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Create superimposed image
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((224, 224))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Superimpose heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img_array[0]
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    
    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(superimposed_img)
    axes[2].set_title(f'Overlay (Pred: {"ALL" if pred_class == 0 else "HEM"}, Conf: {confidence:.2%})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return pred_class, confidence


def log_prediction(image_path, prediction, confidence, model_name, log_file='diagnostic_log.csv'):
    """Log prediction results for medical review"""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    log_path = os.path.join(logs_dir, log_file)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    diagnosis = 'ALL' if prediction == 0 else 'HEM (Healthy)'
    
    log_entry = {
        'Timestamp': timestamp,
        'Image_Path': image_path,
        'Diagnosis': diagnosis,
        'Confidence': f'{confidence:.4f}',
        'Model': model_name
    }
    
    # Append to CSV
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        log_df = pd.DataFrame([log_entry])
    
    log_df.to_csv(log_path, index=False)
    print(f"\n[LOG] Prediction logged to {log_file}")


def predict_sample(image_path, vgg_model, ml_classifiers, feature_extractor):
    """Predict diagnosis for unknown sample using all models"""
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC REPORT: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # Load and preprocess image
    img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # ADD THIS LINE
    
    # VGG16 end-to-end prediction
    vgg_pred = vgg_model.predict(img_array, verbose=0)
    vgg_class = np.argmax(vgg_pred[0])
    vgg_conf = vgg_pred[0][vgg_class]
    
    # Extract features for ML classifiers
    features = feature_extractor.predict(img_array, verbose=0)
    features = features.reshape(1, -1)
    
    results = []
    results.append({
        'Model': 'VGG16 (Deep Learning)',
        'Prediction': 'ALL' if vgg_class == 0 else 'HEM',
        'Confidence': f'{vgg_conf:.4f}'
    })
    
    # ML classifier predictions
    for name, model in ml_classifiers.items():
        ml_pred = model.predict(features)[0]
        ml_prob = model.predict_proba(features)[0]
        ml_conf = ml_prob[ml_pred]
        
        results.append({
            'Model': name,
            'Prediction': 'ALL' if ml_pred == 0 else 'HEM',
            'Confidence': f'{ml_conf:.4f}'
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print(f"{'='*60}\n")
    
    # Log the VGG16 prediction (or choose your preferred model)
    log_prediction(image_path, vgg_class, vgg_conf, 'VGG16')
    
    # Visualize Grad-CAM
    print("Generating Grad-CAM visualization...")
    visualize_gradcam(image_path, vgg_model)
    
    return results_df


hem_img = train_df[train_df['Label'] == 'hem'].sample(3)
all_img = train_df[train_df['Label'] == 'all'].sample(3)
sampled_df = pd.concat([hem_img, all_img])

# Create a figure with subplots to show the images in
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

for i, row in enumerate(sampled_df.iterrows()):
    img = mpimg.imread(row[1]['Image Path'])
    ax = axes[i//3, i%3]
    ax.imshow(img)
    ax.axis('off')    
    if row[1]['Label'] == 'hem':
        ax.set_title(f"Label: hem")
    else:
        ax.set_title(f"Label: all")

plt.show()


batch_size = 40

# Add preprocessing for VGG16
train_data_generator = ImageDataGenerator(
    horizontal_flip=True,
    preprocessing_function=preprocess_input  # Add VGG16 preprocessing
)
valid_data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input  # Add VGG16 preprocessing
)

test_data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input  # Separate generator for test data
)

train_gen = train_data_generator.flow_from_dataframe( train_df, x_col= 'Image Path', y_col= 'Label', target_size= (224, 224), class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)

valid_gen = valid_data_generator.flow_from_dataframe( valid_df, x_col= 'Image Path', y_col= 'Label', target_size= (224, 224), class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)

test_gen = test_data_generator.flow_from_dataframe( test_df, x_col= 'Image Path', y_col= 'Label', target_size= (224, 224), class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= False, batch_size= batch_size)

train_steps = (train_gen.n + batch_size - 1) // batch_size
validation_steps = (valid_gen.n + batch_size - 1) // batch_size

# Instantiate base model FOR FEATURE EXTRACTION
img_shape=(224, 224, 3)
VGG16_base_model = VGG16(weights='imagenet', input_shape=img_shape, include_top=False, pooling='avg')

# This will be our feature extractor
VGG16_base_model.trainable = False

print("\n" + "="*60)
print("EXTRACTING FEATURES FROM VGG16")
print("="*60)

# Extract features for training, validation, and test sets
print("\nExtracting training features...")
train_steps_extract = (len(train_df) + batch_size - 1) // batch_size
X_train_features, y_train_labels = extract_features(VGG16_base_model, train_gen, train_steps_extract)

print("Extracting validation features...")
valid_steps_extract = (len(valid_df) + batch_size - 1) // batch_size
valid_gen.reset()
X_val_features, y_val_labels = extract_features(VGG16_base_model, valid_gen, valid_steps_extract)

print("Extracting test features...")
test_steps_extract = (len(test_df) + batch_size - 1) // batch_size
test_gen.reset()
X_test_features, y_test_labels = extract_features(VGG16_base_model, test_gen, test_steps_extract)

print(f"\nFeature extraction complete!")
print(f"Training features shape: {X_train_features.shape}")
print(f"Validation features shape: {X_val_features.shape}")
print(f"Test features shape: {X_test_features.shape}")

# Train ML Classifiers
print("\n" + "="*60)
print("TRAINING HYBRID ML CLASSIFIERS")
print("="*60)
ml_classifiers = train_ml_classifiers(X_train_features, y_train_labels, X_val_features, y_val_labels)

# Evaluate ML Classifiers
ml_results = evaluate_ml_classifiers(
    ml_classifiers, 
    X_train_features, y_train_labels,
    X_val_features, y_val_labels,
    X_test_features, y_test_labels
)

# Show confusion matrices for ML classifiers
show_ml_confusion_matrix(ml_classifiers, X_test_features, y_test_labels)

# Build VGG16 end-to-end model for comparison and Grad-CAM
print("\n" + "="*60)
print("TRAINING VGG16 END-TO-END MODEL")
print("="*60)

VGG16_base_for_gradcam = VGG16(weights='imagenet', input_shape=img_shape, include_top=False, pooling=None)
VGG16_base_for_gradcam.trainable = False

last_layer = VGG16_base_for_gradcam.get_layer('block5_pool')
last_output = last_layer.output
x = keras.layers.GlobalMaxPooling2D()(last_output)
x = keras.layers.Dropout(0.3)(x)
# x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(2, activation='softmax')(x)

VGG16_model = tf.keras.Model(VGG16_base_for_gradcam.input, x, name="VGG16_model")
VGG16_model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

epochs = 50  # Change from 20 to 50

# Reset generators
train_gen.reset()
valid_gen.reset()

# Add Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history_VGG16 = VGG16_model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=valid_gen,
    validation_steps=validation_steps,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    callbacks=[early_stop]
)

def evaluate_matrix(model):
    """Evaluate VGG16 model on train, validation, and test sets"""
    test_steps = (len(test_df) + batch_size - 1) // batch_size
    train_gen.reset()
    valid_gen.reset()
    test_gen.reset()
    
    train_score = model.evaluate(train_gen, steps=test_steps, verbose=0)
    valid_score = model.evaluate(valid_gen, steps=test_steps, verbose=0)
    test_score = model.evaluate(test_gen, steps=test_steps, verbose=0)

    header = "{:<12} {:<10} {:<10}".format("", "Loss", "Accuracy") 
    separator = '-' * len(header)
    train_row = "{:<12} {:<10.5f} {:<10.5f}".format("Train", train_score[0], train_score[1])
    valid_row = "{:<12} {:<10.5f} {:<10.5f}".format("Validation", valid_score[0], valid_score[1])
    test_row = "{:<12} {:<10.5f} {:<10.5f}".format("Test", test_score[0], test_score[1])

    table = '\n'.join([header, separator, train_row, valid_row, test_row])
    print(table)
    
    return train_score[1], valid_score[1], test_score[1]  # Return accuracies


show_history_plot(history_VGG16.history)

print("\n" + "="*60)
print("VGG16 END-TO-END MODEL EVALUATION")
print("="*60)
vgg16_train_acc, vgg16_val_acc, vgg16_test_acc = evaluate_matrix(VGG16_model)

show_conf_matrix(VGG16_model)

# Demonstrate Grad-CAM visualization on test samples
print("\n" + "="*60)
print("GRAD-CAM EXPLAINABILITY VISUALIZATION")
print("="*60)

# Show Grad-CAM for a few test samples
sample_images = test_df.sample(3)
for idx, row in sample_images.iterrows():
    print(f"\nAnalyzing: {os.path.basename(row['Image Path'])} (True Label: {row['Label']})")
    visualize_gradcam(row['Image Path'], VGG16_model)

# Example: Predict on unknown sample
print("\n" + "="*60)
print("PREDICTION ON UNKNOWN SAMPLE")
print("="*60)
# Get a random test image as demonstration
unknown_sample = test_df.sample(1).iloc[0]['Image Path']
predict_sample(unknown_sample, VGG16_model, ml_classifiers, VGG16_base_model)

print("\n" + "="*60)
print("DIAGNOSTIC SYSTEM READY")
print("="*60)
print("✓ Feature extraction: VGG16")
print("✓ Classifiers trained: SVM, Random Forest")
print("✓ Explainability: Grad-CAM implemented")
print("✓ Logging system: Active")
print(f"✓ Results logged to: logs/diagnostic_log.csv")
print("="*60)


def plot_model_comparison(ml_results, vgg16_test_accuracy):
    """Plot bar chart comparing accuracy of VGG16, SVM, and Random Forest models"""
    
    # Extract test accuracies from ml_results DataFrame
    svm_acc = ml_results[ml_results['Model'] == 'SVM']['Test Accuracy'].values[0] * 100
    rf_acc = ml_results[ml_results['Model'] == 'Random Forest']['Test Accuracy'].values[0] * 100
    vgg_acc = vgg16_test_accuracy * 100
    
    models = ['VGG16\n(Deep Learning)', 'SVM\n(Classical ML)', 'Random Forest\n(Ensemble)']
    accuracies = [vgg_acc, svm_acc, rf_acc]
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: VGG16 vs SVM vs Random Forest', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    legend_text = 'VGG16: Transfer learning\nSVM: RBF kernel\nRandom Forest: 100 trees'
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Model comparison chart saved as: model_comparison.png")
    plt.show()


print("\n💡 Call: plot_model_comparison(ml_results, vgg16_test_acc) to compare all 3 models")