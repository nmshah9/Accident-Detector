"""
Model utilities for the Accident Detection Streamlit app.

Wraps the trained MobileNetV2 transfer-learning model (Model 2 from the
end-to-end notebook) with:
- image preprocessing matching training (128x128, MobileNetV2 preprocess_input)
- prediction with confidence
- Grad-CAM heatmap generation to visualize which region of the frame
  drove the model's decision
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (128, 128)
CLASS_NAMES = ["Accident", "Non Accident"]  # alphabetical, matches training label_mode="binary" indexing
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
WEIGHTS_PATH = os.path.join(_PROJECT_ROOT, "models", "accident_mobilenetv2.weights.h5")


def _build_transfer_model():
    """Rebuild the exact architecture used in training (Model 2 from the notebook):
    MobileNetV2 (ImageNet, frozen) -> GAP -> Dense(128) -> Dropout(0.4) -> Dense(1, sigmoid).

    Building the architecture from code (rather than deserializing a full saved model)
    avoids Keras-version-specific config compatibility issues entirely — only the raw
    weight arrays are loaded, which is stable across a much wider range of TF/Keras
    versions than full-model deserialization."""
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights=None)
    base.trainable = False
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name="mobilenetv2_transfer")


def load_model(weights_path: str = WEIGHTS_PATH):
    """Build the architecture fresh and load only the trained weights.
    More robust across TF/Keras versions than loading a full .keras file."""
    model = _build_transfer_model()
    model.load_weights(weights_path)
    return model


def preprocess_image(pil_image):
    """Resize + preprocess a PIL image exactly as done at training time.
    Returns a (1, 128, 128, 3) float32 array ready for model.predict."""
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)  # scales to [-1, 1], matches MobileNetV2 training
    return np.expand_dims(arr, axis=0)


def predict(model, pil_image):
    """Run inference. Returns (predicted_label, confidence_pct, raw_sigmoid_output)."""
    x = preprocess_image(pil_image)
    raw = float(model.predict(x, verbose=0)[0][0])
    # sigmoid output: label index 1 = "Non Accident" (alphabetical order from training),
    # so raw close to 0 -> Accident, raw close to 1 -> Non Accident
    if raw > 0.5:
        label = "Non Accident"
        confidence = raw
    else:
        label = "Accident"
        confidence = 1 - raw
    return label, confidence * 100, raw


def _find_base_and_last_conv_layer(model):
    """Locate the nested MobileNetV2 base model and its last conv layer name.
    The training script builds: Input -> base(MobileNetV2) -> GAP -> Dense -> Dropout -> Dense.
    The base is stored as a single nested layer inside the outer functional model."""
    base = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) or "mobilenet" in layer.name.lower():
            base = layer
            break
    if base is None:
        raise ValueError("Could not locate nested MobileNetV2 base layer for Grad-CAM.")

    # MobileNetV2's last feature map before pooling is typically named 'out_relu'
    last_conv_name = None
    for candidate in ["out_relu", "Conv_1_relu", "block_16_project_BN"]:
        try:
            base.get_layer(candidate)
            last_conv_name = candidate
            break
        except ValueError:
            continue
    if last_conv_name is None:
        # fall back to the last 4D-output layer in the base model
        for layer in reversed(base.layers):
            if len(layer.output.shape) == 4:
                last_conv_name = layer.name
                break
    return base, last_conv_name


def make_gradcam_heatmap(model, pil_image):
    """Generate a Grad-CAM heatmap (as a 2D numpy array, values in [0,1]) showing
    which spatial region of the frame most influenced the model's prediction.

    Handles the nested-submodel architecture (Input -> MobileNetV2 -> head) by
    building a small grad-model directly from the base model's own input/output,
    then manually replaying the classification head on top under the same tape.
    """
    base, last_conv_name = _find_base_and_last_conv_layer(model)

    x = preprocess_image(pil_image)

    # Grad-model: base's own input -> (last conv activations, base output)
    grad_model = keras.Model(
        inputs=base.input,
        outputs=[base.get_layer(last_conv_name).output, base.output],
    )

    # Identify the head layers that come after the base in the outer model
    head_layers = []
    started = False
    for layer in model.layers:
        if layer is base:
            started = True
            continue
        if started:
            head_layers.append(layer)

    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_model(x)
        tape.watch(conv_outputs)
        h = base_output
        for layer in head_layers:
            h = layer(h)
        predictions = h
        # Watch the "Accident" direction: raw output is P(Non Accident),
        # so we backprop through (1 - output) to explain the Accident class,
        # or through output directly to explain whichever class was predicted.
        target = predictions[:, 0]

    grads = tape.gradient(target, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(pil_image, heatmap, alpha=0.45):
    """Resize heatmap to image size and overlay as a red-hot colormap."""
    import matplotlib.cm as cm

    img = pil_image.convert("RGB").resize(IMG_SIZE)
    img_arr = np.array(img).astype(np.float32) / 255.0

    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE).numpy().squeeze()
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[..., :3]

    overlaid = heatmap_colored * alpha + img_arr * (1 - alpha)
    overlaid = np.clip(overlaid, 0, 1)
    return (overlaid * 255).astype(np.uint8)
