import gradio as gr
import numpy as np
import cv2
from PIL import Image
import torch
from fastai.vision.all import *
import pathlib
import random
from huggingface_hub import hf_hub_download
import os

# The original code had a line 'pathlib.PosixPath = pathlib.WindowsPath'
# This line is for running models trained on Colab (Linux) on a Windows system.
# However, this environment is Linux, and this line causes a NotImplementedError.
# We are removing this line as it's not needed in a Linux environment and causes issues.
# try:
#     pathlib.PosixPath = pathlib.WindowsPath
# except (AttributeError, NameError):
#     pass

# We download the trained model from Hugging Face Hub.
# This helps overcome potential file size limits in certain environments.
hf_token = os.getenv("HF_TOKEN") # Get the Hugging Face token from environment variables
pkl_path = hf_hub_download(
    repo_id="brknozz/ada447_final_data",  # Repository ID of the dataset
    filename="learn_renset50.pkl",      # Name of the model file
    token=hf_token,                      # Hugging Face token for authentication
    repo_type="dataset"                  # Specify that it's a dataset repository
)

# Constants for image processing
IMG_SIZE = 256 # The size to which images are resized for model input

def normalize_lab(lab_img):
    """
    Normalizes L, A, B channels of a LAB image to a range expected by the model.
    L channel: [0, 100] -> [0, 1]
    A/B channels: [-128, 127] -> [-1, 1]
    """
    l, a, b = cv2.split(lab_img) # Split LAB image into individual channels
    l = l / 100.0                # Normalize L channel
    a = (a - 128.0) / 127.0      # Normalize A channel
    b = (b - 128.0) / 127.0      # Normalize B channel
    return np.stack([l, a, b], axis=2) # Stack channels back into a single image

def denormalize_lab(lab_tensor):
    """
    Denormalizes a LAB tensor (from model output) back to the original LAB range.
    [0, 1] -> [0, 100] for L
    [-1, 1] -> [-128, 127] for A/B
    Clips values to ensure they are within valid 8-bit range [0, 255].
    """
    lab_np = lab_tensor.cpu().numpy() # Convert PyTorch tensor to NumPy array
    l, a, b = lab_np[0], lab_np[1], lab_np[2] # Extract L, A, B channels

    l = l * 100.0                # Denormalize L channel
    a = a * 127.0 + 128.0        # Denormalize A channel
    b = b * 127.0 + 128.0        # Denormalize B channel

    # Stack channels and convert to uint8, clipping to [0, 255]
    lab_restored = np.stack([l, a, b], axis=2).astype(np.uint8)
    return lab_restored

def create_sparse_color_hints(lab_img_normalized, num_points=1310, point_size=1):
    """
    Generates a sparse hint mask from the normalized AB channels of an image.
    This simulates initial color hints that a user might provide.
    Hints are placed at random locations within the image.
    """
    h, w, c = lab_img_normalized.shape
    # Initialize hint mask with NaN values. This is crucial for distinguishing
    # actual hints from "no hint" areas (which will be converted to 0 later).
    hint_mask = np.full((h, w, 2), fill_value=np.nan, dtype=np.float32)
    ab_channels = lab_img_normalized[:, :, 1:] # Extract normalized A and B channels

    for _ in range(num_points):
        # Generate random coordinates for the hint point
        y = random.randint(point_size, h - point_size - 1)
        x = random.randint(point_size, w - point_size - 1)

        # Define the region for the hint based on point_size
        y_start, y_end = max(0, y - point_size), min(h, y + point_size)
        x_start, x_end = max(0, x - point_size), min(w, x + point_size)

        # Apply the actual AB color values from the image to the hint mask
        hint_mask[y_start:y_end, x_start:x_end] = ab_channels[y_start:y_end, x_start:x_end]

    # Replace NaN values with 0.0. This makes the mask dense (no NaNs)
    # but still allows the model to differentiate hinted areas (actual AB values)
    # from unhinted areas (0,0 values).
    hint_mask = np.nan_to_num(hint_mask, nan=0.0)
    return hint_mask

def get_model_input(fn):
    """
    Prepares an image for model input, extracting L channel and sparse AB hints.
    This function seems to be for initial data preparation/testing, not directly
    used in the Gradio `colorize_image` function which handles user drawings.
    """
    original_img = Image.open(fn).convert('RGB')
    original_img_resized = original_img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    original_img_np = np.array(original_img_resized)

    # Convert RGB to LAB color space
    lab_img = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2LAB)

    # Normalize LAB channels
    lab_img_normalized = normalize_lab(lab_img)

    # Extract L channel (grayscale information)
    l_channel = lab_img_normalized[:, :, 0:1] # Keep dimension for concatenation

    # Create sparse color hints from the original image's AB channels
    hint_mask_ab = create_sparse_color_hints(lab_img_normalized)

    # Combine hint_mask_ab (2 channels) and l_channel (1 channel)
    # The model expects input in the format [hint_a, hint_b, L]
    combined = np.concatenate([
        hint_mask_ab, # [H, W, 2] -> hint_a, hint_b
        l_channel     # [H, W, 1] -> L
    ], axis=2)

    # Convert to PyTorch tensor and reorder dimensions to [C, H, W]
    return torch.from_numpy(combined).permute(2, 0, 1).float()

def get_target_image(fn):
    """
    Prepares the target image's AB channels for model training.
    Similar to `get_model_input`, this is likely for training setup,
    not directly used in the Gradio inference.
    """
    original_img = Image.open(fn).convert('RGB')
    original_img_resized = original_img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    original_img_np = np.array(original_img_resized)

    lab_img = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2LAB)
    lab_img_normalized = normalize_lab(lab_img)

    # Target is only the a and b channels
    ab_channels = lab_img_normalized[:, :, 1:]

    # Convert to PyTorch tensor and reorder dimensions to [C, H, W]
    return torch.from_numpy(ab_channels).permute(2, 0, 1).float()

# --- START OF FIX ---
# Dummy calls to ensure these functions are 'registered' in the current namespace
# for FastAI's pickling mechanism, just before loading the learner.
# This is a common workaround for "Can't get attribute" errors when loading FastAI models.
try:
    # A dummy Path object is used, as the function expects a path.
    # A FileNotFoundError is expected here and is handled.
    _ = get_model_input(Path('dummy_image_for_fastai_loading.png'))
    _ = get_target_image(Path('dummy_image_for_fastai_loading.png'))
except FileNotFoundError:
    # This error is expected because the dummy file doesn't exist.
    # The purpose is to ensure the functions are in the global namespace.
    pass
except Exception as e:
    # Catch any other unexpected errors during the dummy calls.
    print(f"Warning during dummy function call for FastAI model loading: {e}")
# --- END OF FIX ---

# Load the trained FastAI model
MODEL_PATH = Path(pkl_path)
learn = load_learner(MODEL_PATH)

def colorize_image(editor_data):
    """
    Main function to colorize a grayscale image based on user-provided hints.
    Takes `editor_data` from Gradio's ImageEditor, which includes the background
    image (grayscale) and any drawing layers (color hints).
    """
    # 1. Get inputs and prepare
    if editor_data['background'] is None:
        # If no image is uploaded, return None
        return None

    background_np = editor_data['background'] # The background image (grayscale input)
    layers_list = editor_data['layers']       # List of drawing layers (user's color hints)

    original_h, original_w = background_np.shape[0], background_np.shape[1]

    # Convert background NumPy array to PIL Image for resizing, then back to RGB NumPy
    bw_image_pil = Image.fromarray(background_np).convert('RGB')

    # 2. Convert original image to L channel
    # Resize the input image to the model's expected size (IMG_SIZE)
    img_resized_np = np.array(bw_image_pil.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS))
    lab_original = cv2.cvtColor(img_resized_np, cv2.COLOR_RGB2LAB)
    l_channel = lab_original[:, :, 0] # Extract the L channel

    # Normalize L channel (Model expects L in [0, 1])
    l_norm = l_channel / 100.0

    # 3. Process user's drawing (hints)
    # Initialize hint mask with zeros. Areas without hints will remain (0,0).
    hint_mask_ab = np.zeros((IMG_SIZE, IMG_SIZE, 2), dtype=np.float32)

    # If the user has made a drawing (layers_list is not empty)
    if layers_list:
        # The first (and usually only) layer contains the user's drawing.
        user_drawing_np = layers_list[0]
        user_drawing_pil = Image.fromarray(user_drawing_np).convert('RGBA')

        # Resize the drawing to model input size. Using NEAREST for sharp pixels.
        user_drawing_resized = user_drawing_pil.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)
        drawing_np = np.array(user_drawing_resized)

        # Identify pixels where drawing occurred (alpha channel > 0)
        drawn_pixels_mask = drawing_np[:, :, 3] > 0

        if np.any(drawn_pixels_mask): # Check if any pixels were actually drawn on
            # Get the RGB values of the drawn pixels
            drawn_rgb = drawing_np[drawn_pixels_mask, :3]
            # Convert these RGB hints to LAB. Reshape for cv2.cvtColor to work on multiple pixels.
            lab_hints = cv2.cvtColor(drawn_rgb[np.newaxis, :, :], cv2.COLOR_RGB2LAB)[0]

            # Normalize the A and B channels of the hints
            a_hints = (lab_hints[:, 1] - 128.0) / 127.0
            b_hints = (lab_hints[:, 2] - 128.0) / 127.0

            # Place the normalized A and B hints into the `hint_mask_ab`
            hint_mask_ab[drawn_pixels_mask, 0] = a_hints
            hint_mask_ab[drawn_pixels_mask, 1] = b_hints

    # 4. Create the input for the model
    # Stack the hint_a, hint_b, and normalized L channels.
    # The model expects input in the format [hint_a, hint_b, L]
    model_input_np = np.stack([hint_mask_ab[:, :, 0], hint_mask_ab[:, :, 1], l_norm], axis=2)
    # Convert to PyTorch tensor, permute dimensions to [C, H, W], add a batch dimension [1, C, H, W]
    model_input_tensor = torch.from_numpy(model_input_np).permute(2, 0, 1).float().unsqueeze(0)

    # 5. Make a prediction with the model
    with torch.no_grad(): # Disable gradient calculation for inference
        # Move the input tensor to the same device as the model (CPU or GPU)
        device = next(learn.model.parameters()).device
        model_input_tensor = model_input_tensor.to(device)

        # Get the predicted A and B channels from the model
        pred_ab = learn.model(model_input_tensor)[0] # [0] to remove batch dimension

    # 6. Convert the output to the final colorized image
    pred_ab_cpu = pred_ab.cpu().numpy() # Move prediction back to CPU and convert to NumPy

    # Reshape L channel to [H, W, 1] for concatenation
    l_channel_orig = l_channel[:, :, np.newaxis]

    # Denormalize the predicted A and B channels back to their original LAB range
    a_pred = pred_ab_cpu[0] * 127.0 + 128.0
    b_pred = pred_ab_cpu[1] * 127.0 + 128.0

    # Reshape predicted A and B channels to [H, W, 1]
    a_pred = a_pred[:, :, np.newaxis]
    b_pred = b_pred[:, :, np.newaxis]

    # Concatenate the original L channel with the predicted and denormalized A and B channels
    final_lab = np.concatenate([l_channel_orig, a_pred, b_pred], axis=2)
    # Clip values to ensure they are within valid uint8 range [0, 255] and convert type
    final_lab = np.clip(final_lab, 0, 255).astype(np.uint8)

    # Convert the final LAB image to RGB color space
    final_rgb_256 = cv2.cvtColor(final_lab, cv2.COLOR_LAB2RGB)
    # Resize the final RGB image back to the original input image's dimensions
    final_rgb_original_size = cv2.resize(final_rgb_256, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)

    return final_rgb_original_size


# Define the Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Interactive Image Colorization with U-Net")
    gr.Markdown("Upload a black-and-white image, use the color brush to add hints, and click 'Colorize!'")

    with gr.Row():
        # Gradio ImageEditor allows uploading an image and drawing on it.
        # type="numpy" ensures the image and layers are passed as NumPy arrays.
        image_input = gr.ImageEditor(
            label="🖌️ Upload & Draw on Image",
            type="numpy",
            height=512,
        )
        # Output component for the colorized image
        image_output = gr.Image(type="numpy", label="Colorized Result", height=512)

    colorize_button = gr.Button("Colorize!")

    # Define the click event for the colorize button
    colorize_button.click(
        fn=colorize_image, # The Python function to call
        inputs=image_input,  # Input from the ImageEditor
        outputs=image_output # Output to the Image component
    )

# Launch the Gradio interface
demo.launch(debug=True)
