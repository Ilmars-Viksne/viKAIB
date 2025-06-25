# Python script for segmenting a specific cell in a microscopy image using micro_sam

# --- Environment Setup Assumptions ---
# 1. Anaconda/Miniconda is installed.
# 2. A Conda environment has been created and activated.
# 3. micro_sam is installed in this environment via:
#    conda install -c conda-forge micro_sam
# 4. An NVIDIA GPU with CUDA 12.4 (or compatible) is available and configured for PyTorch.
#    PyTorch should be installed with CUDA support (e.g., from pytorch.org).
# 5. Available GPU VRAM is 4GB. This might be a limitation. Using smaller models
#    like 'vit_b' (ViT-B) is recommended. Larger models or batch processing might
#    lead to out-of-memory errors.

# --- Core Library Imports ---
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import torch # PyTorch is a dependency of micro_sam and useful for checking CUDA

# --- MicroSAM Imports ---
# For loading a pre-trained model and making predictions
from micro_sam.util import get_sam_predictor
# For visualization (optional, can also use matplotlib directly)
from micro_sam.visualization import show_mask, show_points, show_box

# Further micro_sam imports might be needed depending on the specific functions used.
# For instance, if using automated segmentation pipelines:
# from micro_sam.automated_prediction importマスク_and_track_segmentation
# However, for this task, we focus on interactive-like prediction with prompts.

def segment_cell_with_prompt(image_path, model_type="vit_b", prompt_type="box", prompt_coords=None):
    """
    Loads a microscopy image, uses a pre-trained micro_sam model to segment a cell
    based on a given prompt, and visualizes the result.

    Args:
        image_path (str): Path to the 2D microscopy image.
        model_type (str): Type of SAM model to use (e.g., "vit_b", "vit_l", "vit_h").
                          "vit_b" is recommended for 4GB VRAM.
        prompt_type (str): Type of prompt, either "box" or "points".
        prompt_coords: Coordinates for the prompt.
                       For "box": [xmin, ymin, xmax, ymax]
                       For "points": A tuple (points_coords, points_labels)
                                     points_coords: [[x1, y1], [x2, y2], ...]
                                     points_labels: [1, 0, ...] (1 for positive, 0 for negative)
    """
    print(f"--- Initializing Cell Segmentation ---")
    print(f"Image path: {image_path}")
    print(f"Model type: {model_type}")
    print(f"Prompt type: {prompt_type}")

    # --- VRAM and Device Check ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Available GPU VRAM: {vram_gb:.2f} GB")
        if vram_gb < 4 and model_type != "vit_b":
            print("Warning: Low VRAM detected. Consider using 'vit_b' model to avoid errors.")
        if model_type != "vit_b" and vram_gb < 6: # Larger models might need more
             print(f"Warning: Model '{model_type}' might be too large for {vram_gb:.2f} GB VRAM. If you encounter OOM errors, switch to 'vit_b'.")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU. This will be significantly slower.")
        if model_type != "vit_b":
            print("Warning: Running larger models on CPU can be very slow.")

    # --- 1. Load Image ---
    try:
        image = skimage.io.imread(image_path)
        print(f"Image loaded successfully. Shape: {image.shape}, dtype: {image.dtype}")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Ensure image is 2D (e.g., grayscale). If RGB, convert or select a channel.
    if image.ndim == 3 and image.shape[-1] in [3, 4]:  # Assuming trailing channel for RGB/RGBA
        print("Input image is RGB/RGBA. Converting to grayscale for segmentation.")
        # Convert to grayscale using skimage's rgb2gray or simple mean
        # image = skimage.color.rgb2gray(image)
        # Or, if it's a multi-channel microscopy image not strictly RGB, pick a channel
        image = image[..., 0] # Example: using the first channel
        print(f"Converted to grayscale. New shape: {image.shape}")

    if image.ndim != 2:
        print(f"Error: Image must be 2D (grayscale). Current dimensions: {image.ndim}")
        return

    # --- 2. Initialize MicroSAM Predictor ---
    # This will download the model checkpoint if not already cached.
    # The model will be automatically moved to the 'device' (GPU if available).
    print(f"Initializing SAM predictor with model type '{model_type}'...")
    try:
        predictor = get_sam_predictor(model_type=model_type, device=device)
        print("Predictor initialized.")
    except Exception as e:
        print(f"Error initializing SAM predictor: {e}")
        print("This might be due to an incorrect model_type or network issues if downloading the model.")
        return

    # --- 3. Set Image for Predictor ---
    # The image needs to be "set" or "embedded" by the predictor first.
    # This computes the image embeddings, which is a time-consuming step.
    print("Setting image in the predictor (computing embeddings)...")
    try:
        # micro_sam expects the image to be in a certain format (e.g. RGB for standard SAM)
        # For single-channel microscopy, we might need to stack it if the model expects 3 channels.
        # The `get_sam_predictor` and its underlying SAM `set_image` handle this.
        # If image is uint16, SAM might convert it to uint8.
        if image.dtype == np.uint16:
            # Normalize to 0-255 and convert to uint8 if necessary, or let SAM handle it.
            # SAM's `set_image` preprocesses; often converting to RGB uint8.
            # For grayscale, it's typically duplicated across 3 channels.
             print(f"Image dtype is {image.dtype}. SAM will handle conversion.")

        predictor.set_image(image) # Image should be HxW or HxWxC (e.g. RGB)
        print("Image embeddings computed and set in predictor.")
    except Exception as e:
        print(f"Error setting image in predictor: {e}")
        # This can happen if the image format is unexpected or due to OOM on GPU for large images.
        if "CUDA out of memory" in str(e) and device.type == 'cuda':
            print("CUDA out of memory error during set_image. Try a smaller image or a smaller model if not already using 'vit_b'.")
        return

    # --- 4. Define Prompt ---
    # The prompt_coords need to be in (x, y) format for points,
    # and (xmin, ymin, xmax, ymax) for boxes.
    # Matplotlib displays images with (0,0) at top-left, and coordinates are (col, row) or (x,y)
    # SAM expects points as (x,y) and boxes as (xmin, ymin, xmax, ymax)

    input_box = None
    input_points = None
    input_labels = None

    if prompt_type == "box":
        if prompt_coords is None or len(prompt_coords) != 4:
            print("Error: For 'box' prompt, prompt_coords must be [xmin, ymin, xmax, ymax].")
            # Example box: [100, 100, 200, 200] (xmin, ymin, xmax, ymax)
            # Define a default box if none is provided for demonstration
            h, w = image.shape[:2]
            default_box_size = min(h, w) // 4
            cx, cy = w // 2, h // 2
            input_box = np.array([
                cx - default_box_size, cy - default_box_size,
                cx + default_box_size, cy + default_box_size
            ])
            print(f"Using default bounding box: {input_box}")
        else:
            input_box = np.array(prompt_coords)
        print(f"Using bounding box prompt: {input_box}")

    elif prompt_type == "points":
        if prompt_coords is None or not isinstance(prompt_coords, tuple) or len(prompt_coords) != 2:
            print("Error: For 'points' prompt, prompt_coords must be a tuple: (points_list, labels_list).")
            # Example points: ([[150, 150], [160, 140]], [1, 0]) (points, labels)
            # Define default points if none are provided for demonstration
            h, w = image.shape[:2]
            input_points = np.array([[w // 2, h // 2], [w // 2 + 10, h // 2 + 5]]) # Two example points
            input_labels = np.array([1, 1]) # Both positive
            print(f"Using default point prompts: Coords={input_points.tolist()}, Labels={input_labels.tolist()}")
        else:
            input_points = np.array(prompt_coords[0])
            input_labels = np.array(prompt_coords[1])
            if input_points.ndim != 2 or input_points.shape[1] != 2:
                print("Error: Points coordinates must be a list of [x, y] pairs.")
                return
            if len(input_points) != len(input_labels):
                print("Error: Number of points and labels must match.")
                return
        print(f"Using point prompts: Coords={input_points.tolist()}, Labels={input_labels.tolist()}")
    else:
        print(f"Error: Unknown prompt_type '{prompt_type}'. Choose 'box' or 'points'.")
        return

    # --- 5. Perform Segmentation ---
    print("Performing segmentation...")
    try:
        # multimask_output=True returns multiple plausible masks if found
        # For a single specific cell, we often want the "best" one (often masks[0])
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box,
            multimask_output=True, # Get multiple masks; usually 3 for SAM
        )
        # masks is a batch of masks, typically (N, H, W) where N is number of masks
        # scores are the IOU predictions for each mask
        print(f"Segmentation successful. Received {masks.shape[0]} masks with scores: {scores}")

        # Select the mask with the highest predicted IoU score
        best_mask_idx = np.argmax(scores)
        final_mask = masks[best_mask_idx] # This is a boolean HxW array
        print(f"Selected mask with highest score (IoU: {scores[best_mask_idx]:.4f})")

    except Exception as e:
        print(f"Error during segmentation prediction: {e}")
        if "CUDA out of memory" in str(e) and device.type == 'cuda':
            print("CUDA out of memory error during predict. The model might be too complex for the given VRAM even after setting the image, or the prompt is too complex.")
        return

    # --- 6. Visualize Results ---
    print("Visualizing results...")
    plt.figure(figsize=(15, 5))

    # Plot 1: Original Image with Prompt
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    if prompt_type == "box" and input_box is not None:
        show_box(input_box, plt.gca())
    elif prompt_type == "points" and input_points is not None:
        show_points(input_points, input_labels, plt.gca(), marker_size=100)
    plt.title("Original Image with Prompt")
    plt.axis('off')

    # Plot 2: Segmentation Mask
    plt.subplot(1, 3, 2)
    plt.imshow(final_mask, cmap='viridis') # Use a distinct colormap for the mask
    plt.title("Segmentation Mask")
    plt.axis('off')

    # Plot 3: Image Overlaid with Mask
    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap='gray')
    show_mask(final_mask, plt.gca(), random_color=False, color=np.array([0,1,0,0.5])) # Greenish overlay
    if prompt_type == "box" and input_box is not None: # Show prompt again for context
        show_box(input_box, plt.gca(), color='blue')
    elif prompt_type == "points" and input_points is not None:
        show_points(input_points, input_labels, plt.gca(), marker_size=50, edge_color='blue')
    plt.title("Image with Segmentation Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("Visualization complete. If the plot window doesn't appear, ensure matplotlib backend is configured correctly.")

    # --- 7. Save the Mask (Optional) ---
    # Example: Save the mask as a TIFF or PNG file
    mask_filename_png = "segmentation_mask.png"
    # Convert boolean mask to uint8 (0 or 255) for image saving
    mask_to_save = (final_mask * 255).astype(np.uint8)
    try:
        skimage.io.imsave(mask_filename_png, mask_to_save, check_contrast=False)
        print(f"Segmentation mask saved to {mask_filename_png}")
    except Exception as e:
        print(f"Error saving mask: {e}")

    # Example: Save as numpy array
    # mask_filename_npy = "segmentation_mask.npy"
    # np.save(mask_filename_npy, final_mask)
    # print(f"Segmentation mask saved to {mask_filename_npy}")

    print("--- Segmentation Process Finished ---")


# --- Example Usage ---
if __name__ == "__main__":
    # --- IMPORTANT: Define your image path and prompt here ---

    # Option 1: Create a dummy image for testing if you don't have one readily available
    def create_dummy_image(path="dummy_microscopy_image.tif", size=(256, 256)):
        img = np.zeros(size, dtype=np.uint8)
        # Create a simple circular "cell"
        center_x, center_y = size[1] // 2, size[0] // 2
        radius = size[0] // 4
        y, x = np.ogrid[:size[0], :size[1]]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        img[dist_from_center <= radius] = 150
        # Add some noise
        img = img + np.random.normal(0, 20, img.shape).astype(np.uint8)
        img = np.clip(img, 0, 255)
        skimage.io.imsave(path, img, check_contrast=False)
        print(f"Created dummy image at {path}")
        return path

    # IMAGE_PATH = "path/to/your/microscopy_image.tif" # <-- REPLACE THIS
    IMAGE_PATH = create_dummy_image() # Using a dummy image for this example

    # --- Prompt Configuration ---

    # === Option A: Bounding Box Prompt ===
    # Define the bounding box [xmin, ymin, xmax, ymax]
    # These coordinates are relative to the image dimensions.
    # For the dummy image (256x256) with a cell in the center:
    img_h, img_w = skimage.io.imread(IMAGE_PATH).shape[:2]
    box_size = img_h // 3
    center_x, center_y = img_w // 2, img_h // 2
    # A box slightly larger than the cell for the dummy image
    # Example: xmin=80, ymin=80, xmax=180, ymax=180 for a 256x256 image
    BOX_PROMPT_COORDS = [
        center_x - box_size // 2 - 10, center_y - box_size // 2 - 10, # xmin, ymin
        center_x + box_size // 2 + 10, center_y + box_size // 2 + 10  # xmax, ymax
    ]
    # To use box prompt:
    # segment_cell_with_prompt(IMAGE_PATH, model_type="vit_b", prompt_type="box", prompt_coords=BOX_PROMPT_COORDS)


    # === Option B: Point Prompts ===
    # Define points as ([[x1, y1], [x2, y2], ...], [label1, label2, ...])
    # Label 1 for foreground (part of the cell), 0 for background.
    # For the dummy image (256x256) with a cell in the center:
    # Example: A point in the center of the cell (positive)
    #          and a point outside (negative, optional but can improve results)
    POINT_PROMPT_COORDS = (
        [[center_x, center_y], [10, 10]],  # Coordinates: [[x_fg, y_fg], [x_bg, y_bg]]
        [1, 0]                             # Labels: [positive, negative]
    )
    # To use point prompts:
    # segment_cell_with_prompt(IMAGE_PATH, model_type="vit_b", prompt_type="points", prompt_coords=POINT_PROMPT_COORDS)

    # --- Run Segmentation (select one of the prompt types above) ---
    print("Running segmentation with Bounding Box prompt by default for this example.")
    segment_cell_with_prompt(IMAGE_PATH,
                             model_type="vit_b", # "vit_b" is recommended for 4GB VRAM
                             prompt_type="box",
                             prompt_coords=BOX_PROMPT_COORDS)

    # print("\nRunning segmentation with Point prompts for demonstration.")
    # segment_cell_with_prompt(IMAGE_PATH,
    #                          model_type="vit_b",
    #                          prompt_type="points",
    #                          prompt_coords=POINT_PROMPT_COORDS)

    # --- Considerations ---
    # - If your microscopy images are multi-channel (e.g., DAPI, FITC, TRITC),
    #   you'll need to decide which channel to use for segmentation or how to combine them
    #   before passing to micro_sam. The current script converts RGB to grayscale or uses the first channel.
    # - For very large images (e.g., whole slide images), you might need to tile the image,
    #   perform segmentation on tiles, and then stitch the results, which is more complex.
    # - Experiment with different prompt types and coordinates for best results on your specific data.
    # - Model choice ('vit_b', 'vit_l', 'vit_h') affects performance and VRAM usage.
    #   'vit_b' is the smallest, 'vit_h' is the largest and most accurate but requires significant VRAM.
    # - If micro_sam has been fine-tuned on similar microscopy data, using a fine-tuned model
    #   checkpoint (if available and you know how to load it with `get_sam_predictor`) might yield better results.
    #   The default `get_sam_predictor` loads generalist SAM models.
```
