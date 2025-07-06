import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import torch
import micro_sam.util as util

def segment_with_a_single_point(image, model_type: str = "vit_b"):
    """
    Loads an image and segments an object using a single point prompt at the center.
    This version correctly uses the micro-sam utility to get a predictor directly.
    """
    # --- 1. Load Image ---
    if isinstance(image, str):
        image = skimage.io.imread(image)
    print(f"Max value before normalization: {np.max(image)}")
    image = image / np.max(image)
    print(f"Max value after normalization: {np.max(image)}")
    plt.imshow(image)
    plt.title("Normalized Image")
    plt.show()
    # The SAM model expects a 3-channel (RGB) image.
    if image.ndim == 2:
        image = skimage.color.gray2rgb(image)
        print(f"Max value after gray2rgb: {np.max(image)}")
        plt.imshow(image)
        plt.title("Image after gray2rgb")
        plt.show()
    
    # It also expects the image to be in uint8 format.
    if image.dtype != np.uint8:
        print("Converting image to uint8.")
        image = (image / np.max(image)) * 255
        image = image.astype(np.uint8)
        print(f"Max value before normalization: {np.max(image)}")
        plt.imshow(image)
        plt.title("Image after astype(uint8)")
        plt.show()

    # --- 2. Initialize SAM Predictor (The Final, Correct Way) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # This utility function downloads the model and returns a ready-to-use predictor.
    print("Initializing predictor...")
    predictor = util.get_sam_model(model_type=model_type, device=device)
    print("Predictor initialized.")

    # --- 3. Set Image for Predictor ---
    print("Setting image in the predictor (computing embeddings)...")
    predictor.set_image(image)
    print("Embeddings computed.")

    # --- 4. Create a Simple Point Prompt ---
    height, width, _ = image.shape
    center_y, center_x = height // 2, width // 2
    input_points = np.array([[center_x + 20, center_y - 20]])
    input_labels = np.array([1]) # 1 for a foreground point

    print(f"Using a single positive point prompt at: {input_points[0]}")

    # --- 5. Perform Segmentation ---
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )
    # Print the number of pixels and score for each mask
    for i, mask in enumerate(masks):
        print(f"Mask {i+1}: Pixels = {np.sum(mask)}, Score = {scores[i]:.4f}")
    final_mask = masks[np.argmax(scores)]

    # --- 6. Visualize the Result (Using only Matplotlib) ---
    plt.figure(figsize=(12, 6))

    # Visualize the prompt point
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    pos_points = input_points[input_labels == 1]
    plt.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=250)
    plt.title("Image with Point Prompt")
    plt.axis('off')

    # Visualize the mask
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    color = np.array([30/255, 144/255, 255/255, 0.6]) # Dodger blue with 60% alpha
    mask_image = final_mask.reshape(final_mask.shape[0], final_mask.shape[1], 1) * color.reshape(1, 1, -1)
    plt.imshow(mask_image)
    plt.title("Segmentation Result")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    def create_dummy_image(path="dummy_object.tif"):
        img_gray = np.zeros((256, 256), dtype=np.uint8)
        center_x, center_y = 128, 128
        radius = 50
        y, x = np.ogrid[:256, :256]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        img_gray[dist <= radius] = 200
        img_gray = (img_gray + np.random.normal(0, 10, img_gray.shape)).astype(np.uint8)
        skimage.io.imsave(path, img_gray, check_contrast=False)
        return path

    image = None
    try:
        from skimage.data import cells3d
        #raise Exception("No image!")
        image = cells3d()[30, 1]
        print("cells3d image loaded successfully")
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")
        print(f"Number of color channels: {image.ndim} ")
        print(f"Image min value: {np.min(image)}")
        print(f"Image max value: {np.max(image)}")
        plt.imshow(image)
        plt.title("cells3d Image")
        plt.show()
    except Exception as e:
        print(f"Error loading cells3d image: {e}")
        IMAGE_FILE = create_dummy_image()
        image = skimage.io.imread(IMAGE_FILE)
        print("Dummy image created")
        plt.imshow(image)
        plt.title("Dummy Image")
        plt.show()

    if image is not None:
        segment_with_a_single_point(image, "vit_b_lm")
