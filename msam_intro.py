import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.util
import torch
import warnings
import micro_sam.util as util

def load_image(display_image: bool = True):
    """
    Tries to load the cells3d image. If it fails, creates and loads a dummy image.
    Optionally displays the loaded image.
    """
    title = ""
    try:
        from skimage.data import cells3d
        image = cells3d()[30, 1]
        title = "Successfully Loaded cells3d Image"
        print(title)
    except Exception as e:
        print(f"Could not load cells3d image: {e}. Creating a dummy image.")
        img_gray = np.zeros((256, 256), dtype=np.uint8)
        center_x, center_y = 128, 128
        radius = 50
        y, x = np.ogrid[:256, :256]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        img_gray[dist <= radius] = 200
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 10, img_gray.shape).astype(np.uint8)
        image = skimage.util.img_as_ubyte(np.clip(img_gray + noise, 0, 255))
        title = "Created Dummy Image"

    if display_image:
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray' if image.ndim == 2 else None)
        plt.title(title)
        plt.axis('off')
        plt.show()

    return image


def prepare_image(image: np.ndarray) -> np.ndarray:
    """
    Prepares the image for the SAM model.

    Ensures the image is in uint8 format and has 3 channels (RGB).
    """
    if image.dtype != np.uint8:
        warnings.warn("Image is not in uint8 format. Converting now.")
        image = skimage.util.img_as_ubyte(image)

    if image.ndim == 2:
        image = skimage.color.gray2rgb(image)

    return image


def visualize_segmentation(image, input_points, final_mask):
    """
    Displays the original image with the prompt point and the resulting segmentation mask.
    """
    plt.figure(figsize=(12, 6))

    # Subplot 1: Image with Point Prompt
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.scatter(input_points[:, 0], input_points[:, 1], color='green', marker='*', s=250)
    plt.title("Image with Point Prompt")
    plt.axis('off')

    # Subplot 2: Segmentation Result
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    # Create a colored overlay for the mask
    color = np.array([30/255, 144/255, 255/255, 0.6]) # Dodger blue with 60% alpha
    mask_overlay = final_mask[..., np.newaxis] * color.reshape(1, 1, -1)
    plt.imshow(mask_overlay)
    plt.title("Segmentation Result")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def segment_with_point(image, predictor, relative_x: int, relative_y: int):
    """
    Segments the image using a single point prompt with a centered origin.
    Clamps coordinates to the image boundary if they fall outside.
    """
    height, width, _ = image.shape
    center_y, center_x = height // 2, width // 2

    # Calculate absolute coordinates with an inverted y-axis for intuitive control
    absolute_x = center_x + relative_x
    absolute_y = center_y - relative_y

    # Clamp coordinates to image boundaries independently and warn if changed
    final_x = np.clip(absolute_x, 0, width - 1)
    final_y = np.clip(absolute_y, 0, height - 1)

    if final_x != absolute_x:
        print(f"Warning: X coordinate ({absolute_x}) was outside the image boundary [0, {width - 1}] and has been clamped to {final_x}.")
    if final_y != absolute_y:
        print(f"Warning: Y coordinate ({absolute_y}) was outside the image boundary [0, {height - 1}] and has been clamped to {final_y}.")

    input_points = np.array([[final_x, final_y]])
    input_labels = np.array([1])  # 1 for a foreground point

    print(f"Using a single positive point prompt at: {input_points[0]}")

    # Perform segmentation
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )

    # Select the best mask based on the score
    best_mask_index = np.argmax(scores)
    final_mask = masks[best_mask_index]
    print(f"Best mask found (Mask {best_mask_index + 1}): Pixels = {np.sum(final_mask)}, Score = {scores[best_mask_index]:.4f}")

    # Visualize the result
    visualize_segmentation(image, input_points, final_mask)


def run_interactive_session(image, predictor):
    """
    Runs the main loop to get user input for segmentation.
    """
    while True:
        try:
            x_input = input("Enter relative x coordinate (or 'q' to quit): ")
            if x_input.lower() == 'q':
                break
            y_input = input("Enter relative y coordinate (or 'q' to quit): ")
            if y_input.lower() == 'q':
                break

            relative_x = int(x_input)
            relative_y = int(y_input)
            segment_with_point(image, predictor, relative_x, relative_y)

        except ValueError:
            print("Invalid input. Please enter integers for coordinates or 'q' to quit.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # 1. Load the image and display it by default
    image_data = load_image(display_image=True)

    # 2. Prepare the image for the model
    prepared_image = prepare_image(image_data)

    # 3. Initialize the SAM predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Initializing SAM predictor...")
    sam_predictor = util.get_sam_model(model_type="vit_b_lm", device=device)
    print("Predictor initialized.")

    # 4. Set the image in the predictor (computes embeddings)
    print("Setting image in the predictor...")
    sam_predictor.set_image(prepared_image)
    print("Embeddings computed and predictor is ready.")

    # 5. Start interactive segmentation session
    run_interactive_session(prepared_image, sam_predictor)

    print("Exiting program.")
