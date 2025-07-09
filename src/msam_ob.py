import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.util
import cv2
import torch
import micro_sam.util as util
import os

class ImageHandler:
    """
    Handles loading, preparing, and displaying the image.
    This class does NOT interact with the console.
    """
    def __init__(self):
        self.raw_image = None
        self.prepared_image = None
        self.height = 0
        self.width = 0
        self.title = ""

    def load_from_local_file(self, path: str) -> bool:
        """Loads an image from a local file path. Returns True on success."""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The file was not found at '{path}'.")
            self.raw_image = skimage.io.imread(path)
            self.title = f"Loaded: {os.path.basename(path)}"
            print(f"Successfully loaded image from {path}")
            return True
        except Exception as e:
            print(f"An error occurred while loading the file: {e}")
            self.raw_image = None
            return False

    def load_cells3d(self) -> bool:
        """Tries to load the cells3d image. Returns True on success."""
        try:
            from skimage.data import cells3d
            self.raw_image = cells3d()[30, 1]
            self.title = "Successfully Loaded cells3d Image"
            print(self.title)
            return True
        except Exception as e:
            print(f"Could not load cells3d image: {e}. It might need to be downloaded.")
            self.raw_image = None
            return False

    def load_dummy_image(self) -> bool:
        """Generates a simple dummy image. Returns True on success."""
        self._create_dummy_image()
        print("Successfully generated a dummy image.")
        return True

    def _create_dummy_image(self):
        """Generates a simple dummy image and sets it as the raw_image."""
        img_gray = np.zeros((256, 256), dtype=np.uint8)
        center_x, center_y = 128, 128
        radius = 50
        y, x = np.ogrid[:256, :256]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        img_gray[dist <= radius] = 200
        noise = np.random.normal(0, 10, img_gray.shape).astype(np.uint8)
        self.raw_image = skimage.util.img_as_ubyte(np.clip(img_gray + noise, 0, 255))
        self.title = "Created Dummy Image"

    def _prepare(self):
        """Prepares the image for the SAM model (uint8, 3-channel RGB)."""
        image = self.raw_image
        if image.dtype != np.uint8:
            print("Warning: Image is not in uint8 format. Converting now.")
            image = skimage.util.img_as_ubyte(image)

        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)
        
        print(f"Image shape after gray2rgb: {image.shape}")
        print(f"Image dtype after gray2rgb: {image.dtype}")

        # Resize the image so that the longest side is 1024
        target_size = 1024
        long_side = max(image.shape[:2])
        scale = target_size / long_side
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_size)

        print(f"Image shape after resizing: {image.shape}")
        print(f"Image dtype after resizing: {image.dtype}")

        # Convert to RGB if it's not already
        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)
        elif image.shape[2] == 4:
            image = image[:, :, :3]  # Remove alpha channel

        print(f"Image shape after removing alpha channel: {image.shape}")
        print(f"Image dtype after removing alpha channel: {image.dtype}")
        
        self.prepared_image = image
        self.height, self.width, _ = self.prepared_image.shape

    def prepare_and_get_image(self) -> np.ndarray:
        """Prepares the loaded raw image and returns it."""
        if self.raw_image is None:
            raise RuntimeError("Cannot prepare image: No raw image has been loaded.")
        self._prepare()
        return self.prepared_image

    def show(self):
        """Displays the currently loaded raw image."""
        if self.raw_image is None:
            print("No image loaded.")
            return
        plt.figure(figsize=(6, 6))
        plt.imshow(self.raw_image, cmap='gray' if self.raw_image.ndim == 2 else None)
        plt.title(self.title)
        plt.axis('off')
        plt.show()


class SegmentationModel:
    """
    Manages the SAM model, including initialization and prediction.
    """
    def __init__(self, model_type="vit_b_lm"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.predictor = None
        self.is_image_set = False
        print(f"Using device: {self.device}")

    def initialize(self):
        """Initializes the SAM predictor."""
        print(f"Initializing SAM predictor ({self.model_type})...")
        self.predictor = util.get_sam_model(model_type=self.model_type, device=self.device)
        print("Predictor initialized.")

    def set_image(self, image: np.ndarray):
        """Sets the image in the predictor to compute embeddings."""
        if self.predictor is None:
            raise RuntimeError("Model is not initialized. Call 'initialize()' first.")
        print("Setting image in the predictor (computing embeddings)...")
        print(f"Image shape before setting: {image.shape}")
        self.predictor.set_image(image)
        self.is_image_set = True
        print("Embeddings computed and predictor is ready.")

    def predict_from_point(self, point_coords: np.ndarray):
        """Performs segmentation from a single point prompt."""
        if not self.is_image_set:
            raise RuntimeError("An image must be set before prediction. Call 'set_image()' first.")
        
        input_labels = np.array([1])  # 1 for a foreground point
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=input_labels,
            multimask_output=True,
        )
        best_mask_idx = np.argmax(scores)
        final_mask = masks[best_mask_idx]
        print(f"Best mask found: Pixels = {np.sum(final_mask)}, Score = {scores[best_mask_idx]:.4f}")
        return final_mask


class InteractiveSegmenter:
    """
    Orchestrates the interactive segmentation application.
    This class handles all user console interaction.
    """
    def __init__(self, model_type="vit_b_lm"):
        self.image_handler = ImageHandler()
        self.model = SegmentationModel(model_type=model_type)

    def _get_image_from_user(self) -> np.ndarray:
        """
        Handles the user interaction for choosing and loading an image.
        Returns the prepared image upon successful loading.
        """
        while True:
            print("\nPlease choose an image source:")
            print("1. Load a local image file")
            print("2. Load the 'cells3d' sample image")
            print("3. Generate a dummy image")
            choice = input("Enter your choice (1, 2, or 3): ")

            success = False
            if choice == '1':
                path = input("Enter the full path to your image file: ")
                success = self.image_handler.load_from_local_file(path)
            elif choice == '2':
                success = self.image_handler.load_cells3d()
            elif choice == '3':
                success = self.image_handler.load_dummy_image()
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

            if success:
                return self.image_handler.prepare_and_get_image()
            else:
                print("Image loading failed. Please try again.")

    def _visualize_segmentation(self, image, input_points, final_mask):
        """Displays the image with prompt and the resulting mask."""
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.scatter(input_points[:, 0], input_points[:, 1], color='green', marker='*', s=250)
        plt.title("Image with Point Prompt")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        color = np.array([30/255, 144/255, 255/255, 0.6])
        mask_overlay = final_mask[..., np.newaxis] * color.reshape(1, 1, -1)
        plt.imshow(mask_overlay)
        plt.title("Segmentation Result")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def run(self):
        """Runs the main application loop."""
        # 1. Load and prepare image by interacting with the user
        prepared_image = self._get_image_from_user()
        self.image_handler.show()

        # 2. Initialize model and set image
        self.model.initialize()
        self.model.set_image(prepared_image)

        # 3. Start interactive session
        center_y, center_x = self.image_handler.height // 2, self.image_handler.width // 2
        
        while True:
            try:
                x_input = input("Enter relative x coordinate (or 'q' to quit): ")
                if x_input.lower() == 'q': break
                y_input = input("Enter relative y coordinate (or 'q' to quit): ")
                if y_input.lower() == 'q': break

                relative_x, relative_y = int(x_input), int(y_input)
                
                abs_x = np.clip(center_x + relative_x, 0, self.image_handler.width - 1)
                abs_y = np.clip(center_y - relative_y, 0, self.image_handler.height - 1)
                
                input_points = np.array([[abs_x, abs_y]])
                print(f"Using a single positive point prompt at: {input_points[0]} for relative coords: [{relative_x}, {relative_y}]")

                final_mask = self.model.predict_from_point(input_points)
                self._visualize_segmentation(prepared_image, input_points, final_mask)

            except ValueError:
                print("Invalid input. Please enter integers for coordinates or 'q' to quit.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        
        print("Exiting program.")


if __name__ == "__main__":
    app = InteractiveSegmenter()
    app.run()
