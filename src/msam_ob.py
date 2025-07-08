import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.util
import torch
import micro_sam.util as util

class ImageHandler:
    """
    Handles loading, preparing, and displaying the image.
    """
    def __init__(self):
        self.raw_image = None
        self.prepared_image = None
        self.height = 0
        self.width = 0
        self.title = ""

    def load(self, use_dummy=False):
        """
        Tries to load the cells3d image. If it fails or if use_dummy is True,
        creates and loads a dummy image.
        """
        if not use_dummy:
            try:
                from skimage.data import cells3d
                self.raw_image = cells3d()[30, 1]
                self.title = "Successfully Loaded cells3d Image"
                print(self.title)
            except Exception as e:
                print(f"Could not load cells3d image: {e}. Creating a dummy image.")
                self._create_dummy_image()
        else:
            self._create_dummy_image()
        
        self._prepare()
        return self.prepared_image

    def _create_dummy_image(self):
        """Generates a simple dummy image."""
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
        """
        Prepares the image for the SAM model (uint8, 3-channel RGB).
        """
        image = self.raw_image
        if image.dtype != np.uint8:
            print("Warning: Image is not in uint8 format. Converting now.")
            image = skimage.util.img_as_ubyte(image)

        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)
        
        self.prepared_image = image
        self.height, self.width, _ = self.prepared_image.shape

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
    """
    def __init__(self, model_type="vit_b_lm"):
        self.image_handler = ImageHandler()
        self.model = SegmentationModel(model_type=model_type)

    def _visualize_segmentation(self, image, input_points, final_mask):
        """Displays the image with prompt and the resulting mask."""
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
        color = np.array([30/255, 144/255, 255/255, 0.6])
        mask_overlay = final_mask[..., np.newaxis] * color.reshape(1, 1, -1)
        plt.imshow(mask_overlay)
        plt.title("Segmentation Result")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def run(self):
        """Runs the main application loop."""
        # 1. Load and prepare image
        prepared_image = self.image_handler.load()
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
                
                # Calculate and clamp coordinates
                abs_x = np.clip(center_x + relative_x, 0, self.image_handler.width - 1)
                abs_y = np.clip(center_y - relative_y, 0, self.image_handler.height - 1)
                
                input_points = np.array([[abs_x, abs_y]])
                print(f"Using a single positive point prompt at: {input_points[0]} for {[int(abs_x - center_x), int(center_y - abs_y)]}")

                # Get segmentation mask
                final_mask = self.model.predict_from_point(input_points)

                # Visualize result
                self._visualize_segmentation(prepared_image, input_points, final_mask)

            except ValueError:
                print("Invalid input. Please enter integers for coordinates or 'q' to quit.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        
        print("Exiting program.")


if __name__ == "__main__":
    app = InteractiveSegmenter()
    app.run()
