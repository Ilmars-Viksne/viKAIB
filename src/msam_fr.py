import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.util
import torch
import micro_sam.util as util
import os
import glob
from tqdm import tqdm

# Note: The video splitting feature requires 'opencv-python' -> pip install opencv-python
# This version has NO OTHER external dependencies for tracking.

class ImageHandler:
    """
    Handles loading, preparing, and displaying images.
    Can load single images or a sequence of frames from a folder.
    This class does NOT interact with the console.
    """
    def __init__(self):
        self.raw_image = None
        self.prepared_image = None
        self.height = 0
        self.width = 0
        self.title = ""
        self.image_paths = []

    def load_from_local_file(self, path: str) -> bool:
        """Loads a single image from a local file path. Returns True on success."""
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
            print(f"Could not load cells3d image: {e}")
            self.raw_image = None
            return False

    def load_dummy_image(self) -> bool:
        """Generates a simple dummy image. Returns True on success."""
        self._create_dummy_image()
        print("Successfully generated a dummy image.")
        return True

    def load_from_folder(self, folder_path: str) -> bool:
        """Loads the first frame from a folder of images. Returns True on success."""
        try:
            if not os.path.isdir(folder_path):
                raise NotADirectoryError(f"The path '{folder_path}' is not a valid directory.")
            
            self.image_paths = [] # Reset paths
            extensions = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
            for ext in extensions:
                self.image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
            
            if not self.image_paths:
                print("No image files found in the specified folder.")
                return False

            self.image_paths.sort() # Sort alphabetically/numerically
            self.raw_image = skimage.io.imread(self.image_paths[0])
            self.title = f"Loaded First Frame: {os.path.basename(self.image_paths[0])}"
            print(f"Found {len(self.image_paths)} frames. Loaded first frame for prompting.")
            return True
        except Exception as e:
            print(f"An error occurred while loading from the folder: {e}")
            self.image_paths = []
            self.raw_image = None
            return False

    @staticmethod
    def prepare_image(raw_image: np.ndarray) -> np.ndarray:
        """Prepares a raw image array for the SAM model (uint8, 3-channel RGB)."""
        image = raw_image
        if image.dtype != np.uint8:
            image = skimage.util.img_as_ubyte(image)
        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)
        if image.shape[2] == 4:
            image = skimage.color.rgba2rgb(image)
        return image

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

    def prepare_and_get_image(self) -> np.ndarray:
        """Prepares the loaded raw image and returns it."""
        if self.raw_image is None:
            raise RuntimeError("Cannot prepare image: No raw image has been loaded.")
        self.prepared_image = self.prepare_image(self.raw_image)
        self.height, self.width, _ = self.prepared_image.shape
        return self.prepared_image

    def show(self):
        """Displays the currently loaded raw image."""
        if self.raw_image is None:
            print("No image loaded.")
            return
        plt.figure(figsize=(8, 8))
        plt.imshow(self.raw_image, cmap='gray' if self.raw_image.ndim == 2 else None)
        plt.title(self.title)
        plt.axis('off')
        plt.show()


class SegmentationModel:
    """
    Manages the SAM model for single-image prediction.
    """
    def __init__(self, model_type="vit_b_lm"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.predictor = None
        self.is_image_set = False
        print(f"Using device: {self.device}")

    def initialize(self):
        """Initializes the SAM predictor."""
        if self.predictor is None:
            print(f"Initializing SAM predictor ({self.model_type})...")
            self.predictor = util.get_sam_model(model_type=self.model_type, device=self.device)
            print("Predictor initialized.")

    def set_image(self, image: np.ndarray, silent: bool = False):
        """
        Sets the image in the predictor to compute embeddings.
        If silent is True, it will not print status messages.
        """
        if self.predictor is None:
            raise RuntimeError("Model is not initialized. Call 'initialize()' first.")
        
        if not silent:
            print("Setting image in the predictor (computing embeddings)...")
            
        self.predictor.set_image(image)
        self.is_image_set = True
        
        if not silent:
            print("Embeddings computed and predictor is ready.")

    def predict_from_point(self, point_coords: np.ndarray):
        """Performs segmentation from a single point prompt."""
        if not self.is_image_set:
            raise RuntimeError("An image must be set before prediction. Call 'set_image()' first.")
        
        input_labels = np.array([1])
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords, point_labels=input_labels, multimask_output=True,
        )
        best_mask_idx = np.argmax(scores)
        final_mask = masks[best_mask_idx]
        return final_mask


class InteractiveSegmenter:
    """
    Orchestrates the application, handling video splitting, interactive segmentation, and tracking.
    """
    def __init__(self, model_type="vit_b_lm"):
        self.image_handler = ImageHandler()
        self.model = SegmentationModel(model_type=model_type)

    def _split_video_to_frames(self):
        """Handles the user interaction for splitting a video into frames."""
        try:
            import cv2
        except ImportError:
            print("\nError: The 'opencv-python' library is required for this feature.")
            print("Please install it by running: pip install opencv-python")
            return

        video_path = input("Enter the full path to your video file: ")
        output_folder = input("Enter the path for the output folder to save frames: ")

        try:
            if not os.path.exists(video_path):
                print(f"Error: Video file not found at '{video_path}'")
                return

            os.makedirs(output_folder, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Could not open video file.")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            
            print(f"Splitting video '{os.path.basename(video_path)}' into frames...")
            with tqdm(total=total_frames, desc="Splitting Video") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break # End of video
                    
                    frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.png")
                    cv2.imwrite(frame_filename, frame)
                    
                    frame_count += 1
                    pbar.update(1)
            
            cap.release()
            print(f"\nSuccessfully split video into {frame_count} frames.")
            print(f"Frames are saved in: '{output_folder}'")

        except Exception as e:
            print(f"An unexpected error occurred during video splitting: {e}")

    def _visualize_segmentation(self, image, input_points, final_mask, title_suffix=""):
        """Displays the image with prompt and the resulting mask."""
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        if input_points is not None:
            plt.scatter(input_points[:, 0], input_points[:, 1], color='green', marker='*', s=250)
        plt.title(f"Image with Point Prompt{title_suffix}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        color = np.array([30/255, 144/255, 255/255, 0.6])
        mask_overlay = final_mask[..., np.newaxis] * color.reshape(1, 1, -1)
        plt.imshow(mask_overlay)
        plt.title(f"Segmentation Result{title_suffix}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _save_tracking_result(self, image, mask, output_path):
        """Saves a single frame of the tracking result to a file."""
        fig, ax = plt.subplots(figsize=(image.shape[1]/100, image.shape[0]/100), dpi=100)
        ax.imshow(image)
        color = np.array([30/255, 144/255, 255/255, 0.6])
        mask_overlay = mask[..., np.newaxis] * color.reshape(1, 1, -1)
        ax.imshow(mask_overlay)
        ax.axis('off')
        fig.tight_layout(pad=0)
        plt.savefig(output_path)
        plt.close(fig)

    def _get_point_prompt(self):
        """Prompts user for a point and returns it."""
        while True:
            try:
                center_y, center_x = self.image_handler.height // 2, self.image_handler.width // 2
                x_input = input("Enter relative x coordinate for the object: ")
                y_input = input("Enter relative y coordinate for the object: ")
                relative_x, relative_y = int(x_input), int(y_input)
                
                abs_x = np.clip(center_x + relative_x, 0, self.image_handler.width - 1)
                abs_y = np.clip(center_y - relative_y, 0, self.image_handler.height - 1)
                
                input_points = np.array([[abs_x, abs_y]])
                print(f"Using prompt at: {input_points[0]} (relative: [{relative_x}, {relative_y}])")
                return input_points
            except ValueError:
                print("Invalid input. Please enter integers for coordinates.")

    def _run_tracking_session(self):
        """Runs the tracking workflow by re-segmenting each frame."""
        input_folder = input("Enter the full path to the folder with image frames: ")
        output_folder = input("Enter the full path for the output folder to save results: ")
        
        if not self.image_handler.load_from_folder(input_folder):
            return

        os.makedirs(output_folder, exist_ok=True)
        
        prepared_first_frame = self.image_handler.prepare_and_get_image()
        self.image_handler.show()
        
        self.model.initialize()
        # Set the first image with verbose output
        self.model.set_image(prepared_first_frame, silent=False)
        
        input_points = self._get_point_prompt()
        tracked_mask = self.model.predict_from_point(input_points)
        print(f"Initial mask found with {np.sum(tracked_mask)} pixels.")
        self._visualize_segmentation(prepared_first_frame, input_points, tracked_mask, " (First Frame)")

        save_path = os.path.join(output_folder, os.path.basename(self.image_handler.image_paths[0]))
        self._save_tracking_result(prepared_first_frame, tracked_mask, save_path)
        
        remaining_frames = len(self.image_handler.image_paths) - 1
        if remaining_frames > 0:
            print(f"Starting tracking for the remaining {remaining_frames} frames...")
            
            for i in tqdm(range(1, len(self.image_handler.image_paths)), desc="Tracking Frames"):
                if np.sum(tracked_mask) == 0:
                    print(f"\nWarning: Object lost at frame {i-1}. Stopping track.")
                    break

                y_coords, x_coords = np.where(tracked_mask)
                center_y = int(y_coords.mean())
                center_x = int(x_coords.mean())
                new_prompt = np.array([[center_x, center_y]])
                
                frame_path = self.image_handler.image_paths[i]
                raw_next_frame = skimage.io.imread(frame_path)
                prepared_next_frame = self.image_handler.prepare_image(raw_next_frame)
                
                # Set subsequent images silently to keep the progress bar clean
                self.model.set_image(prepared_next_frame, silent=True)
                tracked_mask = self.model.predict_from_point(new_prompt)
                
                save_path = os.path.join(output_folder, os.path.basename(frame_path))
                self._save_tracking_result(prepared_next_frame, tracked_mask, save_path)
            
        print(f"\nTracking complete. Results saved in '{output_folder}'.")

    def _run_interactive_session(self):
        """Runs the original workflow for interactively segmenting a single image."""
        prepared_image = None
        while prepared_image is None:
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
                print("Invalid choice.")

            if success:
                prepared_image = self.image_handler.prepare_and_get_image()
            else:
                print("Image loading failed. Please try again.")
        
        self.image_handler.show()
        self.model.initialize()
        self.model.set_image(prepared_image, silent=False)

        while True:
            try:
                prompt = input("Enter relative coordinates or 'q' to quit. (e.g. '10 20' or 'q'): ")
                if prompt.lower() == 'q': break
                
                parts = prompt.split()
                if len(parts) != 2: raise ValueError("Please provide two numbers for x and y.")

                relative_x, relative_y = int(parts[0]), int(parts[1])
                center_y, center_x = self.image_handler.height // 2, self.image_handler.width // 2
                abs_x = np.clip(center_x + relative_x, 0, self.image_handler.width - 1)
                abs_y = np.clip(center_y - relative_y, 0, self.image_handler.height - 1)
                input_points = np.array([[abs_x, abs_y]])
                print(f"Using prompt at: {input_points[0]} (relative: [{relative_x}, {relative_y}])")

                final_mask = self.model.predict_from_point(input_points)
                print(f"Mask generated with {np.sum(final_mask)} pixels.")
                self._visualize_segmentation(prepared_image, input_points, final_mask)

            except ValueError as e:
                print(f"Invalid input: {e}. Please enter two integers or 'q'.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    def run(self):
        """Starts the application by asking the user for the desired workflow."""
        print("Welcome to the Interactive Segmenter and Tracker!")
        while True:
            print("\n" + "="*30)
            print("Choose your workflow:")
            print("1. Interactively segment a single image")
            print("2. Track object in a folder of frames")
            print("3. Split a video into frames")
            print("q. Quit")
            print("="*30)
            choice = input("Enter your choice (1, 2, 3, or q): ")

            if choice == '1':
                self._run_interactive_session()
            elif choice == '2':
                self._run_tracking_session()
            elif choice == '3':
                self._split_video_to_frames()
            elif choice.lower() == 'q':
                break
            else:
                print("Invalid choice. Please try again.")
        
        print("Exiting program.")


if __name__ == "__main__":
    app = InteractiveSegmenter()
    app.run()
