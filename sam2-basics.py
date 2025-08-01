import cv2 as cv
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import numpy as np


class MultiObjectSegmenter:
    def __init__(self):
        # Device setup
        if torch.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Load SAM2 predictor
        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large", device=self.device)

        # Storage for multiple masks
        self.masks = []  # List to store multiple masks
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Green, Red, Blue, Yellow

        # Storage for current prompts (for building one segmentation)
        self.current_prompts = []  # List of (x, y, label) tuples

        self.current_frame = None

    def mouse_click(self, event, x, y, flags, param):
        if self.current_frame is None:
            return

        if event == cv.EVENT_LBUTTONDOWN:
            # Left click = positive prompt (include)
            self.current_prompts.append((x, y, 1))
            print(f"Added positive prompt at ({x}, {y})")
            self.draw_prompts()

        elif event == cv.EVENT_RBUTTONDOWN:
            # Right click = negative prompt (exclude)
            self.current_prompts.append((x, y, 0))
            print(f"Added negative prompt at ({x}, {y})")
            self.draw_prompts()

    def draw_prompts(self):
        """Draw current prompts on the frame."""
        display_frame = self.current_frame.copy()

        # Draw existing masks first
        if len(self.masks) > 0:
            overlay = self.current_frame.copy()
            for i, mask in enumerate(self.masks):
                color = self.colors[i % len(self.colors)]
                overlay[mask] = color
            display_frame = cv.addWeighted(self.current_frame, 0.7, overlay, 0.3, 0)

        # Draw current prompts on top
        for x, y, label in self.current_prompts:
            if label == 1:  # Positive prompt
                cv.circle(display_frame, (x, y), 8, (0, 255, 0), -1)  # Green circle
                cv.putText(display_frame, "+", (x-5, y+5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:  # Negative prompt
                cv.circle(display_frame, (x, y), 8, (0, 0, 255), -1)  # Red circle
                cv.putText(display_frame, "-", (x-5, y+5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv.imshow("SAM2", display_frame)

    def segment_with_prompts(self):
        """Create/update segmentation using all current prompts."""
        if len(self.current_prompts) == 0:
            return

        # Set image for SAM2
        self.predictor.set_image(self.current_frame)

        # Prepare prompts for SAM2
        points = np.array([[x, y] for x, y, label in self.current_prompts])
        labels = np.array([label for x, y, label in self.current_prompts])

        # Get segmentation
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels
        )

        # Show preview of current segmentation (don't commit yet)
        if len(masks) > 0:
            preview_mask = masks[-1].astype(bool)
            self.show_preview(preview_mask)

    def show_preview(self, preview_mask):
        """Show preview of current segmentation being built."""
        display_frame = self.current_frame.copy()

        # Draw existing committed masks first
        if len(self.masks) > 0:
            overlay = self.current_frame.copy()
            for i, mask in enumerate(self.masks):
                color = self.colors[i % len(self.colors)]
                overlay[mask] = color
            display_frame = cv.addWeighted(self.current_frame, 0.7, overlay, 0.3, 0)

        # Draw preview mask with a different style (more transparent)
        preview_overlay = display_frame.copy()
        preview_color = self.colors[len(self.masks) % len(self.colors)]
        preview_overlay[preview_mask] = preview_color
        display_frame = cv.addWeighted(display_frame, 0.8, preview_overlay, 0.2, 0)

        # Draw current prompts on top
        for x, y, label in self.current_prompts:
            if label == 1:  # Positive prompt
                cv.circle(display_frame, (x, y), 8, (0, 255, 0), -1)  # Green circle
                cv.putText(display_frame, "+", (x-5, y+5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:  # Negative prompt
                cv.circle(display_frame, (x, y), 8, (0, 0, 255), -1)  # Red circle
                cv.putText(display_frame, "-", (x-5, y+5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add instruction text
        cv.putText(display_frame, "PREVIEW - Press ENTER to commit, ESC to cancel",
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv.imshow("SAM2", display_frame)

        # Store preview for potential commit
        self.current_preview = preview_mask

    def commit_segmentation(self):
        """Commit the current preview as a final mask."""
        if hasattr(self, 'current_preview'):
            self.masks.append(self.current_preview)
            print(f"Committed mask {len(self.masks)}")

            # Clear current prompts and preview
            self.current_prompts = []
            delattr(self, 'current_preview')

            # Show final result
            self.display_all_masks()

    def cancel_segmentation(self):
        """Cancel current segmentation and clear prompts."""
        self.current_prompts = []
        if hasattr(self, 'current_preview'):
            delattr(self, 'current_preview')

        # Show existing masks only
        self.display_all_masks()
        print("Cancelled current segmentation")

    def display_all_masks(self):
        if self.current_frame is None or len(self.masks) == 0:
            return

        # Start with original frame
        overlay = self.current_frame.copy()

        # Add each mask with different color
        for i, mask in enumerate(self.masks):
            color = self.colors[i % len(self.colors)]  # Cycle through colors
            overlay[mask] = color

        # Blend this mask with the result
        result = cv.addWeighted(self.current_frame, 0.7, overlay, 0.3, 0)

        cv.imshow("SAM2", result)

    def run(self, video_path):
        video = cv.VideoCapture(video_path)

        ret, frame = video.read()
        if not ret:
            print("Could not read video")
            return

        self.current_frame = frame

        # Display initial frame
        cv.imshow("SAM2", frame)
        cv.setMouseCallback("SAM2", self.mouse_click)

        print("Left click: positive prompt (+), Right click: negative prompt (-)")
        print("SPACE: preview segmentation, ENTER: commit, ESC: cancel, 'r': reset all, 'q': quit")

        while True:
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.masks = []
                self.current_prompts = []
                if hasattr(self, 'current_preview'):
                    delattr(self, 'current_preview')
                cv.imshow("SAM2", self.current_frame)
                print("Reset everything")
            elif key == ord(' '):  # Spacebar - preview
                self.segment_with_prompts()
            elif key == 13:  # Enter key - commit
                self.commit_segmentation()
            elif key == 27:  # Escape key - cancel
                self.cancel_segmentation()

        video.release()
        cv.destroyAllWindows()


def main():
    segmenter = MultiObjectSegmenter()
    segmenter.run("resources/party.mp4")


if __name__ == "__main__":
    main()