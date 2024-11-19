import cv2
from ultralytics import solutions
import sys


def save_heatmap_video(
    input_video_path, output_video_path, model_path="models/yolo11n.pt"
):
    """
    Process a video file and save a heatmap visualization.

    Args:
        input_video_path (str): Path to input video file
        output_video_path (str): Path to save output video
        model_path (str): Path to YOLO model file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {input_video_path}")

        # Get video properties
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Initialize video writer
        video_writer = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

        # Initialize heatmap object
        heatmap_obj = solutions.Heatmap(
            colormap=cv2.COLORMAP_JET, show=True, model=model_path, show_in=True
        )

        frame_count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                if frame_count > 0:
                    print(f"Processing completed. Processed {frame_count} frames.")
                else:
                    print("Error: No frames were processed.")
                break

            # Generate heatmap on the frame
            try:
                im0 = heatmap_obj.generate_heatmap(im0)
                video_writer.write(im0)
                frame_count += 1

                # Optional: Print progress every 100 frames
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames...")

            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                continue

        return True

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

    finally:
        # Ensure resources are released even if an error occurs
        if "cap" in locals():
            cap.release()
        if "video_writer" in locals():
            video_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    input_path = "./assets/video/people_01.mp4"
    output_path = (
        "./outputs/video/people_output.mp4"  # Changed to .mp4 for better compatibility
    )

    success = save_heatmap_video(input_path, output_path)
    if success:
        print(f"Video successfully saved to {output_path}")
    else:
        print("Failed to process video")
