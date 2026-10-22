import cv2
import os

def generate_video():
    image_folder = '_out_parking_2' # Folder with the captured frames
    video_name = 'parking_result_7.mp4'
    fps = 20  # <--- ADJUST (20 for a 0.05 s step, 30 for 0.033 s)

    # Get the sorted list of images
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort() 

    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to get the resolution
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    print(f"Creating video from {len(images)} images...")

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    print(f"Video saved as {video_name}!")

if __name__ == "__main__":
    generate_video()