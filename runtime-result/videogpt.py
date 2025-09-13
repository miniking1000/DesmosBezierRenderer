import cv2
import os
from moviepy.editor import VideoFileClip, AudioFileClip

# Folder paths
image_folder = 'images'
video_name = 'video-tmp.mp4'
sound_file = 'sound.mp3'

# Get list of image files
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

# Read the first image to get dimensions
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Create video object
video = cv2.VideoWriter(video_name, 0, 30, (width, height))

# Write images to video
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# Release video resources
cv2.destroyAllWindows()
video.release()

# Add sound to the video
video_clip = VideoFileClip(video_name)
audio_clip = AudioFileClip(sound_file)
video_clip = video_clip.set_audio(audio_clip)

# Write the final video with sound
video_clip.write_videofile("video_with_sound.mp4", codec='libx264', fps=24)

# Remove the temporary video file without sound
os.remove(video_name)
