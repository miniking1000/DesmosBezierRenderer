from moviepy.editor import *

def extract_audio(video_path, output_path):
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(output_path)
        print("Audio extracted successfully!")
    except Exception as e:
        print("Error:", str(e))

# Replace 'video_file_path' and 'output_audio_path' with your file paths
video_file_path = "frames.mp4"
output_audio_path = "sound.mp3"

extract_audio(video_file_path, output_audio_path)
