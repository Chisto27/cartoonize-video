import streamlit as st
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
from io import BytesIO
import tempfile
import requests
from tqdm import tqdm
import os
import network
import guided_filter
import time

# Disable eager execution
tf.disable_eager_execution()

# Load the pre-trained model from Hugging Face
model_url = "https://huggingface.co/your-username/your-model-repo/resolve/main/saved_model.tar.gz"
model_path = tf.keras.utils.get_file("saved_model", model_url, untar=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Load the model
input_photo = tf.placeholder(tf.float32, [1, None, None, 3], name='input_photo')
network_out = network.unet_generator(input_photo)
final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)
final_out = tf.identity(final_out, name='final_output')

saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint(model_path))


def cartoonize_image(image):
    image = np.array(image)
    image = image / 127.5 - 1
    output = sess.run(final_out, feed_dict={input_photo: np.expand_dims(image, 0)})
    output = (output[0] + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


# Streamlit app interface
st.title("Video Cartoonizer")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = vidcap.read()
    frame_list = []
    while success:
        frame_list.append(image)
        success, image = vidcap.read()

    st.write(f"Extracted {len(frame_list)} frames from the video.")

    cartoonized_frames = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner('Cartoonizing frames...'):
        start_time = time.time()
        for i, frame in enumerate(tqdm(frame_list, desc="Cartoonizing frames")):
            cartoonized_frames.append(cartoonize_image(frame))
            progress = int((i + 1) / total_frames * 100)
            elapsed_time = time.time() - start_time
            time_per_frame = elapsed_time / (i + 1)
            remaining_time = (total_frames - (i + 1)) * time_per_frame
            status_text.text(
                f"Progress: {progress}% | Processed: {i + 1}/{total_frames} | Remaining: {total_frames - (i + 1)} | Time Remaining: {remaining_time:.2f} seconds")
            progress_bar.progress(progress / 100.0)

    st.write("Compiling cartoonized frames into a video...")

    height, width, layers = cartoonized_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "cartoonized_video.mp4"
    video = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    for frame in cartoonized_frames:
        video.write(frame)

    video.release()
    st.success("Cartoonized video created!")

    st.video(output_path)
