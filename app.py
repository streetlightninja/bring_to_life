import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import warnings
import streamlit as st



warnings.filterwarnings("ignore")
imageio.plugins.ffmpeg.download() 
source = st.file_uploader("Upload the source image")
reader = st.file_uploader("Upload the referece video")

if source:
    with open("tempImg.jpg","wb") as f:
        f.write(source.read())
        f.close()
    source_image = imageio.imread("tempImg.jpg")
if reader: 
    with open("tempRef.mp4","wb") as f:
        f.write(reader.read())
        f.close()
    reader = imageio.get_reader("tempRef.mp4")


#Resize image and video to 256x256
if source and reader:
    source_image = resize(source_image, (256, 256))[..., :3]

    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    from demo import load_checkpoints
    generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', 
                            checkpoint_path='checkpoint/vox-cpk.pth.tar')
    from demo import make_animation
    from skimage import img_as_ubyte

    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

    #save resulting video
    imageio.mimsave('generated.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)
    st.video('generated.mp4')