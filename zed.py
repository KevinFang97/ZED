import sys
import cv2
import numpy as np
import glob
import os
from moviepy.editor import *

def getImgPath(vid_name,vid_path,num_digits=6):
    img_folder_path = 'data/'+vid_name
    txt_path = img_folder_path+'/img_num.txt'
    try:
        if os.path.exists(img_folder_path):
            return np.loadtxt(txt_path[0]), img_folder_path
        else:
            os.makedirs(img_folder_path)
    except OSError:
        print('failed to create path for images')

    curr_frame = 0

    cam = cv2.VideoCapture(vid_path)

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
     
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.
     
    if int(major_ver)  < 3 :
        fps = cam.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = cam.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    while(True):

        # reading from frame 
        ret,frame = cam.read()

        if ret:
            img_path = img_folder_path + '/' + str(curr_frame).zfill(num_digits) + '.jpg'
            if curr_frame % 10 == 0:
                print('Creating...' + img_path) 
  
            # writing the extracted images 
            cv2.imwrite(img_path, frame) 
  
            # increasing counter so that it will 
            # show how many frames are created 
            curr_frame += 1
        else: 
            break

    np.savetxt(txt_path, np.array(curr_frame).reshape(1,))

    return curr_frame, img_folder_path, fps

def process_img_canny(img, cache):
    edges = cv2.Canny(img,10,100)
    edges = np.stack((edges,)*3, axis=-1)
    return edges, cache

def processImages(img_num, img_folder_path, vid_name, num_digits=6):
    cache = None
    result_name = vid_name + '_canny'
    result_path = 'result/' + result_name

    try:
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    except OSError:
        print('failed to create path for results')

    for img_idx in range(img_num):
        img = cv2.imread(img_folder_path+'/'+str(img_idx).zfill(num_digits) + '.jpg')
        img_processed, cache = process_img_canny(img, cache)
        cv2.imwrite(result_path+'/'+str(img_idx).zfill(num_digits) + '.jpg', img_processed)

    return result_path, result_name

def genVideos(result_path,result_name,img_num, fps,num_digits=6):
    video_name = 'video/' + result_name + '.avi'

    images = [result_path+'/'+str(img_idx).zfill(num_digits) + '.jpg' for img_idx in range(img_num)]
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape


    #video = cv2.VideoWriter(video_name, -1, 1, (width,height))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc,
                             fps, (width,height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

    return video_name

def addAudio(vid_name, video_name):
    video = VideoFileClip(vid_name)
    audio = video.audio
    my_clip = VideoFileClip(video_name)
    final_clip = my_clip.set_audio(audio)
    final_clip.write_videofile(video_name[:-3] + 'mp4')

if __name__ == '__main__':
    vid_name, vid_path = sys.argv[1:3]
    img_num, img_folder_path, fps = getImgPath(vid_name,vid_path)
    result_path, result_name = processImages(img_num, img_folder_path, vid_name)
    video_name = genVideos(result_path,result_name,img_num,fps)
    addAudio(vid_path, video_name)