import xml.etree.ElementTree as ET
import pandas as pd
import cv2
import os
import numpy as np

def parseXML(file):
    # Parse positive frames from XML file
    tree = ET.parse(file)
    root = tree.getroot()

    df_cols = ["frame", "xtl", "ytl", "xbr", "ybr"]
    rows = []

    # Save in a pandas dataframe
    pos_frames = []
    for box in root.findall("./track/box"): #"./track/box[@keyframe='0']"
        rows.append({"frame": int(box.attrib["frame"]),
                                "xtl": float(box.attrib["xtl"]),
                                "ytl": float(box.attrib["ytl"]),
                                "xbr": float(box.attrib["xbr"]),
                                "ybr": float(box.attrib["ybr"])})
    df = pd.DataFrame(rows, columns = df_cols)
    return df

def extractFrames(video, df):
    # pos_frames has duplicates, so find unique frames and sort them in ascending order
    pos_frames = df["frame"].tolist()
    unique_pos_frames = list(set(pos_frames))
    unique_pos_frames.sort()

    # Extract frames and save to disk
    cap = cv2.VideoCapture(video)

    # THRESHOLD is how "far" (in # of frames) from positive frames is required
    # FREQUENCY is how often to extract a frame (every 10 frames, etc)
    THRESHOLD = 50
    FREQUENCY = 10
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(0,length):
        # Extract positive frames
        if i % FREQUENCY == 0 and i in unique_pos_frames:
            cap.set(1,int(i))
            res, frame = cap.read()
            w, h, layers = frame.shape
            # Resize images for use in neural net
            resized = cv2.resize(frame, (h//4, w//4))
            cv2.imwrite("positive/pos_"+video.split(".")[0]+'_frame_'+str(i)+'.jpg',resized)
        # Extract negative frames
        if i % FREQUENCY == 0 and (i < unique_pos_frames[0] - THRESHOLD or i > unique_pos_frames[-1] + THRESHOLD):
            cap.set(1,int(i))
            res, frame = cap.read()
            w, h, layers = frame.shape
            resized = cv2.resize(frame, (h//4, w//4))
            cv2.imwrite("negative/neg_"+video.split(".")[0]+'_frame_'+str(i)+'.jpg',resized)

def extractTest(video):
    # Same gist as the previous method but not limiting to postiive/ negative examples (unlabelled data)
    if not os.path.exists("test"):
        os.mkdir("test")
    cap = cv2.VideoCapture(video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FREQUENCY = 10
    for i in range(0,length):
        if i % FREQUENCY == 0:
            cap.set(1,int(i))
            res, frame = cap.read()
            w, h, layers = frame.shape
            resized = cv2.resize(frame, (h//4, w//4))
            cv2.imwrite("test/"+video.split(".")[0]+'_frame_'+str(i)+'.jpg',resized)

def drawBoundingBoxes(video, df):
    # Take frames and bounding box coordinates from dataframe
    pos_frames = df["frame"].tolist()
    xtl = df["xtl"].tolist()
    ytl = df["ytl"].tolist()
    xbr = df["xbr"].tolist()
    ybr = df["ybr"].tolist()
    unique_pos_frames = list(set(pos_frames))
    unique_pos_frames.sort()

    # Iterate through video frames
    cap = cv2.VideoCapture(video)

    for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        cap.set(1,int(i))
        res, frame = cap.read()
        # For positive examples, draw bounding box (if it already exists, use unique file name)
        if i in pos_frames:
            cv2.rectangle(frame, (int(xtl[i]), int(ytl[i])), (int(xbr[i]), int(ybr[i])), (255,0,0), 2)
            if not os.path.exists("boxes/pos_"+video.split(".")[0]+'_frame_'+str(pos_frames[i])+'.jpg'):
                cv2.imwrite("boxes/pos_"+video.split(".")[0]+'_frame_'+str(pos_frames[i])+'.jpg',frame)
            else:
                cv2.imwrite("boxes/pos_"+video.split(".")[0]+'_frame_'+str(pos_frames[i])+"_"+str(i)+'.jpg',frame)

if __name__ == '__main__':
    # Make directories for extracted frames
    if not os.path.exists("data/positive"):
        os.mkdir("data/positive")
    if not os.path.exists("data/negative"):
        os.mkdir("data/negative")
    if not os.path.exists("boxes"):
        os.mkdir("boxes")

    # Iterate through all .mp4 files in folder (except clean test video)
    # for file in os.listdir("."):
    #     if file.endswith(".mp4") and file != "clean.mp4":
    #         df = parseXML(file.split(".")[0]+'.xml')
    #         extractFrames(file, df)
    #         drawBoundingBoxes(file, df)
    # extractTest("clean.mp4")

    df = parseXML('10.xml')
    drawBoundingBoxes('10.mp4', df)
