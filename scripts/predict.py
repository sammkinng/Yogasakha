from cv2 import VideoCapture
import numpy as np

from constants import POSE_LABELS
from utils.video_utils import get_video_duration_opencv
from utils.pose_utils import test_pose,extract_features

from collections import deque

from model.model import session,scaler



# Rolling window for stable predictions

WINDOW_SIZE = 5
pose_window = deque(maxlen=WINDOW_SIZE)


def process_video(video_path):
    """Runs pose classification on a video."""
    cap = VideoCapture(video_path)
    frame_count = 0
    
    # Frame skipping interval
    FRAME_INTERVAL =  get_video_duration_opencv(video_path)[0]# Process every 10th frame

    previous_pose = None
    pose_counter=1
    correct=[]
    # wrong=[]
    dbls=[0,0,0,0]
    mm={1:12,2:11,3:10,4:9}
    xtras=[1,1]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % FRAME_INTERVAL == 0:
            ftrs = extract_features(frame)
            if ftrs is not None:
                
                # Combine features
                features = np.array(ftrs).reshape(1, -1)
                features = scaler.transform(features)  # Normalize features
                
                # Predict pose
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                predictions = session.run([output_name], {input_name: features})[0]
                
                prediction=np.argmax(predictions, axis=1)[0]
                pose_name = POSE_LABELS[prediction]
                
                # Rolling window voting
                pose_window.append(pose_name)
                if len(set(pose_window)) == 1:  # If all frames in window predict the same
                    
                    if pose_name != previous_pose and pose_name !="Transition":
                        
                        rss=test_pose(prediction,frame)
                        if(rss[0]):
                        
                            if prediction<5:
                                if(dbls[prediction-1]):
                                    prediction=mm[prediction]
                                else:
                                    dbls[prediction-1]=1
                            correct.append(prediction)
                            pose_counter += 1
                            previous_pose = pose_name
                            
                        else:
                            
                            if prediction==2 or prediction==11:
                                xtras[0]=rss[1]
                            if prediction==6:
                                xtras[1]=rss[1]

        frame_count += 1  
        
    cap.release()
    
    wrong=[]
    for i in range(1,13):
        if i not in correct:
            wrong.append(i)
            
    return wrong,xtras[0],xtras[1]
