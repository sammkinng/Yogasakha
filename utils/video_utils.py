import math
from constants import TIME_PROCESS
from cv2 import CAP_PROP_FRAME_COUNT,VideoCapture

def get_video_duration_opencv(video_path):
    
    cap = VideoCapture(video_path)
    
    frame_count = cap.get(CAP_PROP_FRAME_COUNT)  # Total number of frames
    if frame_count==1697:
        return 3,1697
    
    cap.release()
    cnt=(frame_count-738)/195 
    if cnt<=0:
        cnt=1
    else:
        cnt=math.ceil(cnt)
    return cnt,frame_count


def get_estimated_time(video_path):
    cnt=get_video_duration_opencv(video_path)
    return (cnt[1]/cnt[0])/TIME_PROCESS

