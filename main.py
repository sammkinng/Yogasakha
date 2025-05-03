from constants import CORRECTIONS,POSE_LABELS
from utils.video_utils import get_estimated_time
from scripts.predict import process_video
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from time import sleep
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
from os import path,remove

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)



async def get_pose_feedback(pose, mistake=None):
    mm={12:1,11:2,10:3,9:4}
    npose=0
    if pose>8:
        npose=mm[pose]
    else:
        npose=pose
        
    yield f"Pose : {POSE_LABELS[npose]}\n"
    if mistake:
        details = CORRECTIONS[pose-1][mistake-1]
        yield f"Mistake: {details['mistake']}\n"
        yield f"Correction: {details['correction']}\n"
        yield f"Keypoints: {details['keypoints']}\n"
        yield f"Way to do: {details['way_to_do']}\n\n"
    else:
        details = CORRECTIONS[pose-1]
        yield f"Mistake: {details['mistake']}\n"
        yield f"Correction: {details['correction']}\n"
        yield f"Keypoints: {details['keypoints']}\n"
        yield f"Way to do: {details['way_to_do']}\n\n"

async def predict(video_path: str):
    try:
        tm=get_estimated_time(video_path)
        
        yield f"{tm}"
        sleep(0.5)
        yield f'Processing Video... Please wait for {int(tm)} seconds\n'
        yield "\n"
        op = process_video(video_path)
        for p in op[0]:
            if p == 6:
                async for feedback in get_pose_feedback(6, op[2]):
                    yield feedback
            elif p == 2 or p == 11:
                async for feedback in get_pose_feedback(2, op[1]):
                    yield feedback
            else:
                async for feedback in get_pose_feedback(p):
                    yield feedback
        
    finally:
        if path.exists(video_path):
            remove(video_path)


@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    file_path = f"temp/{uuid4()}_{video.filename}"
    with open(file_path, "wb") as f:
        f.write(await video.read())
    
    return StreamingResponse(predict(file_path), media_type="text/event-stream")
