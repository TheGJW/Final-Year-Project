from fastapi import FastAPI
from services.retrieval_service import semantic_search, get_most_common
from services.crop_service import crop_box
from services.spatial_service import infer_relation
from services.color_service import segment_object, detect_dominant_color,lab_to_color_name
from models.embedding_model import get_embedding
from supabase import create_client
from fastapi import File, UploadFile
import shutil
from models.object_detector import ObjectDetector
from models.scene_classifier import SceneClassifier
from fastapi.middleware.cors import CORSMiddleware
import base64
import numpy as np
from pydantic import BaseModel
from typing import List, Dict

detector = ObjectDetector()
scene_model = SceneClassifier()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

url = "xxx"
key = "xxx"

# initializing supabase client
supabase = create_client(url, key)

# requesting model for frontend -> backend communication
class SelectionRequest(BaseModel):
    top_box: Dict
    wide_box: Dict
    wide_boxes: List[Dict]
    user_object_name: str

@app.post("/process-selection")
async def process_selection(data: SelectionRequest):
    # extracting data from request
    top_box = data.top_box
    wide_box = data.wide_box
    wide_boxes = data.wide_boxes
    user_object_name = data.user_object_name
    # top image pipeline

    # crop selected object
    cropped = crop_box("temp_top.jpeg", top_box)
    cropped.save("selected_object.jpg")

    # segmentation + color
    segmented = segment_object("selected_object.jpg")
    color_lab = detect_dominant_color(segmented)
    color_name = lab_to_color_name(color_lab)

    # embedding
    query_embedding = get_embedding("selected_object.jpg")

    # vector search
    result = supabase.rpc(
        "match_embedding",
        {"query_embedding": query_embedding.tolist()}
    ).execute()

    match_found = False
    object_id = None

    EMBED_THRESHOLD = 0.5
    COLOR_THRESHOLD = 60

    if result.data and len(result.data) > 0:
        # getting the closest match 
        closest = result.data[0]
        distance = closest["distance"]
        # checking if embedding similarity is within threshold
        if distance <= EMBED_THRESHOLD:

            object_id = closest["id"]

            # fetch stored color
            stored = supabase.table("objects").select(
                "color_l,color_a,color_b"
            ).eq("id", object_id).execute()

            if stored.data and len(stored.data) > 0:

                stored_color = stored.data[0]
                # converting stored labs into numpy array
                stored_lab = np.array([
                    stored_color["color_l"],
                    stored_color["color_a"],
                    stored_color["color_b"]
                ])
                # current detected color
                current_lab = np.array(color_lab)
                # compute euclidean distance
                color_distance = np.linalg.norm(stored_lab - current_lab)

                if color_distance <= COLOR_THRESHOLD:
                    match_found = True

    # wide image pipeline
    # infering spacial relationship between selected object and other objects
    relation = infer_relation(
        user_object_name,
        wide_box,
        wide_boxes,   
        label_conf_threshold=0.3
    )
    # peforming scene classfication on wide image
    scene, confidence = scene_model.classify(
        "temp_wide.jpeg",
        threshold=0.45
    )

    if scene != "unknown_scene":
        description = f"{color_name} {relation} at {scene}"
    else:
        description = f"{color_name} {relation}"

    # database 
    # if object already exist, only insert observations
    if match_found:

        supabase.table("observations").insert({
            "object_id": object_id,
            "description": description
        }).execute()

    else:
        # insert new object with embedding and color features and the observation
        insert_obj = supabase.table("objects").insert({
            "object_name": user_object_name,
            "color_l": float(color_lab[0]),
            "color_a": float(color_lab[1]),
            "color_b": float(color_lab[2]),
            "embedding": query_embedding.tolist()
        }).execute()

        object_id = insert_obj.data[0]["id"]

        supabase.table("observations").insert({
            "object_id": object_id,
            "description": description
        }).execute()

    

    return {
        "object_id": object_id,
        "match_found": match_found,
        "color": color_name,
        "relation": relation,
        "scene": scene,
        "description": description
    }



# to verify the API is running
@app.get("/")
def root():
    return {"message": "API is working"}



@app.post("/detect-both")
async def detect_both(
    top_image: UploadFile = File(...),
    wide_image: UploadFile = File(...)
):

    # save uploaded top image and wide image
    with open("temp_top.jpeg", "wb") as buffer:
        shutil.copyfileobj(top_image.file, buffer)

    with open("temp_wide.jpeg", "wb") as buffer:
        shutil.copyfileobj(wide_image.file, buffer)

    # run object detection models on both images
    top_boxes = detector.detect("temp_top.jpeg", conf_threshold=0.1)
    wide_boxes = detector.detect("temp_wide.jpeg", conf_threshold=0.2)

    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # return the coordinates of the detected boxes and the initial image
    return {
        "top": {
            "boxes": top_boxes,
            "image": encode_image("temp_top.jpeg")
        },
        "wide": {
            "boxes": wide_boxes,
            "image": encode_image("temp_wide.jpeg")
        }
    }
# semantic search endpoint using text query
@app.get("/search")
def search(query: str):
    results = semantic_search(supabase, query)
    return results

# endpoint to retrieve the most common location of the object in question
@app.get("/search-common")
def search_common(query: str):
    result = get_most_common(supabase, query)
    return result