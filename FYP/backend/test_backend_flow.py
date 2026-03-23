from models.object_detector import ObjectDetector
from services.crop_service import crop_box
from services.spatial_service import infer_relation
from services.draw_service import draw_boxes
from services.color_service import segment_object, detect_dominant_color,lab_to_color_name
from models.embedding_model import get_embedding
from models.scene_classifier import SceneClassifier
from services.retrieval_service import semantic_search

from supabase import create_client
import numpy as np

url = "xxx"
key = "xxx"




supabase = create_client(url, key)

detector = ObjectDetector()
scene_model = SceneClassifier()
# ---- Top image (for user selection) ----
top_boxes = detector.detect('top.jpeg')

print("Top boxes:", len(top_boxes))

selected_index = 0
selected_box = top_boxes[selected_index]

# Crop selected object
cropped = crop_box('top.jpeg', selected_box)
cropped.save("selected_object.jpg")

segmented = segment_object("selected_object.jpg")

color_lab  = detect_dominant_color(segmented)

color_name = lab_to_color_name(color_lab)

print("Detected color:", color_name)


query_embedding = get_embedding("selected_object.jpg")

result = supabase.rpc(
    "match_embedding",
    {"query_embedding": query_embedding.tolist()}
).execute()

# ---------- Check similarity ----------
match_found = False
object_id = None

EMBED_THRESHOLD = 0.3
COLOR_THRESHOLD = 25


if result.data and len(result.data) > 0:

    closest = result.data[0]
    distance = closest["distance"]

    print("Embedding distance:", distance)

    if distance <= EMBED_THRESHOLD:

        object_id = closest["id"]

        # Fetch stored color
        stored = supabase.table("objects").select(
            "color_l,color_a,color_b"
        ).eq("id", object_id).execute()

        if stored.data and len(stored.data) > 0:

            stored_color = stored.data[0]

            stored_lab = np.array([
                stored_color["color_l"],
                stored_color["color_a"],
                stored_color["color_b"]
            ])

            current_lab = np.array(color_lab)

            color_distance = np.linalg.norm(stored_lab - current_lab)

            print("Color distance:", color_distance)

            if color_distance <= COLOR_THRESHOLD:
                match_found = True

# ---- Wide image (for spatial logic) ----
wide_boxes = detector.detect("wide.jpeg", conf_threshold=0.1)

print("Wide boxes:", len(wide_boxes))

selected_index = 1  # simulate user clicking box in wide image
selected_box = wide_boxes[selected_index]

draw_boxes('wide.jpeg', wide_boxes, selected_box)

user_object_name = "wallet"
relation = infer_relation(user_object_name, selected_box, wide_boxes,label_conf_threshold=0.5)

print("Relation:", relation)

scene, confidence = scene_model.classify("wide.jpeg", threshold=0.5)

print("Scene:", scene)

print("Scene confidence:", confidence)

if scene != "unknown_scene":
    description = f"{color_name} {relation} at {scene}"
else:
    description = f"{color_name} {relation}"

# ---------- Database logic ----------
if match_found:

    print("Existing object detected")

    supabase.table("observations").insert({
        "object_id": object_id,
        "description": description
    }).execute()

else:

    print("New object detected")

    insert_obj = supabase.table("objects").insert({
        "object_name": user_object_name,
        "color_l": float(color_lab[0]),
        "color_a": float(color_lab[1]),
        "color_b": float(color_lab[2]),
        "embedding": query_embedding.tolist()
    }).execute()

    new_id = insert_obj.data[0]["id"]

    supabase.table("observations").insert({
        "object_id": new_id,
        "description": description
    }).execute()

results = semantic_search(supabase, "wallet")

print("Search results:", results)
