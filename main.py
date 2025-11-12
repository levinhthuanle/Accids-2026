#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCRIPT: Extract objects, depth, and spatial relations to a TTL Knowledge Graph.
----------------------------------------------------------------
Requires: pip install ultralytics shapely opencv-python transformers torch torchvision rdflib
Usage: python your_script_name.py "path/to/your/image.jpg"
"""
import sys
import itertools
import cv2
import numpy as np
import os
from ultralytics import YOLO
from shapely.geometry import box
import torch
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from PIL import Image
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS

# --- SPATIAL LOGIC ---

def get_shapely_box(bbox_xyxy):
    """Converts [x1, y1, x2, y2] to a shapely.box object."""
    x1, y1, x2, y2 = bbox_xyxy
    return box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

def get_spatial_relationships(obj1, obj2, iou_threshold=0.1):
    """
    Calculates spatial relationships with a priority: 
    Containment -> Overlap -> Directional.
    """
    relations = []
    shape1 = get_shapely_box(obj1['bbox_xyxy'])
    shape2 = get_shapely_box(obj2['bbox_xyxy'])
    
    if not shape1.is_valid or not shape2.is_valid:
        return relations
    
    # 1. Containment
    if shape1.contains(shape2):
        relations.append('contains')
        return relations
    if shape1.within(shape2):
        relations.append('is_inside')
        return relations
    
    # 2. Overlap / Touching
    if shape1.intersects(shape2):
        try:
            intersection_area = shape1.intersection(shape2).area
            union_area = shape1.union(shape2).area
            iou = 0 if union_area == 0 else intersection_area / union_area
            
            if iou > iou_threshold:
                relations.append('is_overlapping')
            else:
                relations.append('is_touching')
        except Exception:
            pass # Ignore geometric errors
    
    # 3. Directional (if disjoint)
    if not relations:
        cx1, cy1 = obj1['centroid']
        cx2, cy2 = obj2['centroid']
        
        dx = cx2 - cx1
        dy = cy2 - cy1
        
        width1 = shape1.bounds[2] - shape1.bounds[0]
        height1 = shape1.bounds[3] - shape1.bounds[1]
        width2 = shape2.bounds[2] - shape2.bounds[0]
        height2 = shape2.bounds[3] - shape2.bounds[1]
        avg_width = (width1 + width2) / 2
        avg_height = (height1 + height2) / 2
        
        # Use dynamic thresholds based on object size
        if abs(dx) > avg_width * 0.25:
            if dx > 0:
                relations.append('is_left_of') # obj1 is left of obj2
            else:
                relations.append('is_right_of') # obj1 is right of obj2
        
        if abs(dy) > avg_height * 0.25:
            if dy > 0:
                relations.append('is_above') # obj1 is above obj2
            else:
                relations.append('is_below') # obj1 is below obj2
    
    return relations

# --- DEPTH ESTIMATION ---

def load_depth_model(model_size='base'):
    """Loads the Depth Anything v2 model."""
    model_repo = {
        'small': 'depth-anything/Depth-Anything-V2-Small-hf',
        'base': 'depth-anything/Depth-Anything-V2-Base-hf',
        'large': 'depth-anything/Depth-Anything-V2-Large-hf'
    }
    if model_size not in model_repo:
        raise ValueError("Invalid model size. Choose 'small', 'base', or 'large'.")
    
    print(f"[INFO] Loading Depth Anything v2 model ({model_size})...")
    try:
        # Use fast processor to suppress warnings
        image_processor = AutoImageProcessor.from_pretrained(model_repo[model_size], use_fast=True)
        model = AutoModelForDepthEstimation.from_pretrained(model_repo[model_size])
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        return model, image_processor
    except Exception as e:
        print(f"Failed to load Depth Anything v2 model. Error: {e}")
        return None, None

def estimate_depth(image_path, model, image_processor):
    """Generates a depth map for the given image."""
    image = Image.open(image_path).convert('RGB')
    inputs = image_processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Interpolate to original image size
    depth_map = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(image.height, image.width),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    
    return depth_map

def get_object_depth(depth_map, bbox_xyxy):
    """Calculates the average depth for a bounding box region."""
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    object_depth_region = depth_map[y1:y2, x1:x2]
    if object_depth_region.size == 0:
        return 0.0
    return np.mean(object_depth_region)

# --- UTILITIES ---

def resize_to_width(image, width):
    """Resizes an image to a specific width, maintaining aspect ratio."""
    try:
        (h, w) = image.shape[:2]
        if w == 0: return image # Avoid division by zero
        r = width / float(w)
        dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"[WARN] Could not resize image: {e}")
        return image

# --- TTL/RDF EXPORT ---

def save_as_ttl(kg_dict, output_filename="scene_graph.ttl"):
    """Converts the result dictionary to a Turtle (TTL) RDF file."""
    print(f"[INFO] Generating TTL file: {output_filename}")
    
    # 1. Define Namespaces
    SCENE = Namespace("http://example.org/scene/")
    OBJ = Namespace("http://example.org/object/")
    VOCAB = Namespace("http://example.org/vocabulary#")

    # 2. Initialize Graph
    g = Graph()
    g.bind("scene", SCENE)
    g.bind("obj", OBJ)
    g.bind("vocab", VOCAB)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)

    # 3. Create the main Image node
    image_filename = os.path.basename(kg_dict['image_path'])
    image_uri = SCENE[image_filename]
    g.add((image_uri, RDF.type, VOCAB.Image))
    g.add((image_uri, RDFS.label, Literal(f"Scene graph for {image_filename}")))
    g.add((image_uri, VOCAB.filePath, Literal(kg_dict['image_path'])))

    # 4. Add all detected objects as nodes with properties
    for obj in kg_dict['objects']:
        obj_uri = OBJ[str(obj['id'])]
        class_uri = VOCAB[obj['class_name'].replace(" ", "_").capitalize()]
        
        # obj:0 rdf:type vocab:Dog
        g.add((obj_uri, RDF.type, class_uri))
        # obj:0 rdfs:label "dog"
        g.add((obj_uri, RDFS.label, Literal(obj['class_name'])))
        # obj:0 vocab:confidence 0.95
        g.add((obj_uri, VOCAB.confidence, Literal(obj['confidence'])))
        # obj:0 vocab:depth 15.2
        g.add((obj_uri, VOCAB.depth, Literal(obj['depth'])))
        # obj:0 vocab:bbox "[100, 150, 200, 250]"
        g.add((obj_uri, VOCAB.bbox, Literal(str(obj['bbox_xyxy']))))
        # scene:my_image.jpg vocab:containsObject obj:0
        g.add((image_uri, VOCAB.containsObject, obj_uri))

    # 5. Add all relationships as predicates
    for rel in kg_dict['relationships']:
        subj_uri = OBJ[str(rel['subject_id'])]
        obj_uri = OBJ[str(rel['object_id'])]
        predicate_uri = VOCAB[rel['predicate']]
        
        # obj:0 vocab:is_left_of obj:1
        g.add((subj_uri, predicate_uri, obj_uri))

    # 6. Serialize and save the file
    try:
        ttl_data = g.serialize(format="turtle")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(ttl_data)
        print(f"[INFO] Knowledge graph saved to: {output_filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save TTL file. Error: {e}")


# --- MAIN EXECUTION ---

def extract_objects_and_relations(image_path, 
                                yolo_model_name='yolo11x.pt', 
                                depth_model_size='base', 
                                conf_threshold=0.5, 
                                display_width=720):
    """
    Main function: Runs YOLO, Depth Anything, and spatial logic.
    """
    print(f"[INFO] Loading YOLO model: {yolo_model_name}...")
    try:
        yolo_model = YOLO(yolo_model_name)
    except Exception as e:
        print(f"Failed to load YOLO model {yolo_model_name}. Error: {e}")
        return None
    
    depth_model, image_processor = load_depth_model(depth_model_size)
    if depth_model is None:
        return None
    
    print(f"[INFO] Analyzing image: {image_path}")
    yolo_results = yolo_model(image_path, conf=conf_threshold)
    
    depth_map = estimate_depth(image_path, depth_model, image_processor)
    
    # Process YOLO results
    detected_objects = []
    result = yolo_results[0]
    names = result.names
    boxes = result.boxes
    
    print(f"[INFO] Detected {len(boxes)} objects.")
    for i in range(len(boxes)):
        box = boxes[i]
        xyxy = box.xyxy.tolist()[0]
        confidence = box.conf.tolist()[0]
        class_id = int(box.cls.tolist()[0])
        class_name = names[class_id]
        
        centroid_x = (xyxy[0] + xyxy[2]) / 2
        centroid_y = (xyxy[1] + xyxy[3]) / 2
        
        depth = get_object_depth(depth_map, xyxy)
        
        detected_objects.append({
            'id': i,
            'class_name': class_name,
            'confidence': confidence,
            'bbox_xyxy': xyxy,
            'centroid': (centroid_x, centroid_y),
            'depth': depth
        })
    
    # --- Display Results ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image for display: {image_path}")
    else:
        colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in detected_objects]
        
        for obj, color in zip(detected_objects, colors):
            x1, y1, x2, y2 = map(int, obj['bbox_xyxy'])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{obj['class_name']} ({obj['id']}) Depth: {obj['depth']:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        img_display = resize_to_width(img, display_width)
        cv2.imshow('Detected Objects', img_display)
    
    if 'depth_map' in locals():
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        depth_display = resize_to_width(depth_colored, display_width) 
        cv2.imshow('Depth Map', depth_display)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # --- Relationship Calculation ---
    print("[INFO] Calculating pairwise spatial relationships...")
    relationships = []
    
    for obj1, obj2 in itertools.combinations(detected_objects, 2):
        # Check relations in both directions
        relations_1_to_2 = get_spatial_relationships(obj1, obj2)
        for pred in relations_1_to_2:
            relationships.append({
                'subject_id': obj1['id'],
                'predicate': pred,
                'object_id': obj2['id']
            })
        
        relations_2_to_1 = get_spatial_relationships(obj2, obj1)
        for pred in relations_2_to_1:
            relationships.append({
                'subject_id': obj2['id'],
                'predicate': pred,
                'object_id': obj1['id']
            })
    
    output_dict = {
        'image_path': image_path,
        'objects': detected_objects,
        'relationships': relationships
    }
    
    print(f"[INFO] Processing complete. Found {len(relationships)} relationships.")
    return output_dict

# --- SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Error: No image path provided.")
        print("Usage: python your_script_name.py \"path/to/your/image.jpg\"")
        sys.exit(1)
        
    image_file = sys.argv[1]
    
    if not os.path.exists(image_file):
        print(f"Error: File not found at: {image_file}")
        sys.exit(1)

    # Run the main pipeline
    kg_dict = extract_objects_and_relations(image_file, yolo_model_name='yolo11x.pt')
    
    if kg_dict:
        # Print summary to console
        print("\n--- SUMMARY ---")
        print(f"Image: {kg_dict['image_path']}")
        
        print("\nObjects:")
        for obj in kg_dict['objects']:
            print(f" - ID {obj['id']}: {obj['class_name']} (Conf: {obj['confidence']:.2f}, Depth: {obj['depth']:.2f})")
        
        print("\nRelationships (first 15 examples):")
        obj_map = {obj['id']: obj['class_name'] for obj in kg_dict['objects']}
        for rel in kg_dict['relationships'][:15]:
            try:
                subj_name = obj_map[rel['subject_id']]
                obj_name = obj_map[rel['object_id']]
                print(f" - ({subj_name} [ID {rel['subject_id']}]) -> {rel['predicate']} -> ({obj_name} [ID {rel['object_id']}])")
            except KeyError:
                pass
        
        # Save the final TTL file
        output_ttl_file = "scene_graph_output.ttl"
        save_as_ttl(kg_dict, output_ttl_file)