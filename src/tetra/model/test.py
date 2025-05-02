from threading import Thread
from ultralytics import YOLO



def predict(image_path):
    """Predicts objects in an image using a preloaded YOLO model, take path string to image as argument."""
    # Instantiate the model insisde the thread
    shared_model = YOLO("yolo11n.pt")
    results = shared_model.predict(image_path, conf=0.1)
