from imageai.Detection import ObjectDetection
from collections import OrderedDict

def object_detection(input_image_path, output_image_path):
    try:
        # Object Detection From Image
        detector = ObjectDetection()
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath("./models/yolov3.pt")
        detector.loadModel()

        # Perform object detection
        detections = detector.detectObjectsFromImage(input_image_path, output_image_path)

        # Extract entities and quantities
        my_data = []
        for each_object in detections:
            print(each_object["name"], ":", each_object["percentage_probability"])
            my_data.append(each_object["name"])

        entity_list = list(OrderedDict.fromkeys(my_data))
        quantity_list = {item: my_data.count(item) for item in my_data}

        print("Entities:", entity_list)
        print("Quantities:", quantity_list)

        return entity_list, quantity_list

    except Exception as e:
        print("Error:", str(e))
        return None, None