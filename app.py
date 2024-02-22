from flask import Flask, jsonify, request
from deepface import DeepFace
from PIL import Image
from io import BytesIO
import base64,logging,json
from objectdetection import object_detection
from flask_cors import CORS

logging.basicConfig(filename='access.log',level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    result={
        "message" : "success",
        "status": False ,
        "desc": "UNAUTHORIZED ACCESS" 
    }
    return jsonify(result),401

@app.route("/welcome")
def welcome():
    result={
        "message" : "success",
        "status": True ,
        "desc": "Welcome to Python Development" 
    }
    return jsonify(result)

@app.route("/api/facedetection",methods = ['POST','GET'])
def face_detect():
    try:
        if request.method == 'GET':
            return jsonify({"status":False,"message":"success","desc":"This method is not allowed"}),404
        
        if not 'image' in request.form and not 'userId' in request.form:
            return jsonify({"status":False,"message":"success","desc":"some parameters are missing..."}),400
        
        image_byte_code = request.form['image']
        userId = request.form['userId']

        if image_byte_code is not None and userId is not None:
            #writing the image
            image_stream = BytesIO(base64.b64decode(image_byte_code))
            image = Image.open(image_stream)
            image = image.convert("RGB")
            image.save("./images/"+userId+".jpg")

            #Getting information from the image
            #['age', 'gender', 'race', 'emotion']
            emotion_result = DeepFace.analyze(img_path = "./images/"+userId+".jpg", actions = ['age', 'gender', 'race', 'emotion'])
            duplicate_result = DeepFace.find(img_path = "./images/"+userId+".jpg", db_path = "./images")
            #object-detection
            entityList,quantityList=object_detection("./images/"+userId+".jpg","./images/"+userId+"_object.jpg")

            print("Object detection: ",entityList,quantityList)
            logging.info(f"User {userId} has been detected with objects {entityList}")

            
            object_identities = []
            for output in duplicate_result:
                for item in output["identity"]:
                    print("item: ",item)
                    object_identities.append(item) 
            object_identities = [item.replace('./images/', '') for item in object_identities]

            for obj in emotion_result:
                data = {'age':obj["age"],'race':obj["dominant_race"],'emotion':obj["dominant_emotion"],'gender':obj["dominant_gender"],"objects":entityList,"objectTypes":quantityList,"duplicacy":object_identities}
                print(obj["age"]," years old ",obj["dominant_race"]," ",obj["dominant_emotion"]," ", obj["dominant_gender"])

            result= {
                "message" : "success",
                "status": True ,
                "desc": "face detection under process...", 
                "result" : data
            }
        else : 
            result = {
                "message" : "success",
                "status": False ,
                "desc": "userid or image is missing!!"
            }
        return jsonify(result)
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({"status": False, "message": "error", "desc": f"An error occurred: {str(e)}"}), 500



@app.route("/api/multiface")
def duplicate_detect():
    if not 'image' in request.form and not 'userId' in request.form:
        return jsonify({"status":False,"message":"success","desc":"some parameters are missing..."}),400
    
    image_byte_code = request.form['image']
    userId = request.form['userId']

    if image_byte_code is not None and userId is not None:
        #writing the image
        image_stream = BytesIO(base64.b64decode(image_byte_code))
        image = Image.open(image_stream)
        image.save("./images/"+userId+".jpg")

        #Getting information from the image
        duplicate_result = DeepFace.find(img_path = "./images/"+userId+".jpg", db_path = "./images")
        object_identities = []
        for output in duplicate_result:
            for item in output["identity"]:
                print("item: ",item)
                object_identities.append(item) 
        object_identities = [item.replace('./images/', '') for item in object_identities]
        result= {
            "message" : "success",
            "status": True ,
            "desc": "face detection under process...", 
           "result" : object_identities
        }
    else : 
        result = {
            "message" : "success",
            "status": False ,
            "desc": "userid or image is missing!!"
        }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=False,port=8090)