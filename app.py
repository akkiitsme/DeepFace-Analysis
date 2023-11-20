from deepface import DeepFace
from flask import Flask, jsonify, request 
from duplicateDetection import duplicateDetection


#API Generation 
app = Flask(__name__)

@app.route('/api/imagedetection')
def objectResult():
    result = duplicateDetection()

    print(result)

    if result is not None:
        similarity_score = float(result["distance"])
        status = bool(result["verified"])
        description = ""
        if status:
            description = "These images contain the same face."
        else:
            description = "These images do not contain the same face."

        finalResult = {
            "message":"success",
            "face match status": status,
            "face similarity distance score": similarity_score,
            "Description": description,
            "status": True
        }
    return jsonify(finalResult)



if __name__ == "__main__":
    app.run(debug=False)



