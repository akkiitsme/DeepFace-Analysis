from deepface import DeepFace

def duplicateDetection():
    # Load and preprocess the two images
    image1_path = "images/image.jpg"
    image2_path = "images/image_3.jpg"
    #calling VGGFace
    #model_name = "VGG-Face"
    #model = DeepFace.build_model(model_name) 

    # Verify if the faces match
    result = DeepFace.verify(image1_path, image2_path, model_name="Facenet")
    #result = DeepFace.verify(image1_path, image2_path)
    #description of image characters
    emotion_result = DeepFace.analyze(image2_path, actions = ['age', 'gender', 'race', 'emotion'])
   # df = DeepFace.find(image2_path, db_path ='/images/') 
    for obj in emotion_result:
        print(obj["age"]," years old ",obj["dominant_race"]," ",obj["dominant_emotion"]," ", obj["gender"])
        print(obj["dominant_gender"])

    #print(df.head()) 

    print(emotion_result)
    

    # Get the similarity score
    #similarity_score = result["distance"]

    #print(" Similarity Score : ",similarity_score)
    #similarity_threshold = 0.7

    # Compare the similarity score with the threshold
    #if similarity_score <= similarity_threshold:
      #  print("The two images contain the same face.")
    #else:
      #  print("The two images do not contain the same face.")

    #print(result)

    # Print the result
#    if result["verified"]:
#        print("The two images contain the same face.")
#    else:
#        print("The two images do not contain the same face.")

    return result    


