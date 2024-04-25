import json
from django.http import JsonResponse
from django.shortcuts import render
import face_recognition
# Create your views here.
from google.cloud import vision
from rest_framework.decorators import api_view
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"secretVisionApiKey.json"


@api_view(["GET"])
def text_detection_google_cloud_vision(request):
    client = vision.ImageAnnotatorClient()

    image_path = "/home/mango/Downloads/rajesh5.jpeg"
    with open(image_path, "rb") as image_file:
        image_binary = image_file.read()
    image = vision.Image(content=image_binary)
    label_detection_feature = vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)
    request = vision.AnnotateImageRequest(image=image, features=[label_detection_feature])
    response = client.annotate_image(request=request)
    text_annotations = response.text_annotations
    return JsonResponse({"text": text_annotations[0].description})


@api_view(["POST"])
def face_compare(request):
    selfie_image = request.FILES.get("selfie_image")
    personal_card_image = request.FILES.get("personal_card_image")
    if selfie_image and personal_card_image:
        known_image = face_recognition.load_image_file(selfie_image)
        unknown_image = face_recognition.load_image_file(personal_card_image)
        known_image_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_image_encoding = face_recognition.face_encodings(unknown_image)[0]
        results = face_recognition.compare_faces([known_image_encoding], unknown_image_encoding)
        probability = 1- face_recognition.face_distance([known_image_encoding], unknown_image_encoding)
        if results[0] == True:
            return JsonResponse({"is_matched": str(results[0]), "matched_probability": probability[0], "is_error": "False"})
        return JsonResponse({"is_matched": str(results[0]), "matched_probability": probability[0], "is_error": "True"})
    return JsonResponse({"is_error": "True"})