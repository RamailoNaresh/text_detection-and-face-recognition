import json
from django.http import JsonResponse
from django.shortcuts import render
import face_recognition
# Create your views here.
from google.cloud import vision
from rest_framework.decorators import api_view
import os
from wand.image import Image
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"secretVisionApiKey.json"


@api_view(["GET"])
def text_detection_google_cloud_vision(request):
    client = vision.ImageAnnotatorClient()

    image_path = "/home/mango/Downloads/rajesh.jpeg"
    with open(image_path, "rb") as image_file:
        image_binary = image_file.read()
    image = vision.Image(content=image_binary)
    label_detection_feature = vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)
    request = vision.AnnotateImageRequest(image=image, features=[label_detection_feature])
    response = client.annotate_image(request=request)
    text_annotations = response.text_annotations
    return JsonResponse({"text": text_annotations[0].description})


@api_view(["POST"])
def document_verification(request):
    try:
        selfie_image = request.FILES.get("selfie_image")
        personal_card_image = request.FILES.get("personal_card_image")
        document_name = request.POST["document_name"]
        if selfie_image and personal_card_image and document_name:
            # client = vision.ImageAnnotatorClient()
            # # comparing document type from photo and text
            # image = vision.Image(content = personal_card_image.read())
            # label_detection_feature = vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)
            # request = vision.AnnotateImageRequest(image=image, features=[label_detection_feature])
            # response = client.annotate_image(request=request)
            # text_annotate = response.text_annotations
            # text = text_annotate[0].description
            # if document_name not in text:
            #     return JsonResponse({"message": "Document type doesn't matched", "is_error": "False"})
            # comparing faces in an image
            known_image = face_recognition.load_image_file(selfie_image)
            unknown_image = face_recognition.load_image_file(personal_card_image)
            # checking face location
            known_image_face_location = face_recognition.face_locations(known_image)
            unknown_image_face_location = face_recognition.face_locations(unknown_image)
            if not known_image_face_location or not unknown_image_face_location:
                return JsonResponse({"message": "Please upload the document with face", "is_error": "True"})
            if (len(face_recognition.face_encodings(known_image)) != 1):
                return JsonResponse({"messsage": "Please upload image having only one face", "is_error": "True"})
            known_image_encoding = face_recognition.face_encodings(known_image)[0]
            unknown_image_encoding = face_recognition.face_encodings(unknown_image)[0]
            results = face_recognition.compare_faces([known_image_encoding], unknown_image_encoding)
            probability = 1-face_recognition.face_distance([known_image_encoding], unknown_image_encoding)

            if results[0] == True:
                return JsonResponse(
                    {"message": "Face matched", "is_matched": str(results[0]),
                     "matched_probability": probability[0],
                     "is_error": "False"})
            return JsonResponse(
                {"message": "Face didn't matched", "is_matched": str(results[0]),
                 "matched_probability": probability[0],
                 "is_error": "True"})
        return JsonResponse({"message": "Fields cannot be empty", "is_error": "True"})
    except Exception as e:
        return JsonResponse({"message": str(e), "is_error": "True"})


def distort_image(data):
    with Image(filename=data) as img:
        img.virtual_pixel = 'background'
        args = (
            10, 10, 15, 15,  # Point 1: (10, 10) => (15,  15)
            139, 0, 100, 20,  # Point 2: (139, 0) => (100, 20)
            0, 92, 50, 80    # Point 3: (0,  92) => (50,  80)
        )
        img.distort('affine', args)
        img.save(filename='gogdistort.png')
