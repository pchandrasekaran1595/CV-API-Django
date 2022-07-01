import json

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

from static.utils import CFG, decode_data, decode_image, encode_image_to_base64


@csrf_exempt
def classify(request):
    if request.method == "POST":
        cfg = CFG("classify")
        cfg.setup()

        if request.POST.get("data") is not None:
            imageData  = json.loads(request.POST.get("data"))["imageData"]
            _, image = decode_image(imageData)
        else:
            image = decode_data(request.FILES["image"].read())
        
        label = cfg.infer(image)

        return JsonResponse({
            "label" : label,
        })

    return HttpResponse("Classification Endpoint")


@csrf_exempt
def detect(request):
    if request.method == "POST":
        cfg = CFG("detect")
        cfg.setup()

        if request.POST.get("data") is not None:
            imageData  = json.loads(request.POST.get("data"))["imageData"]
            _, image = decode_image(imageData)
        else:
            image = decode_data(request.FILES["image"].read())

        label, (x1, y1, x2, y2) = cfg.infer(image)

        return JsonResponse({
            "label" : label,
            "x1"    : str(x1),
            "y1"    : str(y1),
            "x2"    : str(x2),
            "y2"    : str(y2),
        })

    return HttpResponse("Detection Endpoint")

@csrf_exempt
def segment(request):
    if request.method == "POST":
        cfg = CFG("segment")
        cfg.setup()

        if request.POST.get("data") is not None:
            imageData  = json.loads(request.POST.get("data"))["imageData"]
            _, image = decode_image(imageData)
        else:
            image = decode_data(request.FILES["image"].read())
        
        segmented_image, labels = cfg.infer(image)
        segmented_image_data = encode_image_to_base64(header="data:image/png;base64", image=segmented_image)

        return JsonResponse({
            "labels" : str(labels),
            "imageData" : segmented_image_data,
        })

    return HttpResponse("Segmentation Endpoint")