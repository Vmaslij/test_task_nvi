from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
from PIL import Image
from torchvision.io import read_image
from torchvision.models import vit_h_14, ViT_H_14_Weights

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


def category_image(filepath):
    img = read_image(filepath)

    weights = ViT_H_14_Weights.DEFAULT
    model = vit_h_14(weights=weights)
    model.eval()

    preprocess = weights.transforms(antialias=True)

    batch = preprocess(img).unsqueeze(0)

    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    os.remove(filepath)
    return category_name


# @app.get("/results/{width}/{height}/{category}", response_class=HTMLResponse)
# async def test_result(request: Request, width: int, height: int, category: str):
#     return templates.TemplateResponse("app.html", context={"request": request,
#                                                            "header": "Добро пожаловать",
#                                                            "width": width,
#                                                            "height": height,
#                                                            "category": category})


@app.post("/")
async def test_form(images: UploadFile):
    with open('static/' + images.filename, "wb") as wf:
        shutil.copyfileobj(images.file, wf)
        images.file.close()
    im = Image.open('static/' + images.filename)
    (width, height) = im.size
    im.close()
    category = category_image('static/' + images.filename)
    return width, height, category


# @app.post("/send")
# async def classify(request: Request, images: UploadFile):
#     # contents = await file.read()
#     print("test changes")
#     print(images)
#     return 200
#     # print(arg2.decode('utf-8'))
#     # return templates.TemplateResponse("app.html", context={"request": request, "header": "Добро пожаловать"})


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("app.html", context={"request": request,
                                                           "header": "Классификатор изображений",
                                                           "width": "",
                                                           "height": "",
                                                           "category": ""})
