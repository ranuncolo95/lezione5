from ultralytics import YOLO
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Form

#model = YOLO("yolov8s")

# ESERCIZIO 1 Scrivere un programma che data un'immagine in input, restituisce quante persone ci sono nell'immagine.
"""
results = model("persone.jpg", show=False)

oggetti_trovati = results[0].boxes.cls.tolist()
persone_trovate = oggetti_trovati.count(0)

#input()

print(f"Ho trovato {persone_trovate} persone nell'immagine.")
"""

# ESERCIZIO 2 Scrivere un programma che dato un video in input, restituisce quante persone ci sono nel quinto frame.
"""
oggetti_trovati = results[4].boxes.cls.tolist()
persone_trovate = oggetti_trovati.count(0)

#input()

print(f"Ho trovato {persone_trovate} persone nel frame 5 del video.")
"""

# ESERCIZIO 3 Scrivere un programma che dato un video in input, restitusce il numero di persone totali identificate in tutti i frames.
"""
persone_trovate = 0

for f in results[0:30]:
    oggetti_trovati = f.boxes.cls.tolist()
    persone_trovate += oggetti_trovati.count(0)

#input()

print(f"Ho trovato {persone_trovate} persone per frame nel video.")
"""
#ESERCIZIO 4 Creare un servizio web, che attraverso un form sia possibile inserire l'id di una categoria 
# (es 0 per 'person') e restituisca quante volte quella categoria è presente in un'immagine (statica che inserite voi)

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )

@app.post("/results", response_class=HTMLResponse)
def results(request: Request, input_text: int = Form(...), image = Form(...)):
    model = YOLO("yolov8s")

    analize_image = model(image, show=False)

    oggetti_trovati = analize_image[0].boxes.cls.tolist()
    result = oggetti_trovati.count(input_text)
    oggetti_da_trovare = analize_image[0].names[input_text]

    return templates.TemplateResponse(
        request=request, 
        name="results.html", 
        context={"oggetto": oggetti_da_trovare, "risposta": result})