
import io
import uvicorn
from fastapi import FastAPI, File,HTTPException
from starlette.responses import Response
from get_segment import get_segments, get_segmentor
from get_classification import get_class
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from get_binary_class import get_binary
from fastapi.param_functions import Depends
from features import Features,is_valid



app = FastAPI(
    title="SkinLesion API",
    description="APIs for Segmentation, Multiclass Classification & Binary Classification using deep learning models",
    version="0.1.0",
)



@app.post("/segmentation")
def get_segmentation(image: bytes = File(...)):
    model = get_segmentor()
    segmented_image = get_segments(model, image)
    print('Segment image', segmented_image.size)
    byte_io = io.BytesIO()
    segmented_image.save(byte_io, format='PNG')
    return Response(byte_io.getvalue(), media_type='image/png')

    
@app.post("/classification")
def get_classification(image:bytes = File(...)):
    prediction = get_class(image)
    prediction_json = jsonable_encoder(prediction)
    return JSONResponse(prediction_json) 

@app.post("/binary_classification")
def get_binary_class(image:bytes = File(...),form_data: Features = Depends()):
    if is_valid(form_data.anatom_site):
        prediction = get_binary(image, form_data)
        prediction_json = {"prediction":prediction}
        return JSONResponse(prediction_json) 
    raise HTTPException(status_code=404, detail="""anatom_site should be in head/neck ,upper extremity, lower extremity, torso ,palms/soles,oral/genital""")
