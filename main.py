import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize

# Initialize the FastAPI app
app = FastAPI()

# Define dice coefficient and dice loss functions
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# Load the pre-trained model
model_path = 'res_model_for_anemia.h5'
model = tf.keras.models.load_model(model_path, custom_objects={'dice_coef': dice_coef}, safe_mode=False)

@app.get("/")
def read_root():
    return HTMLResponse(open("index.html").read())

def load_image(image: Image.Image, mask=False):
    img_array = img_to_array(image)
    img_array = img_array / 255.0  # Normalize

    # Resize the image to the expected input shape of the model
    img_array = resize(img_array, (256, 256)).numpy()  # Use tf.image.resize

    if mask:
        img_array = (img_array > 0.5).astype(np.float32)  # Binarize mask
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")

    # Preprocess the image
    image_array = load_image(image)

    # Predict
    pred = model.predict(np.expand_dims(image_array, axis=0))[0]
    pred = (pred > 0.5).astype(np.float32)

    # Return the prediction as JSON
    pred_list = pred.tolist()
    return JSONResponse(content={"prediction": pred_list})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
