from taipy.gui import Gui
import requests
from PIL import Image
import numpy as np
import io
import os

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict_image(url):
    data = {'url': url}
    result = requests.post('https://XXX.execute-api.us-east-1.amazonaws.com/test/predict', json=data).json()
    # Ensure all values are of the same type before sorting
    result = {k: float(v) if isinstance(v, str) else sum(v) / len(v) if isinstance(v, list) else v for k, v in result.items()}

    # Now you can sort the dictionary by value
    result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1])}
    values = np.array(list(result.values()))
    softmax_values = softmax(values) * 100 # apply softmax and scale to 0-100
    result = dict(zip(result.keys(), softmax_values)) # create a new dictionary with softmax values
    top_pred = max(result, key=result.get)
    top_prob = round(result[top_pred])
    return top_prob, top_pred

content = ""
img_path = "placeholder_image.png"
abs_img_path = os.path.abspath(img_path)
prob = 0
pred = ""

index = """
<|text-center|
<|{"logo.png"}|image|width=25vw|>

Enter the URL of an image

<|{content}|input|>

<|{pred}|>

<|{abs_img_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>
"""

def on_change(state, var_name, var_val):
    if var_name == "content": 
        response = requests.get(var_val)
        img = Image.open(io.BytesIO(response.content))
        img.save("temp.png")
        abs_img = os.path.abspath("temp.png")
        top_prob, top_pred = predict_image(var_val)
        state.prob = top_prob
        state.pred = "this is a " + top_pred
        state.abs_img_path = abs_img

app = Gui(page=index)

if __name__ == "__main__":
    app.run(use_reloader=True)
