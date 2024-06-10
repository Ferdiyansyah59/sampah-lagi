import torch
import torchvision.models as models
import torch.nn as nn
import warnings
from skimage import io
from torchvision.transforms import transforms
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
# import cv2
import numpy as np
from skimage.transform import resize
import os
warnings.filterwarnings("ignore", category=UserWarning) 
classes = ['battery', 'cardboard', 'clothes', 'glass', 'human', 'metal', 'organic', 'paper', 'plastic', 'shoes', 'styrofoam']

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(classes))

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

model = ResNet()

newmodel = torch.load("sampah.pth", map_location='cpu')



def predict_image(img, model):
    xb = torch.unsqueeze(img, 0)
    yb = model(xb)
    prob, preds = torch.max(yb, dim=1)

    return classes[preds[0].item()]

def pre_pre(img_param):
    im = io.imread(img_param)
    if(im.shape[1] > 2000):
        im = resize(im, (im.shape[0] // 8, im.shape[1] // 8),
                       anti_aliasing=True)
        im = im.astype(np.float32)
    elif(im.shape[1] > 1000):
        im = resize(im, (im.shape[0] // 5, im.shape[1] // 5),
                       anti_aliasing=True)
        im = im.astype(np.float32)
    elif(im.shape[1] > 600):
        im = resize(im, (im.shape[0] // 3, im.shape[1] // 3),
                       anti_aliasing=True)
        im = im.astype(np.float32)
    
    # im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
    to = transforms.ToTensor()
    im = to(im)
    # plt.imshow(im.permute(1, 2, 0))
    
    return predict_image(im, newmodel)


app = Flask(__name__)
CORS(app)

@app.route('/sampah', methods=['POST'])
def sampah():
#     img_param = request.args.get('img')
    img_param = request.form['img']
    res = pre_pre(img_param)
    print("Typenya adalah ", type(img_param))
    try:
        data = {
            'code': 200,
            'message': 'Berhasil memprediksi!',
            'prediksi': res
        }
        
        return make_response(jsonify(data)), 200
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    
    

@app.route('/test', methods=['GET'])
def test():
    return 'Hallow ddd'
    
if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)