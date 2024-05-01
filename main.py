import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
from skimage import io
from torchvision.transforms import transforms
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import cv2
import numpy as np
from skimage.transform import resize
warnings.filterwarnings("ignore", category=UserWarning) 

classes = ['battery', 'cardboard', 'clothes', 'glass', 'human', 'metal', 'organic', 'paper', 'plastic', 'shoes', 'styrofoam']
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)

        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)

        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_loss = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()

        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, time: {:.4f}".format(epoch+1, result['train_loss'], result['val_loss'], result['val_acc'], result['time']))

class ResNet(ImageClassificationBase):
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

device = torch.device("cpu")

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

model = to_device(ResNet(), device)

newmodel = torch.load("sampah.pth", map_location='cpu')

def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
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
    
    im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
    to = transforms.ToTensor()
    im = to(im)
    plt.imshow(im.permute(1, 2, 0))
    
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
    app.run()