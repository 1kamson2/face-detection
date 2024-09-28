from tools.dset import FaceDataset
import matplotlib.pyplot as plt
from tools.training_app import *


def init_training(epochs: int):
    app = TrainingApp(what_model=Models.ETHNICITY_MODEL)
    for epoch in range(epochs):
        app.training_loop()

    app.save_model(app)


def init_prediction():
    dset = FaceImages()
    _set = FaceDataset()
    model = AgeModel()
    model.load_state_dict(torch.load("../FaceDetection/rsrc/age_weights.pth"))
    model.eval()
    with torch.no_grad():
        for idx in range(len(dset)):
            print(f"Prediction of {dset[idx]['image']}, {model(dset[idx]['data'].unsqueeze(0).unsqueeze(0))}")

def init_prediction2():
    dset = FaceImages()
    _set = FaceDataset()
    model = GenderModel()
    model.load_state_dict(torch.load("../FaceDetection/rsrc/age_weights.pth"))
    model.eval()
    with torch.no_grad():
        for idx in range(len(dset)):
            print(f"Prediction of {dset[idx]['image']}, {model(dset[idx]['data'].unsqueeze(0).unsqueeze(0))}")



def main():
    init_training(epochs=1)


if __name__ == '__main__':
    main()

#todo:
# replace all python arrays with numpy arrays - they are faster
# check whether the model overfits (chpt 12)
# check false positives, etc. (chpt 12)
# fix bad clipped values - done
# fix data set to get rid off random unsqueezes. - partially done
# predict boxes on faces: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# use the dataset from the other networks
# getting rid of stacking pixels. - done
