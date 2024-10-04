from tools.dset import FaceDataset
import matplotlib.pyplot as plt
from tools.training_app import *


def init_training(epochs: int):

    app1 = TrainingApp(what_model=Models.AGE_MODEL)
    app1.training_loop(epochs)
    app1.save_model(app1)

    app2 = TrainingApp(what_model=Models.ETHNICITY_MODEL)
    app2.training_loop(epochs)
    app2.save_model(app2)

    # app3 = TrainingApp(what_model=Models.GENDER_MODEL)
    # app3.training_loop(epochs)
    # app3.save_model(app3)
    # doesn't work in the current loop

def init_prediction():
    dset = FaceImages()
    _set = FaceDataset()
    model = AgeModel()
    model.load_state_dict(torch.load("../FaceDetection/rsrc/Models.AGE_MODEL_weights.pth"))
    model.eval()
    with torch.no_grad():
        for idx in range(len(dset)):
            print(f"Prediction of {dset[idx]['image']}, {model(dset[idx]['data'].unsqueeze(0).unsqueeze(0))}")

def init_prediction2():
    dset = FaceImages()
    _set = FaceDataset()
    model = EthnicityModel()
    model.load_state_dict(torch.load("../FaceDetection/rsrc/Models.ETHNICITY_MODEL_weights.pth"))
    model.eval()
    with torch.no_grad():
        for idx in range(len(dset)):
            print(f"Prediction of {dset[idx]['image']}, {model(dset[idx]['data'].unsqueeze(0).unsqueeze(0))}")



def main():
    init_prediction()
    init_prediction2()


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
