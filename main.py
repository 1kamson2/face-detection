from tools.dset import FaceDataset
import matplotlib.pyplot as plt
from tools.training_app import *


def init_training(epochs: int):
    app = TrainingApp('age')
    for epoch in range(epochs):
        app.training_loop()

    app.save_model(app)


def init_prediction(idx: int):
    dset = FaceImages()
    model = AgeModel()
    model.load_state_dict(torch.load("../FaceDetection/rsrc/model_weights.pth"))
    plt.imshow(dset[idx]['data'].permute(1, 2, 0), cmap='gray')
    model.eval()
    with torch.no_grad():
        print(model(dset[idx]['data'].unsqueeze(0)))


def main():
    init_prediction(0)

if __name__ == '__main__':
    main()

#todo:
# replace all python arrays with numpy arrays - they are faster
# check whether the model overfits (chpt 12)
# check false positives, etc. (chpt 12)

