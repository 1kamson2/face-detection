from tools.dset import FaceDataset
import matplotlib.pyplot as plt
from tools.training_app import *


def main():
    training_app = TrainingApp('age')
    for epoch in range(20):
        training_app.training_loop()


if __name__ == '__main__':
    main()

#todo:
# replace all python arrays with numpy arrays - they are faster
