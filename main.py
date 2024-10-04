from tools.training_app import *



def main():
    app = Evaluation(Models.AGE_MODEL)
    app.main(app)

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
