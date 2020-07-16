import os
import sys
import torch

sys.path.append("/usr/src/app/kaggle/panda-challenge")


TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 8
EPOCHS = 30
NUM_FOLDS = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATIENCE = 3
LR = 1e-4


INPUT_DIR = 'inputs'
OUT_DIR = 'models'
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
TRAIN_IMG_PATH = INPUT_DIR + '/train_images/train_images/'
TEST_IMG_PATH = INPUT_DIR + '/test_images/'
TRAIN_PROCESSED_IMG_PATH = INPUT_DIR + '/processed/'
SAMPLE_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

FOLD0_ONLY = False
DEBUG = False
