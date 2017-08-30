#!/bin/sh
python classification_from_array.py
python classification_from_svm.py
python prediction_from_model.py
python train_from_lmdb.py
python classification_resume_training.py