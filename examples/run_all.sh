#!/bin/sh
python classification.py
python predict_from_model.py
python sklearn_cross_validation.py
python sklearn_grid_search.py