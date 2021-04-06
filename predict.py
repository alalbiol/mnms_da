#!/usr/bin/env python
# coding: utf-8

# ---- My utils ----
from utils.arguments import *
from utils.mnms import test_prediction

if __name__ == "__main__":
    print("\nStart Test Prediction...")
    test_prediction(args)
    print("Finish!")
