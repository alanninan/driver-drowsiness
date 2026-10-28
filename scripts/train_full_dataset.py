"""Legacy entry point: train Folds1-4 and test Fold5; never update the live model.

The original all-fold training source is preserved with its historical artifacts.
"""
from train_holdout import main

if __name__ == "__main__":
    main()
