# Minimal stub for econml.iv.nnet.DeepIV used in causal_pattern_detector imports
class DeepIV:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        return self

    @staticmethod
    def predict(X):
        try:
            import numpy as _np
            if hasattr(X, 'shape'):
                return _np.zeros((X.shape[0],))
            return 0.0
        except Exception:
            return 0.0
