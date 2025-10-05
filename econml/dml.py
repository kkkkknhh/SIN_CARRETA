# coding=utf-8
# Minimal stub for econml.dml providing CausalForestDML used in causal_pattern_detector
class CausalForestDML:
    def __init__(self, model_y=None, model_t=None, discrete_treatment=False, cv=3):
        self.model_y = model_y
        self.model_t = model_t
        self.discrete_treatment = discrete_treatment
        self.cv = cv
        self._fitted = False

    def fit(self, Y, T, X=None, W=None):
        self._fitted = True

    @staticmethod
    def effect(X):
        try:
            import numpy as _np
            if hasattr(X, 'shape'):
                return _np.zeros(X.shape[0])
            return 0.0
        except Exception:
            return 0.0

    def effect_interval(self, X, alpha=0.05):
        eff = self.effect(X)
        return (eff, eff)
