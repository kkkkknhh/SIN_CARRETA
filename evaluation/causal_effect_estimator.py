"""
Bayesian Linear Regression for Causal Effect Estimation

Implements Normal-InverseGamma conjugate priors for estimating causal effects
in Theory of Change models. Supports:
- Closed-form Bayesian posterior updates
- Credible intervals using t-student marginals
- Prediction with uncertainty quantification
- Significance testing for causal effects

Author: MINIMINIMOON Team
Date: 2025
"""

import numpy as np
from scipy.stats import t
from typing import Tuple, Optional, Dict
import json
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BayesLinearEffect:
    """
    Regresión lineal bayesiana con prior Normal-InvGamma.
    Y = β₀ + β₁X + ε, ε ~ N(0, σ²)
    
    Prior conjugado:
    - β | σ² ~ N(μ₀, σ²V₀)
    - σ² ~ InvGamma(a₀, b₀)
    
    Attributes:
        mu0: Prior mean for β (shape 2: [β₀, β₁])
        V0: Prior covariance for β (shape 2x2)
        a0: InverseGamma shape parameter
        b0: InverseGamma scale parameter
        mun: Posterior mean for β (after fit)
        Vn: Posterior covariance for β (after fit)
        an: Posterior InverseGamma shape (after fit)
        bn: Posterior InverseGamma scale (after fit)
        n_obs: Number of observations
        fitted: Whether model has been fitted
    """
    # Prior hiperparámetros
    mu0: np.ndarray = None  # Media prior de β (shape 2)
    V0: np.ndarray = None   # Covarianza prior de β (shape 2x2)
    a0: float = 1.0         # Shape InvGamma
    b0: float = 1.0         # Scale InvGamma
    
    # Posterior (se llenan en fit)
    mun: Optional[np.ndarray] = None
    Vn: Optional[np.ndarray] = None
    an: Optional[float] = None
    bn: Optional[float] = None
    
    # Metadatos
    n_obs: int = 0
    fitted: bool = False
    
    def __post_init__(self):
        """Initialize default priors if not provided."""
        if self.mu0 is None:
            self.mu0 = np.array([0.0, 0.0])  # [β₀, β₁]
        if self.V0 is None:
            self.V0 = np.eye(2) * 1e3  # Prior vago (vague prior)
    
    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Ajusta modelo con datos observados usando conjugado Normal-InvGamma.
        
        Args:
            x: Variable independiente (shape n)
            y: Variable dependiente (shape n)
            
        Returns:
            self: Para method chaining
            
        Raises:
            ValueError: Si x e y tienen tamaños diferentes
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        
        if len(x) != len(y):
            raise ValueError("x e y deben tener mismo tamaño")
        
        self.n_obs = len(x)
        
        # Matriz de diseño [1, x]
        X = np.column_stack([np.ones_like(x), x])
        
        # Posterior Normal-InvGamma (conjugado)
        V0_inv = np.linalg.inv(self.V0)
        XtX = X.T @ X
        
        Vn = np.linalg.inv(V0_inv + XtX)
        mun = Vn @ (V0_inv @ self.mu0 + X.T @ y)
        
        an = self.a0 + self.n_obs / 2
        
        residual = y - X @ mun
        prior_term = (mun - self.mu0).T @ V0_inv @ (mun - self.mu0)
        bn = self.b0 + 0.5 * (residual @ residual + prior_term)
        
        self.mun = mun
        self.Vn = Vn
        self.an = an
        self.bn = bn
        self.fitted = True
        
        logger.info(f"BayesLinearEffect fitted: n={self.n_obs}, "
                   f"β₁ mean={mun[1]:.3f}")
        
        return self
    
    def beta_posterior_mean(self) -> np.ndarray:
        """
        Media posterior E[β] = μₙ
        
        Returns:
            Posterior mean of coefficients [β₀, β₁]
            
        Raises:
            RuntimeError: Si el modelo no ha sido ajustado
        """
        if not self.fitted:
            raise RuntimeError("Modelo no ajustado. Llamar fit() primero.")
        return self.mun
    
    def beta1_credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """
        Intervalo de credibilidad para β₁ (efecto causal).
        Marginal t-student con dof=2aₙ.
        
        Args:
            level: Credibility level (default 0.95 for 95% interval)
            
        Returns:
            Tuple (lower_bound, upper_bound) for β₁
            
        Raises:
            RuntimeError: Si el modelo no ha sido ajustado
        """
        if not self.fitted:
            raise RuntimeError("Modelo no ajustado")
        
        dof = 2 * self.an
        scale = np.sqrt(self.bn / self.an * self.Vn[1, 1])
        center = self.mun[1]
        
        alpha = (1 - level) / 2
        q_lo = t.ppf(alpha, df=dof)
        q_hi = t.ppf(1 - alpha, df=dof)
        
        return (center + q_lo * scale, center + q_hi * scale)
    
    def predict(self, x_new: np.ndarray, return_std: bool = False):
        """
        Predicción con incertidumbre.
        
        Args:
            x_new: Nuevos valores de X
            return_std: Si retornar desviación estándar
        
        Returns:
            (mean, std) si return_std=True, sino solo mean
            
        Raises:
            RuntimeError: Si el modelo no ha sido ajustado
        """
        if not self.fitted:
            raise RuntimeError("Modelo no ajustado")
        
        x_new = np.asarray(x_new, dtype=float).ravel()
        X_new = np.column_stack([np.ones_like(x_new), x_new])
        
        # Media predictiva
        y_mean = X_new @ self.mun
        
        if not return_std:
            return y_mean
        
        # Varianza predictiva
        sigma2_mean = self.bn / self.an
        y_var = sigma2_mean * (1 + np.diag(X_new @ self.Vn @ X_new.T))
        y_std = np.sqrt(y_var)
        
        return y_mean, y_std
    
    def effect_is_significant(self, threshold: float = 0.0) -> Tuple[bool, float]:
        """
        Verifica si el efecto causal β₁ es significativamente diferente de threshold.
        
        Args:
            threshold: Threshold value to compare against (default 0.0)
        
        Returns:
            (is_significant, probability): Tuple with significance flag and 
                                          probability that β₁ > threshold
        """
        if not self.fitted:
            return False, 0.0
        
        # P(β₁ > threshold) using t-distribution
        dof = 2 * self.an
        scale = np.sqrt(self.bn / self.an * self.Vn[1, 1])
        center = self.mun[1]
        
        # Avoid division by zero
        if scale < 1e-10:
            # If scale is very small, use posterior mean to determine significance
            prob = 1.0 if center > threshold else 0.0
        else:
            # β₁ ~ center + scale * T where T ~ t(dof)
            # P(β₁ > threshold) = P(center + scale*T > threshold)
            #                   = P(T > (threshold - center)/scale)
            #                   = P(T > -t_stat) where t_stat = (center - threshold)/scale
            t_stat = (center - threshold) / scale
            
            # For numerical stability:
            if t_stat > 10:  # Very strong evidence for β₁ > threshold
                prob = 1.0
            elif t_stat < -10:  # Very strong evidence for β₁ < threshold
                prob = 0.0
            else:
                # P(T > -t_stat) = sf(-t_stat)
                prob = t.sf(-t_stat, df=dof)
        
        # Significativo si prob > 0.95 o prob < 0.05
        is_sig = prob > 0.95 or prob < 0.05
        return is_sig, prob
    
    def get_summary(self) -> Dict:
        """
        Resumen del efecto causal estimado.
        
        Returns:
            Dict con estadísticas del modelo (fitted, n_obs, beta estimates, etc.)
        """
        if not self.fitted:
            return {"fitted": False}
        
        interval = self.beta1_credible_interval(level=0.95)
        is_sig, prob = self.effect_is_significant()
        
        return {
            'fitted': True,
            'n_obs': self.n_obs,
            'beta0_mean': float(self.mun[0]),
            'beta1_mean': float(self.mun[1]),
            'beta1_interval_95': interval,
            'sigma2_mean': float(self.bn / self.an),
            'effect_significant': is_sig,
            'prob_positive_effect': prob
        }
    
    def save(self, path: Path):
        """
        Persiste modelo a archivo JSON.
        
        Args:
            path: Path donde guardar el modelo
        """
        data = {
            'mu0': self.mu0.tolist(),
            'V0': self.V0.tolist(),
            'a0': self.a0,
            'b0': self.b0,
            'mun': self.mun.tolist() if self.fitted else None,
            'Vn': self.Vn.tolist() if self.fitted else None,
            'an': self.an,
            'bn': self.bn,
            'n_obs': self.n_obs,
            'fitted': self.fitted
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'BayesLinearEffect':
        """
        Carga modelo desde archivo JSON.
        
        Args:
            path: Path del archivo a cargar
            
        Returns:
            BayesLinearEffect instance con estado cargado
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        model = cls(
            mu0=np.array(data['mu0']),
            V0=np.array(data['V0']),
            a0=data['a0'],
            b0=data['b0']
        )
        
        if data['fitted']:
            model.mun = np.array(data['mun'])
            model.Vn = np.array(data['Vn'])
            model.an = data['an']
            model.bn = data['bn']
            model.n_obs = data['n_obs']
            model.fitted = True
        
        return model
