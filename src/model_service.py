from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap


class SepsisModelService:
    def __init__(self, bundle_path: str | Path):
        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            raise FileNotFoundError(f"Model bundle not found: {bundle_path.resolve()}")

        self.bundle = joblib.load(bundle_path)
        self.pipe = self.bundle["base_model"]  # sklearn Pipeline
        self.threshold = float(self.bundle["chosen_threshold"])
        self.feature_columns = self.bundle.get("feature_columns", None)

        # Lazy SHAP init (only when you ask for an explanation)
        self._explainer = None
        self._preprocess = None
        self._model = None
        self._feature_names = None

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure incoming data matches training columns (adds missing cols as NaN, drops extras)."""
        if self.feature_columns is None:
            return X
        return X.reindex(columns=self.feature_columns)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = self._align_features(X)
        return self.pipe.predict_proba(X)[:, 1]

    def predict_alert(self, X: pd.DataFrame, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        thr = self.threshold if threshold is None else float(threshold)
        proba = self.predict_proba(X)
        alert = (proba >= thr).astype(int)
        return proba, alert

    def explain_one_row(self, X_row: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        """
        Returns per-feature SHAP values for one row.
        Note: these are for transformed features (includes missing-indicator features).
        """
        if len(X_row) != 1:
            raise ValueError("X_row must be a single-row DataFrame.")

        X_row = self._align_features(X_row)

        # Lazy init explainer for speed
        if self._explainer is None:
            self._preprocess = self.pipe.named_steps["preprocess"]
            self._model = self.pipe.named_steps["model"]

            try:
                self._feature_names = self._preprocess.get_feature_names_out()
            except Exception:
                self._feature_names = None

            self._explainer = shap.TreeExplainer(self._model)

        Xt = self._preprocess.transform(X_row)
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt

        sv = self._explainer.shap_values(Xt)

        # Pick positive class safely
        if isinstance(sv, list):
            vals = sv[1][0]
        elif getattr(sv, "ndim", 0) == 3:
            vals = sv[:, :, 1][0]
        else:
            vals = sv[0]

        names = (
            list(self._feature_names)
            if self._feature_names is not None
            else [f"f{i}" for i in range(len(vals))]
        )

        out = pd.DataFrame({"feature": names, "shap_value": vals})
        out["abs"] = out["shap_value"].abs()
        out = out.sort_values("abs", ascending=False).drop(columns="abs")

        return out.head(top_n)
