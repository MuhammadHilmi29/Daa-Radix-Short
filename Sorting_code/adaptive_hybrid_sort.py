"""AHS-inspired controller integrated with the user's sorting modules.

This is an educational reimplementation based on the paper description,
not the author's original source code. It deliberately uses only the
provided Insertion Sort, Counting Sort, and Radix Sort modules; Quick Sort
is not imported or used.
"""

from collections import Counter
from dataclasses import dataclass
from enum import Enum, auto
import math
from pathlib import Path

from .counting_sort import counting_sort
from .insertion_sort import insertion_sort
from .radix_sort import radix_sort


class State(Enum):
    START = auto()
    FEATURE_EXTRACTION = auto()
    CHECK_SMALL_INPUT = auto()
    CHECK_SMALL_RANGE = auto()
    CHECK_LARGE_RANGE_ENTROPY = auto()
    XGBOOST_PREDICTION = auto()
    EXECUTE_SORT = auto()
    DONE = auto()


@dataclass
class Features:
    n: int
    k: int
    entropy: float
    normalized_entropy: float


def extract_features(data):
    """Calculate n, key range k, Shannon entropy H, and normalized H."""
    n = len(data)
    if n == 0:
        return Features(0, 0, 0.0, 0.0)

    min_value = min(data)
    max_value = max(data)
    k = max_value - min_value + 1

    frequencies = Counter(data)
    entropy = 0.0
    for frequency in frequencies.values():
        probability = frequency / n
        entropy -= probability * math.log2(probability)

    normalized_entropy = entropy / math.log2(k) if k > 1 else 0.0
    return Features(n, k, entropy, normalized_entropy)


class AHS:
    """Adaptive Hybrid Sort controller.

    Parameters
    ----------
    model_path : str or Path
        Optional path to an XGBoost model saved in JSON format.
    use_xgboost : bool
        If True, load and use the model when the deterministic FSM rules
        have not selected an algorithm. If the model is unavailable, Radix
        Sort is used as fallback because this project excludes Quick Sort.
    """

    def __init__(self, model_path="ahs_model.json", use_xgboost=True):
        self.model_path = Path(model_path)
        self.use_xgboost = use_xgboost
        self.model = None
        self.trace = []
        self.last_features = None
        self.last_algorithm = None
        if self.use_xgboost and self.model_path.exists():
            self._load_model()

    def _load_model(self):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "File model ditemukan, tetapi XGBoost belum terpasang. "
                "Jalankan: pip install -r requirements.txt"
            ) from exc
        self.model = XGBClassifier()
        self.model.load_model(str(self.model_path))

    def _log(self, state, detail=""):
        self.trace.append(state.name if not detail else f"{state.name}: {detail}")

    def _xgboost_predict(self, features):
        if self.model is None:
            return None
        row = [[features.n, features.k, features.entropy,
                features.normalized_entropy]]
        predicted_class = int(self.model.predict(row)[0])
        # Class IDs are defined in train_model.py.
        class_to_algorithm = {0: "insertion", 1: "counting", 2: "radix"}
        return class_to_algorithm.get(predicted_class)

    def _select_algorithm(self, features):
        self.trace = []
        self._log(State.START)
        self._log(
            State.FEATURE_EXTRACTION,
            f"n={features.n}, k={features.k}, H={features.entropy:.4f}, "
            f"H_norm={features.normalized_entropy:.4f}"
        )

        # Deterministic FSM gates, following the paper's methodology thresholds.
        self._log(State.CHECK_SMALL_INPUT)
        if features.n <= 20:
            return "insertion"

        self._log(State.CHECK_SMALL_RANGE)
        if features.k <= 1000:
            return "counting"

        self._log(State.CHECK_LARGE_RANGE_ENTROPY)
        if features.k > 1_000_000 and features.normalized_entropy < 0.7:
            return "radix"

        self._log(State.XGBOOST_PREDICTION)
        prediction = self._xgboost_predict(features)
        if prediction is not None:
            self._log(State.XGBOOST_PREDICTION, f"prediction={prediction}")
            return prediction

        # User explicitly does not use Quick Sort, so Radix Sort is the fallback.
        self._log(State.XGBOOST_PREDICTION,
                  "model unavailable; fallback=radix (Quick Sort excluded)")
        return "radix"

    def sort(self, arr):
        data = list(arr)
        if any(not isinstance(value, int) or isinstance(value, bool)
               for value in data):
            raise TypeError("AHS hanya menerima elemen integer.")

        if not data:
            self.trace = ["START", "DONE: empty input"]
            self.last_features = extract_features(data)
            self.last_algorithm = None
            return []

        features = extract_features(data)
        self.last_features = features
        algorithm = self._select_algorithm(features)
        self.last_algorithm = algorithm

        algorithms = {
            "insertion": insertion_sort,
            "counting": counting_sort,
            "radix": radix_sort,
        }
        self._log(State.EXECUTE_SORT, algorithm)
        result = algorithms[algorithm](data)
        self._log(State.DONE)
        return result

    def explain(self):
        """Return the state-transition log for the latest sort operation."""
        return self.trace.copy()


def adaptive_hybrid_sort(arr, model_path="ahs_model.json", use_xgboost=True):
    """Convenience function returning a sorted copy of arr."""
    return AHS(model_path=model_path, use_xgboost=use_xgboost).sort(arr)


if __name__ == "__main__":
    sample = [12, -3, 5, 5, 100, 0, -3, 8, 1]
    sorter = AHS(use_xgboost=False)
    print("Input:", sample)
    print("Output:", sorter.sort(sample))
    print("Selected algorithm:", sorter.last_algorithm)
    print("FSM trace:")
    for item in sorter.explain():
        print(" -", item)