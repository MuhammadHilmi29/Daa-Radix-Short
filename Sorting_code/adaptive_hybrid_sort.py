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
    VALIDATE_INPUT = auto()
    FEATURE_EXTRACTION = auto()
    CHECK_UNIFORM_ARRAY = auto()
    CHECK_SMALL_INPUT = auto()
    CHECK_SMALL_RANGE = auto()
    CHECK_LARGE_RANGE_ENTROPY = auto()
    XGBOOST_PREDICTION = auto()
    EXECUTE_SORT = auto()
    DONE = auto()


@dataclass
class Features:
    n: int
    min_val: int
    max_val: int
    k: int
    entropy: float
    normalized_entropy: float


def extract_features(data):
    """Menghitung n, min, max, rentang k, entropi Shannon H, dan H ternormalisasi."""
    n = len(data)
    if n == 0:
        return Features(0, 0, 0, 0, 0.0, 0.0)

    min_value = min(data)
    max_value = max(data)
    k = max_value - min_value + 1

    if k == 1:
        return Features(n, min_value, max_value, 1, 0.0, 0.0)

    frequencies = Counter(data)
    entropy = 0.0
    for frequency in frequencies.values():
        probability = frequency / n
        entropy -= probability * math.log2(probability)

    # Sesuai jurnal Bagian 3.1: kondisi entropi rendah adalah H < 0.7 * log2(k)
    # yang ekuivalen dengan normalized_entropy < 0.7
    max_entropy = math.log2(k)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return Features(n, min_value, max_value, k, entropy, normalized_entropy)


class AHS:
    """Adaptive Hybrid Sort (AHS) controller sesuai jurnal (Balasubramanian, 2026)
    yang disesuaikan untuk 3 algoritma: Insertion Sort, Counting Sort, dan Radix Sort.
    """

    # Threshold sesuai Bagian 3.1, Fig. 1, dan Fig. 2 jurnal
    N_THRESHOLD = 20
    K_THRESHOLD = 1000
    K_LARGE_THRESHOLD = 1_000_000
    ENTROPY_RATIO_THRESHOLD = 0.7
    ML_MIN_SIZE = 1000  # Bagian 3.8 & Tabel 4: ML aktif untuk n >= 1000

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
                "Jalankan: pip install xgboost"
            ) from exc
        self.model = XGBClassifier()
        self.model.load_model(str(self.model_path))

    def _log(self, state, detail=""):
        self.trace.append(state.name if not detail else f"{state.name}: {detail}")

    def _xgboost_predict(self, features):
        if self.model is None:
            return None

        # Mendukung model dengan 3 fitur [n, k, H] (sesuai Listing 5 jurnal)
        # maupun 4 fitur [n, k, H, H_norm] jika dilatih dengan fitur tambahan
        expected_features = getattr(self.model, "n_features_in_", 3)
        if expected_features == 4:
            row = [[features.n, features.k, features.entropy, features.normalized_entropy]]
        else:
            row = [[features.n, features.k, features.entropy]]

        predicted_class = int(self.model.predict(row)[0])
        class_to_algorithm = {0: "insertion", 1: "counting", 2: "radix"}
        return class_to_algorithm.get(predicted_class)

    def _select_algorithm(self, features):
        # 1. Cek array seragam (k == 1) -> langsung selesai O(n) sesuai Bagian 3.11 jurnal
        self._log(State.CHECK_UNIFORM_ARRAY)
        if features.k == 1:
            return "none"

        # 2. Cek ukuran kecil (n <= 20) -> Insertion Sort
        self._log(State.CHECK_SMALL_INPUT)
        if features.n <= self.N_THRESHOLD:
            return "insertion"

        # 3. Cek rentang nilai sempit (k <= 1000) -> Counting Sort
        self._log(State.CHECK_SMALL_RANGE)
        if features.k <= self.K_THRESHOLD:
            return "counting"

        # 4. Cek rentang sangat besar & entropi rendah (k > 10^6 dan H < 0.7 * log2(k)) -> Radix Sort
        self._log(State.CHECK_LARGE_RANGE_ENTROPY)
        entropy_limit = self.ENTROPY_RATIO_THRESHOLD * math.log2(features.k)
        if features.k > self.K_LARGE_THRESHOLD and features.entropy < entropy_limit:
            return "radix"

        # 5. Area abu-abu: gunakan XGBoost Classifier untuk n >= 1000 (Bagian 3.8 jurnal)
        self._log(State.XGBOOST_PREDICTION)
        if self.use_xgboost and self.model is not None and features.n >= self.ML_MIN_SIZE:
            prediction = self._xgboost_predict(features)
            if prediction is not None:
                self._log(State.XGBOOST_PREDICTION, f"prediction={prediction}")
                return prediction

        # 6. Fallback umum: Radix Sort (menggantikan QuickSort sesuai ketentuan)
        reason = (
            f"n < {self.ML_MIN_SIZE} (bypass ML overhead)"
            if features.n < self.ML_MIN_SIZE
            else "model unavailable"
        )
        self._log(State.XGBOOST_PREDICTION, f"{reason}; fallback=radix")
        return "radix"

    def sort(self, arr):
        self.trace = []
        self._log(State.START)

        # Validasi tipe input ketat: hanya menerima bilangan bulat (Bagian 3.11 jurnal)
        self._log(State.VALIDATE_INPUT)
        data = list(arr)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in data):
            raise TypeError("AHS hanya menerima elemen integer.")

        # Penanganan array kosong dalam O(1) (Bagian 3.11 jurnal)
        if not data:
            self.last_features = extract_features(data)
            self.last_algorithm = None
            self._log(State.DONE, "empty input")
            return []

        # Ekstraksi fitur statistik (n, k, H, H_norm)
        features = extract_features(data)
        self.last_features = features
        self._log(
            State.FEATURE_EXTRACTION,
            f"n={features.n}, k={features.k}, H={features.entropy:.4f}, "
            f"H_norm={features.normalized_entropy:.4f}",
        )

        # Pemilihan strategi melalui Decision Engine
        algorithm = self._select_algorithm(features)
        self.last_algorithm = algorithm

        if algorithm == "none":
            self._log(State.DONE, "uniform array (k=1), skipped sorting")
            return data

        self._log(State.EXECUTE_SORT, algorithm)
        if algorithm == "insertion":
            result = insertion_sort(data)
        elif algorithm == "counting":
            result = counting_sort(data, features.min_val, features.max_val)
        elif algorithm == "radix":
            result = radix_sort(data, key_range=features.k)
        else:
            raise ValueError(f"Algoritma tidak dikenal: {algorithm}")

        self._log(State.DONE)
        return result

    def explain(self):
        """Mengembalikan jejak transisi state FSM untuk eksekusi terakhir."""
        return self.trace.copy()


def adaptive_hybrid_sort(arr, model_path="ahs_model.json", use_xgboost=True):
    """Fungsi pembantu untuk langsung mengembalikan salinan array yang terurut."""
    return AHS(model_path=model_path, use_xgboost=use_xgboost).sort(arr)


if __name__ == "__main__":
    sorter = AHS(use_xgboost=False)

    test_cases = {
        "1. Kasus kecil (n <= 20) -> Insertion": [12, -3, 5, 5, 100, 0, -3, 8, 1],
        "2. Rentang sempit (k <= 1000) -> Counting": [500 - (i % 50) for i in range(100)],
        "3. Rentang besar & entropi rendah -> Radix (base 256)": [0] * 95 + [5_000_000, -2_000_000, 10, 20, 30],
        "4. Nilai seragam (k = 1) -> Skip": [7] * 50,
    }

    for title, sample in test_cases.items():
        sorted_data = sorter.sort(sample)
        print(f"\n=== {title} ===")
        print("Algoritma terpilih:", sorter.last_algorithm)
        print("Terurut dengan benar:", sorted_data == sorted(sample))
        print("Jejak FSM:", " -> ".join(sorter.explain()))