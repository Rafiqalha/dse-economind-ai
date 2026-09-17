"""Load and run this project's scikit-learn forest without importing sklearn.

The workstation's Application Control policy blocks unsigned extension modules
used by pandas, SciPy, and scikit-learn.  Joblib and NumPy's core modules are
still available, so the serialized decision trees can be read and evaluated
directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from joblib.numpy_pickle import NumpyUnpickler


class _SerializedObject:
    """Minimal target for scikit-learn objects stored in the joblib file."""

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._constructor_args = args
        instance._constructor_kwargs = kwargs
        return instance

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._serialized_state = state


class _PortableUnpickler(NumpyUnpickler):
    def find_class(self, module, name):
        if module.startswith("sklearn."):
            return _SerializedObject
        return super().find_class(module, name)


class PortableRandomForest:
    """Small inference-only implementation for a fitted RandomForestClassifier."""

    def __init__(self, serialized_model):
        self._model = serialized_model
        self.feature_names = tuple(str(name) for name in serialized_model.feature_names_in_)
        self.classes = tuple(str(label) for label in serialized_model.classes_)
        self._trees = tuple(serialized_model.estimators_)

        if not self._trees:
            raise ValueError("Model Random Forest tidak memiliki decision tree.")

    def predict_one(self, values: Sequence[float]) -> str:
        if len(values) != len(self.feature_names):
            raise ValueError(
                f"Model membutuhkan {len(self.feature_names)} fitur, "
                f"tetapi menerima {len(values)}."
            )

        probabilities = [0.0] * len(self.classes)

        for estimator in self._trees:
            tree = estimator.tree_
            state = tree.__getstate__() if hasattr(tree, "__getstate__") else tree.__dict__
            nodes = state["nodes"]
            class_values = state["values"]
            node_index = 0

            while nodes[node_index]["left_child"] != -1:
                node = nodes[node_index]
                feature_value = float(values[int(node["feature"])])
                if feature_value <= float(node["threshold"]):
                    node_index = int(node["left_child"])
                else:
                    node_index = int(node["right_child"])

            leaf_values = class_values[node_index][0]
            total = float(sum(leaf_values))
            if total == 0:
                continue
            for class_index, value in enumerate(leaf_values):
                probabilities[class_index] += float(value) / total

        winner = max(range(len(probabilities)), key=probabilities.__getitem__)
        return self.classes[winner]


def load_portable_forest(path: str | Path) -> PortableRandomForest:
    model_path = Path(path).resolve()
    with model_path.open("rb") as model_file:
        unpickler = _PortableUnpickler(
            str(model_path),
            model_file,
            ensure_native_byte_order=True,
        )
        serialized_model = unpickler.load()

    return PortableRandomForest(serialized_model)
