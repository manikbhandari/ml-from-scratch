import pdb  # noqa: F401
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy import stats

_SEED = 23
_TEST_SIZE = 0.2
_K = 10


class KNNClassifier:
    def __init__(self, k: int, distance_strategy: str = "l2") -> None:
        self.k = k
        self.distance_strategy = distance_strategy

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.X_train = X_train  # N X D
        self.y_train = y_train  # N

    def fit_transform(self, X_test: np.ndarray) -> np.ndarray:
        closest_neighbors = self._get_closest_neighbors(X_test)  # N2 X k
        y_train = np.expand_dims(self.y_train, 0)  # 1 X N
        y_train = np.tile(y_train, (X_test.shape[0], 1))  # N2 X N
        closest_y_train = np.take_along_axis(
            y_train,
            closest_neighbors,
            axis=1,
        )
        mode = stats.mode(closest_y_train, axis=1, keepdims=True)  # N2 X 1
        return np.squeeze(mode.mode, axis=1)

    def _get_closest_neighbors(self, X_test: np.ndarray) -> np.ndarray:
        """
        Returns a 1-d array of the indices of closest neighbors in X_train
        """
        norms = self._get_norms(X_test)
        sorted_neighbors = np.argsort(norms, axis=1)  # N2 X N
        return sorted_neighbors[:, : self.k]

    def _get_norms(self, X_test: np.ndarray) -> np.ndarray:
        if self.distance_strategy == "l2":
            X_train = np.expand_dims(self.X_train, 0)  # 1 X N X D
            X_test = np.expand_dims(X_test, 1)  # N2 X 1 X D
            differences = X_test - X_train  # N2 X N X D
            norms = np.linalg.norm(differences, axis=2)  # N2 X N
        else:
            raise NotImplementedError(
                f"Strategy {self.distance_strategy} has not been implemented",
            )
        return norms


def get_accuracy(y_preds: np.ndarray, y_test: np.ndarray) -> float:
    if y_preds.shape != y_test.shape:
        raise ValueError(
            f"Shape mismatch. y_preds has shape {y_preds.shape} "
            f"but y_test has shape {y_test.shape}"
        )
    return sum(y_preds == y_test) / y_preds.shape[0]


def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, random_state=_SEED
    )
    clf = KNNClassifier(k=_K)
    clf.fit(X_train, y_train)
    y_preds = clf.fit_transform(X_test)
    accuracy = get_accuracy(y_preds, y_test)
    print(f"{accuracy=}")


if __name__ == "__main__":
    main()
