from typing import Optional, List
import numpy as np
from random import randint


class KMeans:
    def __init__(self, k: int, max_steps: int = 1000, rtol: float = 0.05):
        self.k = k
        self.max_steps = max_steps
        self.rtol = rtol
        self.centers = None

    def fit(self, X: np.ndarray, centers_init: Optional[np.ndarray] = None) -> None:
        if centers_init is None:
            centers_init = self._get_init(X=X)

        new_centers = centers_init
        for _ in range(self.max_steps):
            converged = self._is_converged(new_centers)
            self.centers = new_centers
            if converged:
                break
            assignment = self._get_assignment(X, new_centers)
            new_centers = self._get_new_centers(assignment, X)

        return self.centers

    def _get_new_centers(self, assignment: List[int], X: np.ndarray) -> np.ndarray:
        center_to_point = dict()
        for i, center in enumerate(assignment):
            if center in center_to_point:
                center_to_point[center].append(i)
            else:
                center_to_point[center] = [i]

        new_centers = []
        for center in center_to_point:
            points = X[center_to_point[center]]
            new_centers.append(np.mean(points, axis=0))
        return np.array(new_centers)

    def _get_assignment(self, X: np.ndarray, centers: np.ndarray) -> List[int]:
        assignment = []
        for point in X:
            distances = []
            for center in centers:
                distances.append(np.linalg.norm(point - center))
            closest_centroid = np.argmin(distances)

            assignment.append(closest_centroid)

        return assignment

    def _is_converged(self, new_centers: np.ndarray) -> bool:
        if self.centers is None:
            return False
        return np.isclose(self.centers, new_centers, rtol=self.rtol).all()

    def _get_init(self, X: np.ndarray) -> np.ndarray:
        centroid_indices = [randint(0, X.shape[0] - 1) for _ in range(self.k)]
        return X[centroid_indices]


def main():
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    k = 2
    centers_init = np.array([[1, 5], [10, 5]], dtype=float)

    k_means_model = KMeans(k=k)
    k_means_model.fit(X=X, centers_init=centers_init)

    print(f"cluster centers: {k_means_model.centers}")
    assert np.array_equal(k_means_model.centers, np.array([[1, 2], [10, 2]]))


if __name__ == "__main__":
    main()
