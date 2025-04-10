from typing import Callable, Tuple
from Node import Node
import numpy as np
import abc


class DecisionTreeBase(abc.ABC):
    max_depth: int
    max_features: int | None
    min_samples_split: int
    min_samples_leaf: int
    random_state: int | None
    criterion: Callable[[np.ndarray] , float]
    rng: np.random.default_rng
    start_node_: "Node"

    """
    Base class for all trees, 
    which includes universal methods for finding optimal splits
    """

    @staticmethod
    def _solve_optimal_split_for_feature(
            X: np.ndarray,
            y: np.ndarray,
            min_samples: int,
            criterion: Callable[[np.ndarray] , float]
    ) -> Tuple[float, float]:
        """
        Tries to minimize the criterion,
        uses Histogram-Based Splitting to find the optimal split
        Returns:
            - Best splitting value
            - Loss after split
        """
        num_samples = X.shape[0]
        p = np.arange(0.1, 1, 0.1)

        quantiles = np.quantile(X, p).round(5)

        masks = X >= quantiles.reshape(-1, 1)

        losses = np.full(quantiles.shape[0], np.inf)

        for i in range(masks.shape[0]):
            y_pos = y[masks[i]]
            y_neg = y[~masks[i]]

            y_pos_size = y_pos.shape[0]
            y_neg_size = y_neg.shape[0]

            if y_pos_size <= min_samples or y_neg_size <= min_samples:
                continue

            split_loss = (
                                 y_pos_size * criterion(y_pos) + y_neg_size * criterion(y_neg)
                         ) / num_samples

            losses[i] = split_loss

        min_arg = np.argmin(losses)

        return quantiles[min_arg], losses[min_arg]

    def _solve_split(self,
                     X: np.ndarray,
                     y: np.ndarray
    ) -> Tuple[int, float, float]:

        """
        Tries to minimize the criterion,
        by finding the optimal feature to split
        Returns:
            - column id of best split
            - best split value for this column
            - loss diff of  best split
        """
        if self.max_features:
            max_features = min(self.max_features, X.shape[1])

        else:
            max_features = X.shape[1]

        indices = self.rng.choice(X.shape[1],
                                  size=max_features,
                                  replace=False
                                  )

        start_loss = self.criterion(y)

        splits, losses = np.zeros(indices.shape[0]), np.full(indices.shape[0], np.inf)
        for i, el in enumerate(indices):
            splits[i], losses[i] = self._solve_optimal_split_for_feature(
                X[:, el],
                y,
                min_samples=self.min_samples_leaf,
                criterion=self.criterion
            )

        delta_loss = losses - start_loss

        min_arg = np.argmin(delta_loss)

        best_feature = indices[min_arg]
        best_t = splits[min_arg]
        best_diff = delta_loss[min_arg]

        return best_feature, best_t, best_diff


    def predict(self,
                X: np.ndarray
    ) -> np.ndarray:
        pass


    def fit(self,
            X: np.ndarray,
            y: np.ndarray
    ) -> None:
        pass




