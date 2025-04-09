import numpy as np
from typing import Optional, Literal

from Node import Node, Condition
from DecisionTreeBase import DecisionTreeBase
from CriterionStorage import CriterionStorageRegressor


class DecisionTreeRegressor(DecisionTreeBase, CriterionStorageRegressor):
    def __init__(
            self,
            max_depth: int,
            loss: Literal["MAE", "MSE"],
            max_features: Optional[int] = None,
            min_samples_split: Optional[int] = None,
            min_samples_leaf: Optional[int] = None,
            random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth

        self._add_criterion(loss)
        self.loss = loss

        self.max_features = max_features

        if min_samples_leaf is None:
            self.min_samples_leaf = 0
        else:
            self.min_samples_leaf = min_samples_leaf

        if min_samples_split is None:
            self.min_samples_split = 0
        else:
            self.min_samples_split = min_samples_split

        self.rng = np.random.RandomState(random_state)

    def _set_value(self, y: np.ndarray) -> float:
        if self.loss == "MAE":
            return float(
                np.median(y)
            )

        elif self.loss == "MSE":
            return float(
                np.mean(y)
            )

        else:
            raise ValueError

    def fit(self,
            X: np.ndarray,
            y: np.ndarray
    ) -> None:
        self.start_node_ = Node()

        nodes = [(self.start_node_, X, y)]
        depth = 0
        while len(nodes) != 0 and depth < self.max_depth:
            new_nodes = []
            for el in nodes:
                node = el[0]
                X_node, y_node = el[1:]

                if (X_node.shape[0] <= self.min_samples_split or
                    X_node.shape[0] <= 2 * self.min_samples_leaf or
                    depth == (self.max_depth - 1)
                ):
                    node.value = self._set_value(y_node)
                    continue


                best_feature, best_t, best_diff = self._solve_split(X_node, y_node)

                condition = Condition(best_feature, best_t)
                mask = condition(X_node)

                if best_diff == np.inf or best_diff >= 0:
                    node.value = self._set_value(y_node)
                else:
                    node.condition = condition

                    node.right_node = Node()
                    node.left_node = Node()
                    new_nodes.append(
                        (node.right_node, X_node[mask], y_node[mask])
                    )
                    new_nodes.append(
                        (node.left_node, X_node[~mask], y_node[~mask])
                    )

            depth += 1
            nodes = new_nodes

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.start_node_.predict(X)
