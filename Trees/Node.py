import numpy as np
from typing import Union

class Condition:
    feature_num: int
    t: float

    """
    A class that determines whether the data 
    belongs to one of the groups i.e. defines a split 
    """
    def __init__(self, feature_num: int, t: float) -> None:
        self.feature_num = feature_num
        self.t = t

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.feature_num] >= self.t


class Node:
    left_node: Union["Node", None]
    right_node: Union["Node", None]
    value: Union[int, float, np.ndarray]
    condition: "Condition"
    """"
    This class represents a building block of a tree, 
    depending on the context it can both refer 
    to other nodes and store a certain value in itself
    """
    def __init__(self):
        self.left_node = None # False cond
        self.right_node = None # True cond

        self.value = None

        self.condition = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Recursively called function, is called until
        we come to a node that does not reference any others
        """
        if self.left_node is None or self.right_node is None:
            return np.array([self.value] * X.shape[0], dtype=np.float32)

        mask = self.condition(X)

        true_pred = self.right_node.predict(X[mask])
        false_pred = self.left_node.predict(X[~mask])

        if true_pred.ndim == 1:
            shape = X.shape[0]
        else:
            shape = (X.shape[0], true_pred.shape[1])

        pred = np.empty(
            shape,
            dtype=np.float32
        )
        pred[mask] = true_pred
        pred[~mask] = false_pred

        return pred.copy()