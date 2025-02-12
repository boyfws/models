import numpy as np

from typing import Optional, Callable, Tuple, Literal
from nptyping import NDArray, Shape, Float32
from numpy.typing import ArrayLike


class GradOptimizer:
    @staticmethod
    def _optimize(
            param_shape: Tuple[int, 1],
            gradient: Callable[
                     [
                         NDArray[Shape["Any, 1"], Float32],
                         NDArray[Shape["Any, Any"], Float32],
                         NDArray[Shape["Any, 1"], Float32]
                      ],
                     NDArray[Shape["Any, 1"], Float32]
                 ],
            loss: Callable[
                     [
                         NDArray[Shape["Any, 1"], Float32],
                         NDArray[Shape["Any, 1"], Float32]
                     ],
                     float
                 ],
            predict: Callable[
                    [
                        NDArray[Shape["Any, Any"], Float32],
                        NDArray[Shape["Any, 1"], Float32]
                    ],
                    NDArray[Shape["Any, 1"], Float32]
                ],
            max_iter: int,
            learning_rate: float,
            X: NDArray[Shape["Any, Any"], Float32],
            y: NDArray[Shape["Any, 1"], Float32] = None,
            threshold: float = 0,
            start_point: Optional[NDArray[Shape["Any, 1"], Float32]] = None,
            random_state: Optional[int] = None,
    ) -> Tuple[
        NDArray[Shape["Any, 1"], Float32],
        NDArray[Shape["Any, Any, 1"], Float32],
        NDArray[Shape["Any"], Float32]
    ]:
        """
        Function use such functions:

        gradient(
            point: NDArray[Shape[k, 1]],
            X: NDArray[Shape[n, m]],
            y: NDArray[Shape[n, 1]]
        ) -> NDArray[Shape[k, 1]]

        loss(
            y_pred: NDArray[Shape[n, 1]],
            y_true:  NDArray[Shape[n, 1]]
        ) -> float

        predict(
            X: NDArray[Shape[n, m]],
            point: NDArray[Shape[k, 1]]
        ) -> NDArray[Shape[n, 1]]

        :return: final point, gradient history, loss history
        """

        if start_point is not None:
            assert start_point.shape == param_shape, "'start_point' must have the same shape as 'param_shape'"
        else:
            rng = np.random.default_rng(seed=random_state)
            start_point = rng.normal(loc=0, scale=1, size=param_shape).astype(np.float32)

        point = start_point.copy()

        grad_history = np.zeros((max_iter, param_shape[0], param_shape[1]), dtype=np.float32)
        loss_history = np.zeros(max_iter, dtype=np.float32)

        final_index = max_iter

        for i in range(max_iter):
            grad = gradient(point, X, y)
            grad_history[i] = grad

            point -= learning_rate * grad

            prediction = predict(X, point)
            loss_val = loss(prediction, y)
            loss_history[i] = loss_val

            if loss_val <= threshold:
                final_index = i + 1
                break

        return (point,
                grad_history[:final_index],
                loss_history[:final_index])


class GradStorage:
    """
    Class which stores gradient functions
    """
    @staticmethod
    def _MSE_grad(
            point: NDArray[Shape["Any, 1"], Float32],
            X: NDArray[Shape["Any, Any"], Float32],
            y: NDArray[Shape["Any, 1"], Float32]
    ) -> NDArray[Shape["Any, 1"], Float32]:
        return 2 * X.T @ (X @ point - y) / X.shape[0]

    @staticmethod
    def _MAE_grad(
            point: NDArray[Shape["Any, 1"], Float32],
            X: NDArray[Shape["Any, Any"], Float32],
            y: NDArray[Shape["Any, 1"], Float32]
    ) -> NDArray[Shape["Any, 1"], Float32]:
        return X.T @ np.sign(X @ point - y) / X.shape[0]

    @staticmethod
    def _MAPE_grad(
            point: NDArray[Shape["Any, 1"], Float32],
            X: NDArray[Shape["Any, Any"], Float32],
            y: NDArray[Shape["Any, 1"], Float32]
    ) -> NDArray[Shape["Any, 1"], Float32]:
        return X.T @ (np.sign(X @ point - y) / y) / X.shape[0]



class LossStorage:
    """
    Class which stores loss funcions
    """
    @staticmethod
    def _MSE_loss(
            y_pred: NDArray[Shape["Any, 1"], Float32],
            y_true: NDArray[Shape["Any, 1"], Float32]
    ) -> float:
        return ((y_pred - y_true) ** 2).mean()

    @staticmethod
    def _MAE_loss(
            y_pred: NDArray[Shape["Any, 1"], Float32],
            y_true: NDArray[Shape["Any, 1"], Float32]
    ) -> float:
        return (np.abs(y_pred - y_true)).mean()

    @staticmethod
    def _MAPE_loss(
            y_pred: NDArray[Shape["Any, 1"], Float32],
            y_true: NDArray[Shape["Any, 1"], Float32]
    ) -> float:
        return (np.abs(y_pred - y_true) / y_true).mean()


class NotFittedError(ValueError):
    pass


class LinRegGD(GradOptimizer,
               LossStorage,
               GradStorage
               ):

    def _make_stochastic(self, func, share: float) -> Callable:
        def wrapper(
            point: NDArray[Shape["Any, 1"], Float32],
            X: NDArray[Shape["Any, Any"], Float32],
            y: NDArray[Shape["Any, 1"], Float32]
        ):
            new_last_index = self._flag + int(X.shape[0] * share)

            div = new_last_index % X.shape[0]
            if div == 0:
                X = X[self._flag: new_last_index]
                y = y[self._flag: new_last_index]
                self._flag = new_last_index
            else:
                X = np.vstack([X[self._flag:], X[:div]])
                y = np.vstack([y[self._flag:], y[:div]])
                self._flag = div

            return func(point, X, y)

        return wrapper

    def _init_grad_loss(self, loss_function: str, share: Optional[float] = None) -> None:
        match loss_function:
            case "MSE":
                self._loss = self._MSE_loss
                self._grad = self._MSE_grad
            case "MAE":
                self._loss = self._MAE_loss
                self._grad = self._MAE_grad
            case "MAPE":
                self._loss = self._MAPE_loss
                self._grad = self._MAPE_grad
            case _:
                raise ValueError("No matching loss function")

        if self._stochastic:
            self._grad = self._make_stochastic(self._grad, share)

    def __init__(self,
                loss_function: Literal["MSE", "MAE", "MAPE"],
                stochastic: bool,
                add_const: bool,
                learning_rate: float,
                max_iter: int,
                random_state: Optional[int] = None,
                threshold: Optional[float] = None,
                stochastic_share: Optional[float] = 0.1) -> None:
        self._flag = 0

        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._random_state = random_state

        if threshold is None:
            self._threshold = 0
        else:
            self._threshold = threshold

        self._const = add_const
        self._stochastic = stochastic

        self._init_grad_loss(loss_function, stochastic_share)


    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        X = np.array(X, dtype=np.float32)
        if self._const:
            X = np.hstack(
                [X,
                 np.ones((X.shape[0], 1), dtype=np.float32)
                 ]
            )
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        weights, grad_history, loss_history = self._optimize(
            (X.shape[1], 1),
            self._grad,
            self._loss,
            lambda data, w: data @ w,
            self._max_iter,
            self._learning_rate,
            X,
            y,
            random_state=self._random_state,
            threshold=self._threshold
        )
        self._weights = weights
        self._loss_history = loss_history

    def predict(self, X: ArrayLike) -> NDArray[Shape["Any, 1"], Float32]:
        if not hasattr(self, "_weights"):
            raise NotFittedError("You must fit model first")

        X = np.array(X, dtype=np.float32)
        if self._const:
            X = np.hstack(
                [X,
                 np.ones((X.shape[0], 1), dtype=np.float32)
                 ]
            )

        if X.shape[1] != self._weights.shape[0]:
            raise ValueError(f"Input data has got wrong number of features, expected: {self._weights.shape[0]}, got: {X.shape[1]}")

        return X @ self._weights

    def get_loss_history(self) -> NDArray[Shape["Any"], Float32]:
        if not hasattr(self, "_loss_history"):
            raise NotFittedError("You must fit model first")

        return self._loss_history.copy()





