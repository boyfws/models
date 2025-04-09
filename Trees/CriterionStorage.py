import numpy as np


def mse_criterion(y: np.ndarray) -> float:
    return float(
        np.mean( (y - np.mean(y)) ** 2)
    )

def mae_criterion(y: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(y - np.median(y))
        )
    )


class CriterionStorageRegressor:
    """
    Storage class for criteria used in the evaluation of splits
    """
    def _add_criterion(self, loss_name: str) -> None:
        if loss_name == "MSE":
            self.criterion = mse_criterion
        elif loss_name == "MAE":
            self.criterion = mae_criterion
        else:
            raise NotImplementedError(f"Loss {loss_name} not implemented")