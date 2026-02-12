from metrics.fit_functions import rmse, mae
from readers.slb_eclipse.eclipse_reader import EclipseReader

__all__ = ["EclipseReader"]

EclipseReader.register_metric("RMSE", rmse, overwrite=True)
EclipseReader.register_metric("MAE", mae, overwrite=True)
