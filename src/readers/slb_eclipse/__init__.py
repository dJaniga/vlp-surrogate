from readers.metrics import rmse, mae
from readers.slb_eclipse.eclipse_summary_reader import EclipseSummaryReader

__all__ = ["EclipseSummaryReader"]

EclipseSummaryReader.register_metric("RMSE", rmse, overwrite=True)
EclipseSummaryReader.register_metric("MAE", mae, overwrite=True)
