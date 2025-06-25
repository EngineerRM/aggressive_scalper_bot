import backtesting.lib
import functools

def fix_partial_module():
    original_all = backtesting.lib.__all__
    backtesting.lib.__all__ = [
        getattr(v, "__name__", k)
        for k, v in backtesting.lib.__dict__.items()
        if (callable(v) and (getattr(v, "__module__", None) == backtesting.lib.__name__)) or
           k in original_all
    ]

fix_partial_module()