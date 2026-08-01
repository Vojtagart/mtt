from .. import _core
import numpy as np
from numpy.typing import DTypeLike
from typing import Optional, Tuple, Type, Any


_DTYPE_SUFF = {
    "float64": "d",
    "float32": "f",
    np.dtype("float64"): "d",
    np.dtype("float32"): "f",
    np.float64: "d",
    np.float32: "f",
    float: "d",
}

def _get_binding_class(base_name: str, dtype: DTypeLike = np.float64, params: Optional[Tuple[Any, ...]] = None) -> Type:

    type_suff = _DTYPE_SUFF.get(dtype)
    if type_suff is None:
        raise ValueError(f"Unsupported dtype: {dtype}. Supported dtypes are {_DTYPE_SUFF.keys()}")

    if params is not None:
        param_str = '_'.join([str(x) for x in params])
        class_name = f"{base_name}_{type_suff}_{param_str}"
        if hasattr(_core, class_name):
            return getattr(_core, class_name)
    
    class_name = f"{base_name}_{type_suff}"
    if hasattr(_core, class_name):
        return getattr(_core, class_name)
        
    raise NotImplementedError(f"No binding found for {base_name} with dtype={dtype} and params={'None' if params is None else params}")
