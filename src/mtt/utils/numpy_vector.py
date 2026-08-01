import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from typing import Optional, Sequence, Tuple, Union


class NumpyVector:
    """
    Lightweight dynamic container optimized for appending small arrays

    Provides a contiguous, NumPy-backed dynamic array with explicit capacity 
    management to amortize allocations, behaving similarly to a C++ std::vector

    Parameters:
    - shape: Shape of the individual stored elements (e.g., `(4,)` or `4`)
    - dtype: NumPy data type for storage
    - initial: Optional array-like providing initial rows
    - cap: Minimal initial capacity
    """

    def __init__(self, shape: Union[int, Tuple[int, ...]], dtype: DTypeLike, initial: Optional[ArrayLike] = None, cap: int = 0):
        """
        Create a NumpyVector container

        Parameters:
        - shape: shape of the stored elements
        - dtype: numpy dtype for storage
        - initial: optional array-like providing initial rows. Will be coerced
        to shape `(N, *shape)`; `cap` will be set to at least `N`
        - cap: initial capacity; if `initial` is provided, capacity of the
        NumpyVector will be taken as max(`cap`, len(`initial`))
        """
        if isinstance(shape, int):
            shape = (shape,)

        if initial is None:
            cap = max(int(cap), 0)
            self._data = np.empty((cap, *shape), dtype=dtype)
            self.size = 0
        else:
            data = np.asarray(initial, dtype=dtype).reshape((-1, *shape))
            self.size = data.shape[0]
            cap = max((int(cap), data.shape[0]))
            self._data = np.empty((cap, *shape), dtype=dtype)
            self._data[:self.size] = data

    @property
    def cap(self) -> int:
        """Current allocated capacity (number of rows)"""
        return self._data.shape[0]
    
    @property
    def elem_shape(self) -> Tuple[int, ...]:
        """Per-row dimensionality (number of columns)"""
        return self._data.shape[1:]
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """(size, dim) tuple describing the active array shape"""
        return (self.size, *self.elem_shape)
    
    @property
    def data(self) -> np.ndarray:
        """
        View of the active storage

        Returns:
        - Internal buffer view of shape `(size, *elem_shape)`
        """
        return self._data[:self.size]
    
    @property
    def dtype(self):
        """NumPy dtype of the storage buffer"""
        return self._data.dtype
    
    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)
    
    def __len__(self) -> int:
        return self.size
    
    def __iter__(self):
        for i in range(self.size):
            yield self._data[i]

    def __getitem__(self, idx: Union[int, slice, np.ndarray, tuple]):
        return self.data[idx]
    
    def __setitem__(self, idx: Union[int, slice, np.ndarray, tuple], val: ArrayLike):
        self.data[idx] = val

    def reserve(self, cap: int):
        """
        Ensures at least `cap` elements of capacity are allocated

        Parameters:
        - cap: Minimum desired capacity
        """
        cap = int(cap)
        if cap <= self.cap:
            return
        data = np.empty((cap, *self.elem_shape), dtype=self.dtype)
        data[:self.size] = self._data[:self.size]
        self._data = data

    def _ensure_cap(self, cap: int):
        """
        Internal helper to grow capacity geometrically

        Parameters:
        - cap: Minimum required capacity
        """
        cap = int(cap)
        if cap <= self.cap:
            return
        ncap = max(cap, self.cap * 2 + 1)
        self.reserve(ncap)

    def append(self, elem: ArrayLike):
        """
        Appends one or more elements to the container

        Parameters:
        - elem: A single array of shape `elem_shape` or an array-like of shape `(N, *elem_shape)`
        """
        elem = np.asarray(elem, dtype=self.dtype).reshape((-1, *self.elem_shape))
        nsize = elem.shape[0] + self.size
        self._ensure_cap(nsize)
        self._data[self.size:nsize] = elem
        self.size = nsize

    def extend(self, elems: Sequence[ArrayLike]):
        """Alias for `append` accepting a sequence of rows"""
        self.append(elems)

    def pop(self, index: int = -1, preserve_order: bool = True) -> np.ndarray:
        """
        Removes and returns an element at `index`

        Parameters:
        - index: Integer index (supports negative indexing)
        - preserve_order: If False, swaps the removed element with the last element 
          before popping, operating in O(1) time. If True, shifts array in O(N)

        Returns:
        - A copy of the removed element

        Raises:
        - IndexError: If `index` is out of bounds
        """
        if index < 0:
            index = self.size + index
        if not (0 <= index < self.size):
            raise IndexError("index out of range")
        elem = self._data[index].copy()
        last = self.size - 1
        if index != last:
            if preserve_order:
                self._data[index:last] = self._data[index+1:self.size]
            else:
                self._data[index] = self._data[last]
        self.size -= 1
        return elem
    
    def clear(self):
        """Reset the container to empty (capacity retained)"""
        self.size = 0

    def resize(self, nsize):
        """Resizes the buffer"""
        nsize = max(0, int(nsize))
        self._ensure_cap(nsize)
        self.size = nsize

    def to_numpy(self) -> np.ndarray:
        """Return a contiguous copy of the active data as an ndarray"""
        return self.data.copy()
    
    def copy(self) -> "NumpyVector":
        """Return a deep copy"""
        return NumpyVector(self.elem_shape, self.dtype, initial=self.data)

    def shrink_to_fit(self):
        """Reduces underlying capacity to match the current size"""
        if self.size == self.cap:
            return
        data = np.empty((self.size, *self.elem_shape), dtype=self.dtype)
        data[:self.size] = self._data[:self.size]
        self._data = data
