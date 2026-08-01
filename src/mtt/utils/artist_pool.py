import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from itertools import islice
from typing import List, Callable, Iterator, Union


class ArtistPool:
    """
    Reusable pool of Matplotlib artist objects

    Maintains an internal list of artists created by a user-supplied 
    `factory(ax)` callable. It avoids reallocation and excessive 
    removal/creation overhead during animations or repeated redraws 
    (highly beneficial for blitting). The array is partitioned in-place:
    `artists[:self.active]` are visible, and `artists[self.active:]` are idle
    """

    def __init__(self, ax: plt.Axes, factory: Callable[[plt.Axes], Artist], initial: int = 0):
        """
        Creates an ArtistPool and provisions initial capacity

        Parameters:
        - ax: Matplotlib `Axes` instance passed to the `factory`
        - factory: Callable `factory(ax) -> Artist` that creates a Matplotlib artist
        - initial: Number of artists to pre-create (default 0)
        """
        self.ax = ax
        self.factory = factory
        self.arts: List[Artist] = []
        self.active = 0
        self.reserve(initial)

    def __len__(self) -> int:
        """Total number of artists currently allocated in the pool"""
        return len(self.arts)
    
    def active_artists(self) -> List[Artist]:
        """
        Returns a slice of currently active (visible) artists

        Returns:
        - List[Artist]: Active artist objects in their creation order
        """
        return self.arts[:self.active]
    
    def reserve(self, n: int):
        """
        Ensures at least `n` artists are allocated in the pool

        Calls the factory repeatedly until the pool contains `n` entries
        Newly created artists are initialized as invisible

        Parameters:
        - n: Target allocation size
        """
        n = int(n)
        while len(self.arts) < n:
            self.arts.append(self._make_artist())

    def acquire(self, n: int) -> List[Artist]:
        """
        Makes exactly `n` artists active and returns them

        Adjusts visibility so that indices `[:n]` are visible and the remainder 
        are hidden. Allocates more artists automatically if `n` exceeds current capacity

        Parameters:
        - n: Desired number of active artists

        Returns:
        - The list of active artist objects
        """
        n = int(n)
        self.reserve(n)

        for i in range(self.active, n):
            self.arts[i].set_visible(True)

        for i in range(n, self.active):
            self.arts[i].set_visible(False)
        self.active = n
        return self.active_artists()

    def release_all(self):
        """
        Removes all allocated artists from the axes and clears the pool

        Calls each artist's `.remove()` method and empties internal storage
        Resets active count to zero
        """
        for x in self.arts:
            x.remove()
        self.arts.clear()
        self.active = 0

    def __getitem__(self, idx: Union[int, slice]) -> Artist:
        """
        Index into the active artists

        Raises:
        - IndexError: if index is outside the active range
        """
        if isinstance(idx, slice):
            return self.arts[:self.active][idx]
        if idx < 0:
            idx = self.active + idx
        if not (0 <= idx < self.active):
            raise IndexError(f"Invalid index. Expected index in range [0, {self.active}), got {idx}")
        return self.arts[idx]
    
    def __iter__(self) -> Iterator[Artist]:
        """Iterate over active artists (yields same as `active_artists()`)"""
        return islice(self.arts, 0, self.active)

    def _make_artist(self) -> Artist:
        """
        Internal helper to create an artist and default it to hidden

        Returns:
        - The newly created and hidden artist
        """
        art = self.factory(self.ax)
        art.set_visible(False)
        return art
