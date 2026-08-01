import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from typing import Optional, Union

class Model:
    """
    Model for tracking objects using a linear Gaussian state-space model

    - `F`: State transition matrix (s,s)
    - `H`: Measurement matrix (m,s)
    - `Q`: Process (plant) noise covariance (s,s)
    - `R`: Measurement noise covariance (m,m)

    Typical usage:
    ```py
    model = Model.CVM(dt=0.1, q=0.5, r=1.0, dim=2)
    ```
    """

    def __init__(self, F: ArrayLike, H: ArrayLike, Q: ArrayLike, R: ArrayLike, dtype: DTypeLike = np.float64,
                 state_pos: Optional[Union[slice, list, np.ndarray]] = None,
                 state_vel: Optional[Union[slice, list, np.ndarray]] = None,
                 meas_pos: Optional[Union[slice, list, np.ndarray]] = None):
        """
        Initializes the linear Gaussian state-space model

        Parameters:
        - F: State transition matrix of shape (s, s)
        - H: Measurement matrix of shape (m, s). Maps state to measurement space
        - Q: Process noise covariance matrix of shape (s, s)
        - R: Measurement noise covariance matrix of shape (m, m)
        - dtype: NumPy dtype used to cast the provided matrices (default: np.float64)
        - state_pos: Position's position in the state vector
        - state_vel: Velocity's position in the state vector
        - meas_pos: Position's position in the measurement vector

        Raises:
        - ValueError: If matrix shapes are incompatible or F is not square
        """
        self.dtype = dtype = np.dtype(dtype)

        self.F = np.asarray(F, dtype=self.dtype).copy()
        self.H = np.asarray(H, dtype=self.dtype).copy()
        self.Q = np.asarray(Q, dtype=self.dtype).copy()
        self.R = np.asarray(R, dtype=self.dtype).copy()

        if self.F.ndim != 2 or self.F.shape[0] != self.F.shape[1]:
            raise ValueError("F must be square matrix")
        s = self.F.shape[0]
        if self.Q.shape != (s, s):
            raise ValueError("Q must match F shape (s,s)")
        if self.H.ndim != 2 or self.H.shape[1] != s:
            raise ValueError("H must have shape (m,s)")
        m = self.H.shape[0]
        if self.R.shape != (m, m):
            raise ValueError("R must be (m,m) matching H rows")
        
        # Ensure symmetry
        self.Q = ((self.Q + self.Q.T) / 2.).astype(self.dtype)
        self.R = ((self.R + self.R.T) / 2.).astype(self.dtype)

        # ColMajor F and H for passing to colMajor Eigen in C++
        # Covariances can be passed as rowMajor since they are symmetric
        self.F_CM = np.asfortranarray(self.F)
        self.H_CM = np.asfortranarray(self.H)

        self.state_pos = state_pos
        self.state_vel = state_vel
        self.meas_pos = meas_pos

    @property
    def state_dim(self) -> int:
        """
        The full state vector dimension

        Returns:
        - int: Number of elements in the state vector (size of F)
        """
        return self.F.shape[0]
    
    @property
    def meas_dim(self) -> int:
        """
        The measurement vector dimension

        Returns:
        - int: Number of elements in the measurement vector (rows of H)
        """
        return self.H.shape[0]
    
    def get_pos(self, state: np.ndarray) -> Optional[np.ndarray]:
        """Extracts position from the state vector(s)"""
        if self.state_pos is None:
            return None
        return state[..., self.state_pos]

    def get_vel(self, state: np.ndarray) -> Optional[np.ndarray]:
        """Extracts velocity from the state vector(s)"""
        if self.state_vel is None:
            return None
        return state[..., self.state_vel]

    def get_meas_pos(self, meas: np.ndarray) -> Optional[np.ndarray]:
        """Extracts position from the measurement vector(s)"""
        if self.meas_pos is None:
            return None
        return meas[..., self.meas_pos]
        
    @classmethod
    def CVM(cls, dt: float = 1.0, q: float = 1.0, r: float = 1.0, dim: int = 2, dtype: DTypeLike = np.float64):
        """
        Constructs a Constant Velocity Model (CVM)

        Builds a standard discrete-time constant-velocity linear model
        The state vector ordering is assumed to be `[pos, vel]`

        Parameters:
        - dt: Time step (delta t) between discrete updates
        - q: Process noise standard deviation
        - r: Measurement noise standard deviation
        - dim: Spatial dimension (e.g., 2 for 2D, 3 for 3D)
        - dtype: NumPy dtype for the generated matrices

        Returns:
        - Model: Configured instance with (F, H, Q, R) matrices
        """
        dt = float(dt)
        dtype = np.dtype(dtype)
        dim = int(dim)

        I = np.eye(dim, dtype=dtype)
        Z = np.zeros((dim, dim), dtype=dtype)

        F = np.block([
            [I, dt * I],
            [Z, I     ]
        ])

        Q = q**2 * np.block([
            [(dt**3 / 3.) * I, (dt**2 / 2.) * I],
            [(dt**2 / 2.) * I, dt * I          ]
        ])

        H = np.eye(dim, 2 * dim, dtype=dtype)
        R = r**2 * I

        return cls(F, H, Q, R, dtype=dtype,
                   state_pos=slice(0, dim),
                   state_vel=slice(dim, 2 * dim),
                   meas_pos=slice(0, dim))
    
    @classmethod
    def CAM(cls, dt: float = 1.0, q: float = 1.0, r: float = 1.0, dim: int = 2, dtype: DTypeLike = np.float64):
        """
        Constructs a Constant Acceleration Model (CAM)

        Expands the CVM to include acceleration
        The state vector ordering is assumed to be `[pos, vel, acc]`

        Parameters:
        - dt: Time step (delta t) between discrete updates
        - q: Process noise standard deviation for the acceleration
        - r: Measurement noise standard deviation
        - dim: Spatial dimension (e.g., 2 for 2D, 3 for 3D)
        - dtype: NumPy dtype for the generated matrices
        """
        dt = float(dt)
        dtype = np.dtype(dtype)
        dim = int(dim)

        I = np.eye(dim, dtype=dtype)
        Z = np.zeros((dim, dim), dtype=dtype)

        F = np.block([
            [I, dt * I, 0.5 * (dt**2) * I],
            [Z, I,      dt * I           ],
            [Z, Z,      I                ]
        ])
        
        Q = q**2 * np.block([
            [(dt**5 / 20.0) * I, (dt**4 / 8.0) * I, (dt**3 / 6.0) * I],
            [(dt**4 / 8.0) * I,  (dt**3 / 3.0) * I, (dt**2 / 2.0) * I],
            [(dt**3 / 6.0) * I,  (dt**2 / 2.0) * I, dt * I           ]
        ])

        H = np.eye(dim, 3 * dim, dtype=dtype)
        R = r**2 * I

        return cls(F, H, Q, R, dtype=dtype,
                   state_pos=slice(0, dim),
                   state_vel=slice(dim, 2 * dim),
                   meas_pos=slice(0, dim))

    @classmethod
    def CTM(cls, omega: float, dt: float = 1.0, q: float = 1.0, r: float = 1.0, dtype: DTypeLike = np.float64):
        """
        Constructs a 2D Coordinated Turn Model (CTM) with a known, fixed turn rate

        The state vector ordering is `[x, y, vx, vy]`. Because this is a strictly 
        linear model class, the turn rate `omega` must be provided as a constant

        Parameters:
        - omega: Known turn rate in radians per second
        - dt: Time step (delta t)
        - q: Process noise standard deviation
        - r: Measurement noise standard deviation
        - dtype: NumPy dtype for the generated matrices
        """
        dt = float(dt)
        omega = float(omega)
        dtype = np.dtype(dtype)

        if np.abs(omega) < 1e-8:
            return cls.CVM(dt=dt, q=q, r=r, dim=2, dtype=dtype)

        sin_w_dt = np.sin(omega * dt)
        cos_w_dt = np.cos(omega * dt)

        F = np.array([
            [1.0, 0.0,  sin_w_dt / omega,        -(1.0 - cos_w_dt) / omega],
            [0.0, 1.0,  (1.0 - cos_w_dt) / omega, sin_w_dt / omega        ],
            [0.0, 0.0,  cos_w_dt,                -sin_w_dt                ],
            [0.0, 0.0,  sin_w_dt,                 cos_w_dt                ]
        ], dtype=dtype)

        I = np.eye(2, dtype=dtype)
        Q = q**2 * np.block([
            [(dt**3 / 3.0) * I, (dt**2 / 2.0) * I],
            [(dt**2 / 2.0) * I, dt * I           ]
        ])

        H = np.eye(2, 4, dtype=dtype)
        R = r**2 * I

        return cls(F, H, Q, R, dtype=dtype,
                   state_pos=slice(0, 2),
                   state_vel=slice(2, 4),
                   meas_pos=slice(0, 2))
