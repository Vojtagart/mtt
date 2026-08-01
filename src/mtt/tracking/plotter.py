import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Callable, Tuple
from matplotlib.patches import Ellipse, Polygon
import copy
from ..utils.artist_pool import ArtistPool
from .tracker import Tracker
from .radar import Radar
from .trajectory import Trajectory
from .airport import Airport
from ..utils.utils import get_cov_ellipse


class Plotter:
    """
    High-level plotting helper for trajectories, radar returns, and tracker estimates

    Manages Matplotlib artists and pools for efficient redrawing during animations 
    or interactive updates. Separates trajectory rendering, radar display, and 
    tracker visualization to allow flexible composition
    """

    def __init__(self, ax: plt.Axes, config: Optional[Dict[str, bool]] = None, params: Optional[Dict[str, Dict]] = None):
        """
        Creates a Plotter bound to a Matplotlib Axes

        Initializes artist pools and scatter/patch artists according to the `config`
        Prepares proxy invisible artists for legend labeling to avoid early allocation

        Parameters:
        - ax: Matplotlib Axes used for plotting
        - config: Optional mapping of feature flags to toggle visual elements
        - params: Optional mapping of Matplotlib kwargs per visual element
        """
        self.ax = ax
        self.ax.set_aspect('equal')
        self.config: Dict[str, bool] = {
            # Trajectory config:
            'traj_line': True,
            'traj_meas': False,
            'traj_arrow': True,
            'pregen_traj': False,
            'pregen_meas': False,
            # Radar config:
            'radar_base': True,
            'clutter': True,
            'cur_dets': True,
            # Tracker:
            'target': False,
            'est_traj': False,
            'cov_ellipse': True,
            'utarget': False,
            'uest_traj': False,
            'ucov_ellipse': False,
            # Airports
            'inner_airport': True,
            'edge_airport': True
        }
        self.params: Dict[str, Dict] = {
            # Trajectory config:
            'traj_line': {'color':'gray', 'linestyle':'--', 'label':'trajectory'},
            'traj_meas': {'color':'lightblue', 'marker':'.', 'label':'measurements'},
            'traj_arrow': {'closed':True, 'facecolor':'black', 'edgecolor':'none', 'zorder':10, 'arrow_scale':12.},
            # Radar config:
            'clutter': {'marker':'x', 'color':'gray', 'zorder':0, 'alpha':0.5, 'label': 'clutter'},
            'cur_dets': {'color':'blue', 'marker':'.', 'zorder':8, 'label': 'cur. dets.'},
            'radar_base': {'size': None, 'circles': 2},
            # Tracker:
            'target': {'color':'red', 'marker':'.', 'label':'target'},
            'est_traj': {'color':'red', 'label':'track'},
            'cov_ellipse': {'alpha':0.5, 'facecolor':'orange', 'edgecolor':'red', 'label':'cov. ell.', 'n_std':2.},
            'utarget': {'color':'green', 'marker':'.', 'label':'unconf. target'},
            'uest_traj': {'color':'gray', 'label':'unconf. track'},
            'ucov_ellipse': {'alpha':0.3, 'facecolor':'green', 'edgecolor':'none', 'label':'unconf. cov. ell.', 'n_std':2.},
            # Airports
            'inner_airport': {'alpha':0.5, 'facecolor': 'yellow', 'edgecolor':'black', 'linestyle':'--', 'n_std':2.},
            'edge_airport': {'alpha':0.5, 'facecolor': 'none', 'edgecolor':'black', 'linestyle':'--', 'n_std':2.},
        }

        if config:
            self.config.update(copy.deepcopy(config))
        if params:
            for k, v in params.items():
                if k in self.params:
                    self.params[k].update(copy.deepcopy(v))
                else:
                    self.params[k] = copy.deepcopy(v)

        self.params['arrow_scale'] = self.params['traj_arrow'].pop('arrow_scale', 12.)
        self.params['ell_std'] = self.params['cov_ellipse'].pop('n_std', 1.)
        self.params['uell_std'] = self.params['ucov_ellipse'].pop('n_std', 1.)
        self.params['inner_airport_std'] = self.params['inner_airport'].pop('n_std', 1.)
        self.params['edge_airport_std'] = self.params['edge_airport'].pop('n_std', 1.)

        self.proxy = []
        self._label_proxy()

        self.traj_arts = [
            ArtistPool(ax, lambda x: x.plot([], [], **self.params['traj_line'])[0]) if self.config['traj_line'] else None,
            ArtistPool(ax, lambda x: x.scatter([], [], **self.params['traj_meas'])) if self.config['traj_meas'] else None,
            ArtistPool(ax, lambda x: x.add_patch(Polygon(np.zeros((4, 2)), **self.params['traj_arrow']))) if self.config['traj_arrow'] else None
        ]
        self.radar_arts = [
            ax.scatter([], [], **self.params['clutter']) if self.config['clutter'] else None,
            ax.scatter([], [], **self.params['cur_dets']) if self.config['cur_dets'] else None
        ]
        self.tracker_arts = [
            ax.scatter([], [], **self.params['target']) if self.config['target'] else None,
            ArtistPool(ax, lambda x: x.plot([], [], **self.params['est_traj'])[0]) if self.config['est_traj'] else None,
            ArtistPool(ax, lambda x: x.add_patch(Ellipse((0,0),width=0,height=0,angle=0, **self.params['cov_ellipse']))) if self.config['cov_ellipse'] else None
        ]
        self.utracker_arts = [
            ax.scatter([], [], **self.params['utarget']) if self.config['utarget'] else None,
            ArtistPool(ax, lambda x: x.plot([], [], **self.params['uest_traj'])[0]) if self.config['uest_traj'] else None,
            ArtistPool(ax, lambda x: x.add_patch(Ellipse((0,0),width=0,height=0,angle=0, **self.params['ucov_ellipse']))) if self.config['ucov_ellipse'] else None
        ]
        self.airport_arts = [[], []]

    def plot_trajs(self, time: int, trajs: List[Trajectory]):
        """
        Renders a list of trajectories at a given time

        Parameters:
        - time: Integer time index to render
        - trajs: List of `Trajectory` instances
        """
        n = len(trajs)
        for pool in self.traj_arts:
            if pool is not None:
                pool.acquire(n)
        for i in range(n):
            x_line = self.traj_arts[0][i] if self.traj_arts[0] else None
            y_line = self.traj_arts[1][i] if self.traj_arts[1] else None
            arrow  = self.traj_arts[2][i] if self.traj_arts[2] else None
            
            trajs[i].plot(time, arrow_scale=self.params['arrow_scale'], x_line=x_line, y_line=y_line, arrow=arrow,
                          pregen_traj=self.config['pregen_traj'], pregen_meas=self.config['pregen_meas'])

    def plot_radar(self, time: int, radar: Radar):
        """
        Draws the radar static base and updates clutter/detection scatter artists

        Parameters:
        - time: Integer time (used to draw base only on first call at t==0)
        - radar: `Radar` instance providing the detection outputs
        """
        if time == 0 and self.config['radar_base']:
            radar.plot_base(self.ax, circles=self.params['radar_base']['circles'], size=self.params['radar_base']['size'])
        radar.plot(self.radar_arts[0], self.radar_arts[1])

    def plot_tracker(self, tracker: Tracker, pos_idx: Tuple[int, int] = (0, 1)):
        """
        Updates artists to visualize tracker state

        Parameters:
        - tracker: `Tracker` instance providing `confirmed_tracks()` and `unconfirmed_tracks()`
        - pos_idx: Tuple of two integers specifying which state vector indices 
          correspond to the 2D (x, y) position. Defaults to (0, 1)

        Raises:
        - ValueError: If confirmed tracks are not a tuple of two arrays
        """
        pos_idx = list(pos_idx)
        cov_ix = np.ix_(pos_idx, pos_idx)

        def _plot(tracks, arts, n_std):
            n = 0 if tracks is None else len(tracks[0])
            if arts[2] is not None:
                arts[2].acquire(n)
                for i in range(n):
                    mu, cov = tracks[0][i], tracks[1][i]
                    mu, cov = mu[pos_idx], cov[cov_ix]
                    ellipse_art = arts[2][i]
                    w, h, angle = get_cov_ellipse(cov, n_std=n_std)
                    ellipse_art.set_width(w)
                    ellipse_art.set_height(h)
                    ellipse_art.set_angle(angle)
                    ellipse_art.set_center((mu[0], mu[1]))

            if arts[0] is not None:
                if n > 0:
                    arts[0].set_offsets(tracks[0][:, pos_idx])
                else:
                    arts[0].set_offsets(np.empty((0, 2)))

        ct = tracker.confirmed_tracks()
        uct = tracker.unconfirmed_tracks()

        # fix for the PMBM
        if ct is not None and len(ct) > 2:
            ct = ct[:2]
        if uct is not None and len(uct) > 2:
            uct = uct[:2]

        _plot(ct, self.tracker_arts, self.params['ell_std'])
        _plot(uct, self.utracker_arts, self.params['uell_std'])

    def plot_airports(self, inner_airports: Optional[List[Airport]] = None, edge_airports: Optional[List[Airport]] = None, pos_idx: Tuple[int, int] = (0, 1)):
        """
        Plots active airport locations on the map

        Designed to be called once at the start of the plotting process for speed

        Parameters:
        - inner_airports: Classical airports inside the FOV
        - edge_airports: Artificial airports at the edge of the FOV
        - pos_idx: Tuple of two integers specifying which state vector indices 
          correspond to the 2D (x, y) position. Defaults to (0, 1)
        """
        if inner_airports is not None:
            for x in inner_airports:
                ell = self.ax.add_patch(Ellipse((0,0),width=0,height=0,angle=0, **self.params['inner_airport']))
                x.plot(ell, n_std=self.params['inner_airport_std'], pos_idx=pos_idx)
                self.airport_arts[0].append(ell)

        if edge_airports is not None:
            for x in edge_airports:
                ell = self.ax.add_patch(Ellipse((0,0),width=0,height=0,angle=0, **self.params['edge_airport']))
                x.plot(ell, n_std=self.params['edge_airport_std'], pos_idx=pos_idx)
                self.airport_arts[1].append(ell)

    def _label_proxy(self):
        """
        Prepares invisible proxy artists used solely for legend labeling

        Creates zero-size/invisible artists for keys whose `params` include a 
        `label` to enforce legend visibility without allocating real artists prematurely
        """
        line_fn = lambda x: self.ax.plot([], [], **x)[0]
        scatter_fn = lambda x: self.ax.scatter([], [], **x)
        ellipse_fn = lambda x: self.ax.add_patch(Ellipse((0,0),width=0,height=0,angle=0, **x))
        
        self._ext_set_label('traj_line', line_fn)
        self._ext_set_label('traj_meas', scatter_fn)
        self._ext_set_label('target', scatter_fn)
        self._ext_set_label('est_traj', line_fn)
        self._ext_set_label('cov_ellipse', ellipse_fn)
        self._ext_set_label('utarget', scatter_fn)
        self._ext_set_label('uest_traj', line_fn)
        self._ext_set_label('ucov_ellipse', ellipse_fn)
    
    def _ext_set_label(self, key: str, plot_fn: Callable):
        """
        Creates an invisible artist with a label for legend grouping

        Parameters:
        - key: Configuration key matching `self.params`
        - plot_fn: Callable to instantiate the appropriate Matplotlib artist
        """
        if self.config[key]:
            params = self.params[key]
            label = params.get('label', None)
            if label is not None:
                art = plot_fn(params)
                self.proxy.append(art)
                params.pop('label')
