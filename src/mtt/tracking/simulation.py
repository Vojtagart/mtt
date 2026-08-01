import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import imageio
import time
from typing import Tuple, Optional, List, Union, Dict
from .radar import Radar
from .traj_map import TrajMap
from .tracker import Tracker
from .plotter import Plotter
from .gospa import gospa
from murty import AssignmentWorkers


def make_gif(fig: plt.Figure, tmap: TrajMap, radar: Radar, tracker: Tracker,
             plotter: Plotter, path_name: str, fps: float = 5.0,
             calc_gospa: bool = False, gospa_c: float = 100., gospa_p: float = 2.,
             state_pos: Union[slice, list, np.ndarray] = slice(0, 2)) -> Dict[str, List[float]]:
    """
    Simulates the tracking scenario and saves the output directly to a GIF

    Parameters:
    - fig: Matplotlib Figure associated with the plotter
    - tmap: TrajMap instance containing the generated trajectory timeline
    - radar: Radar instance for generating measurements
    - tracker: Tracker instance to process the measurements
    - plotter: Plotter instance for visualization
    - path_name: File path to save the generated GIF (e.g., 'output.gif')
    - fps: Frames per second for the resulting GIF
    - calc_gospa: Whether to compute and fetch GOSPA or not
    - gospa_c: Cutoff distance for gospa
    - gospa_p: Order parameter for gospa
    - state_pos: Slice to extract position from state vector
    """
    frames = []
    ax = plotter.ax

    gospa_dict = {'gospa': [], 'gospa_loc': [], 'gospa_miss': [], 'gospa_false': []}
    W = AssignmentWorkers()

    for t, trajs in enumerate(tmap):
        ax.set_title(f"Time {t + 1}")
        dets = radar.detect(t, trajs)
        tracker.step(dets)

        plotter.plot_radar(t, radar)
        plotter.plot_trajs(t, trajs)
        plotter.plot_tracker(tracker)

        ax.legend(loc='upper right')

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        frames.append(frame)

        if calc_gospa:
            states = np.array([traj.state_at(t)[state_pos] for traj in trajs])
            ct = tracker.confirmed_tracks()
            estimates = ct[0][:, state_pos] if ct is not None else np.array([])

            gospa_all, gospa_loc, gospa_miss, gospa_false = gospa(
                states, estimates, c=gospa_c, p=gospa_p, ass_workers=W, sparse=True
            )
            gospa_dict['gospa'].append(gospa_all)
            gospa_dict['gospa_loc'].append(gospa_loc)
            gospa_dict['gospa_miss'].append(gospa_miss)
            gospa_dict['gospa_false'].append(gospa_false)
    
    imageio.mimsave(path_name, frames, fps=fps, loop=1)
    plt.close('all')
    
    return gospa_dict

def get_gospa_time(states: List[np.ndarray], dets: List[np.ndarray], tracker: Tracker, gospa_c: float = 100., gospa_p: float = 2.,
                   state_pos: Union[slice, list, np.ndarray] = slice(0, 2)) -> Tuple[Dict[str, List[float]], List[float]]:
    """
    Simulates the tracker on given tmap and computes gospa

    Parameters:
    - states: List of real states
    - dets: List of detections/measurements
    - tracker: Tracker instance to process the measurements
    - gospa_c: Cutoff distance for gospa
    - gospa_p: Order parameter for gospa
    - state_pos: Slice to extract position from state vector

    Returns:
    - Tuple[gospa_dict, times]
    """

    gospa_dict = {'gospa': [], 'gospa_loc': [], 'gospa_miss': [], 'gospa_false': []}
    times = []
    W = AssignmentWorkers()
    assert len(states) == len(dets), "Length of states and detections must match"

    for t in range(len(states)):

        start = time.perf_counter()
        tracker.step(dets[t])
        end = time.perf_counter()
        times.append(end - start)

        ct = tracker.confirmed_tracks()
        estimates = ct[0][:, state_pos] if ct is not None else np.array([])

        gospa_all, gospa_loc, gospa_miss, gospa_false = gospa(
            states[t][:, state_pos], estimates, c=gospa_c, p=gospa_p, ass_workers=W, sparse=True
        )
        gospa_dict['gospa'].append(gospa_all)
        gospa_dict['gospa_loc'].append(gospa_loc)
        gospa_dict['gospa_miss'].append(gospa_miss)
        gospa_dict['gospa_false'].append(gospa_false)

    return gospa_dict, times

def plot_gospa(gospa_data: List[Dict[str, List[float]]], labels: Optional[List[str]] = None, 
               p: float = 2.0, rms: bool = False, cols: int = 2, ignore_t0: bool = True):
    """
    Plots GOSPA scores over time for multiple trackers

    Parameters:
    - gospa_data: List of dictionaries containing the GOSPA metrics for each tracker
    - labels: Optional list of names for each tracker
    - p: The order parameter used during GOSPA calculation
    - rms: Whether to report RMS of the gospa metrics or not
    - cols: Number of columns
    """
    n_trackers = len(gospa_data)
    if labels is None:
        labels = [f"Tracker {i+1}" for i in range(n_trackers)]
        
    if len(labels) != n_trackers:
        raise ValueError("Number of labels must match number of trackers")

    rows = math.ceil(n_trackers / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for i, data in enumerate(gospa_data):
        ax = axes_flat[i]
        
        total = np.asarray(data['gospa'])
        loc = np.asarray(data['gospa_loc'])
        miss = np.asarray(data['gospa_miss'])
        fal = np.asarray(data['gospa_false'])
        time_steps = np.arange(len(total))

        if ignore_t0:
            total = total[1:]
            loc = loc[1:]
            miss = miss[1:]
            fal = fal[1:]
            time_steps = time_steps[1:]

        if rms:
            ax.plot(time_steps, total**(1/p), label='RMS GOSPA', color='black', linewidth=2)
            ax.plot(time_steps, loc**(1/p), label='RMS Localization', color='blue', linestyle='--')
            ax.plot(time_steps, miss**(1/p), label='RMS Missed', color='red', linestyle='-.')
            ax.plot(time_steps, fal**(1/p), label='RMS False', color='green', linestyle=':')
            ax.set_ylabel("Cost (Physical Units)")
        else:
            ax.plot(time_steps, total, label='Total GOSPA', color='black', linewidth=2)
            ax.plot(time_steps, loc, label='Localization', color='blue', linestyle='--')
            ax.plot(time_steps, miss, label='Missed', color='red', linestyle='-.')
            ax.plot(time_steps, fal, label='False', color='green', linestyle=':')
            ax.set_ylabel(f"Cost")
        ax.set_xlabel(f"Time step")

        ax.set_title(f"{labels[i]}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    for i in range(n_trackers, len(axes_flat)):
        fig.delaxes(axes_flat[i])
    fig.tight_layout()
    
    return fig

class MockTracker(Tracker):
    """Internal mock to feed cached historical states back to the Plotter"""
    def __init__(self, ct: Optional[Tuple[np.ndarray, np.ndarray]],
                 uct: Optional[Tuple[np.ndarray, np.ndarray]]):
        self.ct = ct
        self.uct = uct

    def step(self, dets: np.ndarray):
        pass
    
    def confirmed_tracks(self, type: int = 1) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self.ct
    
    def unconfirmed_tracks(self, type: int = 1) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self.uct

class InteractiveSimulator:
    """
    Interactive interface for the tracking simulation

    Pre-computes the entire tracking loop to allow arbitrary time-scrubbing 
    (forwards, backwards, pause, play) for optionally more than one tracker
    at a time, synchronized size by side
    """
    def __init__(self, fig: plt.Figure, tmap: TrajMap, radar: Radar, 
                 trackers: Union[Tracker, List[MockTracker], List[Union[Tracker, List[MockTracker]]]],
                 plotters: Union[Plotter, List[Plotter]], labels: Optional[Union[str, List[str]]] = None,
                 fps: float = 5., calc_gospa: bool = False, gospa_c: float = 100., gospa_p: float = 2.,
                 state_pos: Union[slice, list, np.ndarray] = slice(0, 2), estimators: Optional[List[int]] = None):
        """
        Initializes and pre-computes the interactive simulation

        Parameters:
        - fig: Matplotlib Figure
        - tmap: TrajMap instance
        - radar: Radar instance
        - trackers: List of tracker-likes. Can be:
            - Tracker. Will be fed with data generated from tmap and radar
            - List[MockTracker] list of already precomputed tracker states
        - plotters: List of Plotters, one for each tracker
        - labels: List of labels for the plots
        - fps: FPS for animation
        - calc_gospa: Whether to compute and fetch GOSPA or not
        - gospa_c: Cutoff distance for gospa
        - gospa_p: Order parameter for gospa
        - state_pos: Slice to extract position from state vector
        - estimators: List of estimator type to be used
        """
        
        self.fig = fig
        self.radar = radar
        self.fps = float(fps)
        self.state_pos = state_pos
        
        self.dets = []
        self.meas = []
        self.cluts = []
        self.trajs = []
        self.history: List[List[MockTracker]] = []
        self.gospa: List[Dict[str, List[float]]] = []
        
        # Must work also with the cpp classes
        if hasattr(trackers, 'step') and hasattr(trackers, 'confirmed_tracks') and hasattr(trackers, 'unconfirmed_tracks'):
            trackers = [trackers]
        elif isinstance(trackers, list) and len(trackers) > 0 and isinstance(trackers[0], MockTracker):
            trackers = [trackers]
        if isinstance(plotters, Plotter):
            plotters = [plotters]
        if isinstance(labels, str):
            labels = [labels]
        if isinstance(estimators, int):
            estimators = [estimators]

        if len(trackers) != len(plotters) or (labels is not None and len(labels) != len(trackers)) \
                or (estimators is not None and len(estimators) != len(trackers)):
            raise ValueError(f"Number of trackers must match the number of plotters, labels and estimators if present")
        
        self.plotters = plotters
        self.labels = labels
        self.estimators = estimators

        for t, trajs in enumerate(tmap):
            self.dets.append(self.radar.detect(t, trajs))
            self.meas.append(self.radar.dets.copy())
            self.cluts.append(self.radar.clut.copy())
            self.trajs.append(list(trajs))

        self.ndat = tmap.size

        for i, tracker in enumerate(trackers):
            if isinstance(tracker, list):
                if len(tracker) != self.ndat:
                    raise ValueError("MockTracker sequence has imcopatible lenght")
                self.history.append(tracker)
            else:
                self._precompute_tracker(tracker, i)
    
        if calc_gospa:
            W = AssignmentWorkers()
            self.gospa = [{'gospa': [], 'gospa_loc': [], 'gospa_miss': [], 'gospa_false': []} for _ in range(len(trackers))]
            self.gospa_p = gospa_p
            for t in range(self.ndat):
                states = np.array([traj.state_at(t)[self.state_pos] for traj in self.trajs[t]])
                for i in range(len(trackers)):
                    ct = self.history[i][t].confirmed_tracks(self._get_estimator(i))
                    estimates = ct[0][:, self.state_pos] if ct is not None else np.array([])
                    gospa_all, gospa_loc, gospa_miss, gospa_false = gospa(
                        states, estimates, c=gospa_c, p=gospa_p, ass_workers=W, sparse=True
                    )
                    self.gospa[i]['gospa'].append(gospa_all)
                    self.gospa[i]['gospa_loc'].append(gospa_loc)
                    self.gospa[i]['gospa_miss'].append(gospa_miss)
                    self.gospa[i]['gospa_false'].append(gospa_false)
        
        self._build_ui()
        self.cur_time = 0
        self.is_playing = False
        self._update_plot(0)

    def plot_gospa(self, rms: bool = False, cols: int = 2, ignore_t0: bool = True):
        """Generates a plot comparing GOSPA metrics"""
        if not self.gospa:
            raise RuntimeError("Gospa wasnt computed")
        return plot_gospa(self.gospa, labels=self.labels, p=self.gospa_p, rms=rms, cols=cols, ignore_t0=ignore_t0)

    def _get_estimator(self, idx: int):
        if self.estimators is None:
            return 1
        return self.estimators[idx]

    def _precompute_tracker(self, tracker, idx):
        cur = []
        for t in range(self.ndat):
            tracker.step(self.dets[t])

            ct = tracker.confirmed_tracks(self._get_estimator(idx))
            uct = tracker.unconfirmed_tracks(self._get_estimator(idx))

            ct = (ct[0].copy(), ct[1].copy()) if ct is not None else None
            uct = (uct[0].copy(), uct[1].copy()) if uct is not None else None

            cur.append(MockTracker(ct, uct))
        self.history.append(cur)

    def _build_ui(self):
        self.fig.subplots_adjust(bottom=0.25)
        
        ax_slider = self.fig.add_axes([0.15, 0.1, 0.65, 0.03])
        self.slider = Slider(ax_slider, 'Time', 0, self.ndat - 1, valinit=0, valstep=1, valfmt='%0.0f')
        self.slider.on_changed(self._on_slider_change)

        ax_prev = self.fig.add_axes([0.3, 0.025, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_prev.on_clicked(self._step_backward)

        ax_play = self.fig.add_axes([0.45, 0.025, 0.1, 0.04])
        self.btn_play = Button(ax_play, '▶/■')
        self.btn_play.on_clicked(self._toggle_play)

        ax_next = self.fig.add_axes([0.6, 0.025, 0.1, 0.04])
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self._step_forward)
        
        interval = int(1. / self.fps * 1000.)
        self.timer = self.fig.canvas.new_timer(interval=interval)
        self.timer.add_callback(self._step_forward)

    def _update_plot(self, t: int):
        self.radar.dets = self.meas[t]
        self.radar.clut = self.cluts[t]

        for i, plotter in enumerate(self.plotters):
            if self.labels is not None:
                plotter.ax.set_title(f"{self.labels[i]} - Time: {t + 1}")
            else:
                plotter.ax.set_title(f"Time: {t + 1}")

            plotter.plot_radar(t, self.radar)
            plotter.plot_trajs(t, self.trajs[t])
            mock = self.history[i][t]
            plotter.plot_tracker(mock)
            plotter.ax.legend(loc='upper right')

        self.fig.canvas.draw_idle()

    def _on_slider_change(self, val):
        t = int(val)
        if t != self.cur_time:
            self.cur_time = t
            self._update_plot(t)

    def _step_forward(self, event=None):
        if self.cur_time < self.ndat - 1:
            self.slider.set_val(self.cur_time + 1)
        elif self.is_playing:
            self._toggle_play()

    def _step_backward(self, event=None):
        if self.cur_time > 0:
            self.slider.set_val(self.cur_time - 1)

    def _toggle_play(self, event=None):
        self.is_playing = not self.is_playing
        if self.is_playing:
            if self.cur_time == self.ndat - 1:
                self.slider.set_val(0)
            self.timer.start()
        else:
            self.timer.stop()

    def to_gif(self, path_name: str, fps: Optional[int] = None):
        """Saves the simulation as GIF"""
        frames = []
        fps = float(fps) if fps is not None else self.fps
        
        orig_time = self.cur_time
        playing = self.is_playing
    
        if playing:
            self._toggle_play()

        for t in range(self.ndat):
            self._update_plot(t)
            self.fig.canvas.draw()
            frame = np.asarray(self.fig.canvas.buffer_rgba())
            frames.append(frame)
        
        imageio.mimsave(path_name, frames, fps=fps, loop=1)

        self.slider.set_val(orig_time)
        self._update_plot(orig_time)
        if playing:
            self._toggle_play()

    def show(self):
        plt.show()
