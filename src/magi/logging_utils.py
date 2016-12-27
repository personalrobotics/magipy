"""MAGI logging utility functions."""

from datetime import datetime
import logging

from prpy.planning.base import Tags
from prpy.util import GetTrajectoryTags

def get_logger():
    """
    Return the metrics logger.

    @return metrics logger
    """
    metrics_logger = logging.getLogger('planning_metrics')
    return metrics_logger


def setup_logger():
    """
    Configure metrics logger.

    @return metrics logger
    """
    logfile = datetime.now().strftime('trial_%Y%m%d_%H%M.log')
    metrics_logger = get_logger()
    hdlr = logging.FileHandler('%s' % logfile)
    formatter = logging.Formatter('%(asctime)s %(message)s',
                                  '%Y%m%d %H:%M:%S')  # date/time plus message
    hdlr.setFormatter(formatter)
    metrics_logger.addHandler(hdlr)
    metrics_logger.setLevel(logging.INFO)

    return metrics_logger


def _log_data(path, action_name, header, tag, log_metadata=False):
    """
    Log data about a path or trajectory.

    @param path: trajectory after postprocessing
    @param action_name: name of Action that generated the trajectory
    @param header: one-letter header for logs
    @param tag: tag to filter trajectory tags with
    @param log_metadata: True if metadata should be logged
    """
    logger = get_logger()

    path_tags = GetTrajectoryTags(path)
    log_data = [header, action_name, path_tags.get(tag, 'unknown')]
    if log_metadata:
        log_data += [
            path_tags.get(Tags.PLANNER, 'unknown'),
            path_tags.get(Tags.METHOD, 'unknown')
        ]
    logger.info(' '.join([str(v) for v in log_data]))


def log_plan_data(path, action_name):
    """
    Log timing and metadata about planning of a path or trajectory.

    @param path: trajectory after postprocessing
    @param action_name: name of Action that generated the trajectory
    """
    _log_data(path, action_name, 'P', Tags.PLAN_TIME, log_metadata=True)


def log_postprocess_data(traj, action_name):
    """
    Log timing and metadata about postprocessing of a path or trajectory.

    @param traj: trajectory after postprocessing
    @param action_name: name of Action that generated the trajectory
    """
    _log_data(traj, action_name, 'S', Tags.POSTPROCESS_TIME, log_metadata=True)


def log_execution_data(traj, action_name):
    """
    Log timing data about execution of a trajectory or path.

    @param traj: trajectory to log
    @param action_name: name of Action that generated the trajectory
    """
    _log_data(traj, action_name, 'E', Tags.EXECUTION_TIME)
