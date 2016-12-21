import logging


def get_logger():
    metrics_logger = logging.getLogger('planning_metrics')
    return metrics_logger


def setup_logger():
    # Setup metrics logging
    from datetime import datetime
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
    Log data about a path or trajectory
    """
    logger = get_logger()

    from prpy.util import GetTrajectoryTags
    from prpy.planning.base import Tags
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
    Log timing and metadata about planning of a path or trajectory
    @param path The trajectory after postprocessing
    @param action_name The HGPC action that generated the trajectory
    """
    from prpy.planning.base import Tags
    _log_data(path, action_name, 'P', Tags.PLAN_TIME, log_metadata=True)


def log_postprocess_data(traj, action_name):
    """
    Log timing and metadata about postprocessing of a path or trajectory
    @param traj The trajectory after postprocessing
    @param action_name The HGPC action that generated the trajectory
    """
    from prpy.planning.base import Tags
    _log_data(traj, action_name, 'S', Tags.POSTPROCESS_TIME, log_metadata=True)


def log_execution_data(traj, action_name):
    """
    Log timing data about execution of a trajectory or path
    @param traj The trajectory to log
    @param action_name The HGPC action that generated the trajectory
    """
    from prpy.planning.base import Tags
    _log_data(traj, action_name, 'E', Tags.EXECUTION_TIME)
