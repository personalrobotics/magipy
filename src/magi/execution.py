"""MAGI parallel execution."""

from Queue import Queue
from threading import current_thread, Thread, Lock
import logging
import sys

from prpy.clone import Clone

from magi.actions.Sequence import SequenceSolution, SequenceExecutableSolution

LOGGER = logging.getLogger('execute_pipeline')


class AtomicValue(object):
    """Value that is automatically locked."""

    def __init__(self, value):
        """
        @param value: value to be locked
        """
        self.lock = Lock()
        self.value = value

    def set_value(self, value):
        """
        Set the atomic value.

        @param value: new value
        """
        with self.lock:
            self.value = value

    def get_value(self):
        """
        Get the atomic value.

        @return the current value
        """
        with self.lock:
            return self.value


class ExceptionWrapper(object):
    """Exception wrapper class."""

    def __init__(self, exception):
        """
        @param exception: original exception to wrap
        """
        self.exception = exception


class TerminationRequest(Exception):
    """Exception class for termination requests."""

    pass


def flatten_solution(solution):
    """
    Flatten a SequenceSolution into a list of Solutions.

    @param solution: SequenceSolution
    @return a list of Solutions
    """
    if isinstance(solution, SequenceSolution):
        return sum([flatten_solution(s) for s in solution.solutions], [])
    else:
        return [solution]


def flatten_executable_solution(executable_solution):
    """
    Flatten an ExecutableSequenceSolution into a list of ExecutableSolutions.

    @param executable_solution: ExecutableSequenceSolution
    @return a list of ExecutableSolutions
    """
    if isinstance(executable_solution, SequenceExecutableSolution):
        return sum([
            flatten_executable_solution(s)
            for s in executable_solution.executable_solutions
        ], [])
    else:
        return [executable_solution]


def worker_thread(env, is_running, lock, input_queue, output_queue, work_fn,
                  split_fn):
    """
    Worker thread for parallel planning.

    @param env: OpenRAVE environment
    @param is_running: AtomicValue, whether work is happening
    @param lock: boolean, whether to lock the environment before working with
      this thread
    @param input_queue: Queue containing input values to do work with
    @param output_queue: Queue to put output values to
    @param work_fn: function that takes environment, input queue value and
      returns output queue value
    @param split_fn: function that takes raw input queue values and returns
      list of input queue values
    """
    name = current_thread().name
    internal_queue = []

    if lock:
        env.Lock()

    try:
        while is_running.get_value():
            # Grab items from the shared queue if the internal queue is empty.
            if not internal_queue:
                input_value_raw = input_queue.get()

                # Check if an error occurred in planning. This is indicated by
                # wrapping the underlying exception in an ExceptionWrapper.
                if isinstance(input_value_raw, ExceptionWrapper):
                    exc_input = input_value_raw.exception
                    LOGGER.info('Stopping %s by request: %s', name, exc_input)
                    break

                # Split the input_value into multiple work items.
                work_items = split_fn(input_value_raw)
                internal_queue.extend(work_items)

            input_value = internal_queue.pop(0)

            # Process the current item.
            try:
                output_value = work_fn(env, input_value)
                if output_queue is not None:
                    output_queue.put(output_value)
            except:
                LOGGER.error(
                    'Encountered error in %s thread.', name, exc_info=True)

                if output_queue is not None:
                    _, exc_input, _ = sys.exc_info()
                    output_queue.put(ExceptionWrapper(exc_input))

                break
    finally:
        is_running.set_value(False)
        LOGGER.info('Exiting %s thread.', name)

        if lock:
            env.Unlock()


class ExecutionEngine(object):
    """Execution engine."""

    def __init__(self, simulated, monitor=None):
        """
        @param simulated: flag to run in simulation
        @param monitor: ActionMonitor visualizer
        """
        self.simulated = simulated
        self.monitor = monitor

    def run(self, env, plan_callback):
        """
        Plan, postprocess, and execute.

        @param env: OpenRAVE environment
        @param plan_callback: function that takes environment and Solution
        @return result of plan_callback
        """
        with Clone(env, lock=False) as planning_env, \
             Clone(env, lock=False) as postprocessing_env:
            return self._run_internal(
                planning_env=planning_env,
                postprocessing_env=postprocessing_env,
                execution_env=env,
                plan_callback=plan_callback)

    def _run_internal(self, planning_env, postprocessing_env, execution_env,
                      plan_callback):
        """
        Helper function for planning, postprocessing, and executing.

        @param planning_env: planning OpenRAVE environment
        @param postprocessing_env: postprocessing OpenRAVE environment
        @param execution_env: execution OpenRAVE environment
        @param plan_callback: function that takes environment and Solution
        @return result of plan_callback
        """
        is_postprocessing_running = AtomicValue(True)
        is_execution_running = AtomicValue(True)
        solution_queue = Queue()
        executable_solution_queue = Queue()

        postprocessing_thread = Thread(
            name='post-processing',
            target=worker_thread,
            args=(postprocessing_env, is_postprocessing_running, True,
                  solution_queue, executable_solution_queue,
                  self._postprocess_callback, flatten_solution))
        execution_thread = Thread(
            name='execution',
            target=worker_thread,
            args=(execution_env, is_execution_running, False,
                  executable_solution_queue, None, self._execute_callback,
                  flatten_executable_solution))

        LOGGER.info('Starting post-processing and execution threads.')
        postprocessing_thread.start()
        execution_thread.start()

        try:
            # Run the planner in this thread.
            with planning_env:
                return plan_callback(planning_env, solution_queue)
        except:
            # Forcefully terminate the background threads as soon as possible.
            LOGGER.error('Encountered error while planning.', exc_info=True)
            is_postprocessing_running.set_value(False)
            is_execution_running.set_value(False)
        finally:
            # Wait for post-processing to finish.
            LOGGER.info('Waiting for post-processing to finish.')
            solution_queue.put(ExceptionWrapper(TerminationRequest()))
            postprocessing_thread.join()

            # Wait for execution to finish.
            LOGGER.info('Waiting for execution to finish.')
            executable_solution_queue.put(
                ExceptionWrapper(TerminationRequest()))
            execution_thread.join()

    def _postprocess_callback(self, env, solution):
        """
        Postprocessing worker function.

        @param env: postprocessing OpenRAVE environment
        @param solution: Solution to postprocess
        @return ExecutableSolution
        """
        if self.monitor is not None:
            self.monitor.set_post_processing_action(solution.action)

        executable_solution = solution.postprocess(env)
        solution.jump(env)
        return executable_solution

    def _execute_callback(self, env, executable_solution):
        """
        Execution worker function.

        @param env: execution OpenRAVE environment
        @param executable_solution: ExecutableSolution to execute
        """
        if self.monitor is not None:
            self.monitor.set_executing_action(
                executable_solution.solution.action)

        executable_solution.execute(env, self.simulated)


def execute_serial(env, solution, simulate):
    """
    Execute solution in serial.

    @param env: OpenRAVE environment
    @param solution: Solution to execute
    @param simulate: flag to run in simulation
    """
    solution.postprocess(env).execute(env, simulate)


def execute_interleaved(env, full_solution, simulate):
    """
    Execute interleaved solution.

    @param env: OpenRAVE environment
    @param full_solution: Solution/SequenceSolution to execute
    @param simulate: flag to run in simulation
    """
    for solution in flatten_solution(full_solution):
        solution.postprocess(env).execute(env, simulate)


def execute_pipeline(env, solution, simulate, monitor=None):
    """
    Execute solution using the parallel ExecutionEngine pipeline.

    @param env: OpenRAVE environment
    @param solution: Solution to execute
    @param simulate: flag to run in simulation
    @param monitor: ActionMonitor visualizer
    """
    def plan_callback(_, solution_queue):
        """
        Simple callback that just puts the solution in the queue.

        @param solution_queue: Queue to put solutions into
        """
        solution_queue.put(solution)

    engine = ExecutionEngine(simulated=simulate, monitor=monitor)
    engine.run(env, plan_callback)


def plan_execute_pipeline(env,
                          planner,
                          action,
                          simulate,
                          monitor=None,
                          timelimit=None):
    """
    Plan and execute solution using the parallel ExecutionEngine pipeline.

    @param env: OpenRAVE environment
    @param planner: Planner to plan with
    @param action: Action to plan
    @param simulate: flag to run in simulation
    @param monitor: ActionMonitor visualizer
    @param timelimit: time limit (seconds) for planners to plan
    """
    if timelimit is not None:

        def plan_callback(planning_env, solution_queue):
            """
            Callback that uses the timed planner.

            @param planning_env: OpenRAVE environment to plan with
            @param solution_queue: Queue to put solutions into
            """
            return planner.plan_timed(
                planning_env, action, timelimit, output_queue=solution_queue)
    else:

        def plan_callback(planning_env, solution_queue):
            """
            Callback that uses the untimed planner.

            @param planning_env: OpenRAVE environment to plan with
            @param solution_queue: Queue to put solutions into
            """
            return planner.plan_action(
                planning_env, action, output_queue=solution_queue)

    engine = ExecutionEngine(simulated=simulate, monitor=monitor)
    engine.run(env, plan_callback)
