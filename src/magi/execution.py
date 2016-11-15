from actions.Sequence import SequenceSolution, SequenceExecutableSolution
from prpy.clone import Clone
from Queue import Queue, Empty
from threading import current_thread, Thread, Lock
import logging
import sys

logger = logging.getLogger('execute_pipeline')

class AtomicValue(object):
    def __init__(self, value):
        self.lock = Lock()
        self.value = value

    def set_value(self, value):
        with self.lock:
            self.value = value

    def get_value(self):
        with self.lock:
            return self.value

class ExceptionWrapper(object):
    def __init__(self, exception):
        self.exception = exception


class TerminationRequest(Exception):
    pass


def flatten_solution(solution):
    if isinstance(solution, SequenceSolution):
        return sum([flatten_solution(s) for s in solution.solutions], [])
    else:
        return [solution]


def flatten_executable_solution(executable_solution):
    if isinstance(executable_solution, SequenceExecutableSolution):
        return sum([flatten_executable_solution(s)
                    for s in executable_solution.executable_solutions], [])
    else:
        return [executable_solution]

def worker_thread(env, is_running, lock, input_queue, output_queue, work_fn, split_fn):
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
                    e = input_value_raw.exception
                    logger.info('Stopping %s by request: %s', name, e)
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
                logger.error('Encountered error in %s thread.', name,
                    exc_info=True)

                if output_queue is not None:
                    _, e, _ = sys.exc_info()
                    output_queue.put(ExceptionWrapper(e))

                break
    finally:
        is_running.set_value(False)
        logger.info('Exiting %s thread.', name)

        if lock:
            env.Unlock()


class ExecutionEngine(object):
    def __init__(self, simulated, monitor=None):
        self.simulated = simulated
        self.monitor = monitor

    def run(self, env, plan_callback):
        with Clone(env, lock=False) as planning_env, \
             Clone(env, lock=False) as postprocessing_env:
            return self._run_internal(
                planning_env=planning_env,
                postprocessing_env=postprocessing_env,
                execution_env=env,
                plan_callback=plan_callback)

    def _run_internal(self, planning_env, postprocessing_env, execution_env, plan_callback):
        is_postprocessing_running = AtomicValue(True)
        is_execution_running = AtomicValue(True)
        solution_queue = Queue()
        executable_solution_queue = Queue()

        postprocessing_thread = Thread(
            name='post-processing', target=worker_thread,
            args=(postprocessing_env, is_postprocessing_running, True,
                solution_queue, executable_solution_queue,
                self._postprocess_callback, flatten_solution))
        execution_thread = Thread(
            name='execution', target=worker_thread,
            args=(execution_env, is_execution_running, False,
                executable_solution_queue, None,
                self._execute_callback, flatten_executable_solution))

        logger.info('Starting post-processing and execution threads.')
        postprocessing_thread.start()
        execution_thread.start()

        try:
            # Run the planner in this thread.
            with planning_env:
                return plan_callback(planning_env, solution_queue)
        except:
            # Forcefully terminate the background threads as soon as possible.
            logger.error('Encountered error while planning.', exc_info=True)
            is_postprocessing_running.set_value(False)
            is_execution_running.set_value(False)
        finally:
            # Wait for post-processing to finish.
            logger.info('Waiting for post-processing to finish.')
            solution_queue.put(ExceptionWrapper(TerminationRequest()))
            postprocessing_thread.join()

            # Wait for execution to finish.
            logger.info('Waiting for execution to finish.')
            executable_solution_queue.put(ExceptionWrapper(TerminationRequest()))
            execution_thread.join()

    def _postprocess_callback(self, env, solution):
        if self.monitor is not None:
            self.monitor.set_post_processing_action(solution.action)

        executable_solution = solution.postprocess(env)
        solution.jump(env)
        return executable_solution

    def _execute_callback(self, env, executable_solution):
        if self.monitor is not None:
            self.monitor.set_executing_action(executable_solution.solution.action)

        executable_solution.execute(env, self.simulated)


def execute_serial(env, solution, simulate, monitor=None):
    solution.postprocess(env).execute(env, simulate)


def execute_interleaved(env, full_solution, simulate, monitor=None):
    for solution in flatten_solution(full_solution):
        solution.postprocess(env).execute(env, simulate)


def execute_pipeline(env, solution, simulate, monitor=None):
    def plan_callback(_, solution_queue):
        solution_queue.put(solution)

    engine = ExecutionEngine(simulated=simulate, monitor=monitor)
    engine.run(env, plan_callback)


def plan_execute_pipeline(env, planner, action, simulate, monitor=None, timelimit=None):
    if timelimit is not None:
        def plan_callback(planning_env, solution_queue):
            return planner.plan_timed(planning_env, action, timelimit,
                    output_queue=solution_queue)
    else:
        def plan_callback(planning_env, solution_queue):
            return planner.plan_action(planning_env, action,
                    output_queue=solution_queue)

    engine = ExecutionEngine(simulated=simulate, monitor=monitor)
    engine.run(env, plan_callback)
