# magi

`MAGI`(Multistep Action Graph Instantiator) lets you construct a multistep action graph and find a sequence of executable trajectories using various backtracking strategies and `PrPy` planners.

To install and run MAGI you should folow the [development environment guide] (https://www.personalrobotics.ri.cmu.edu/software/development-environment) to create your Catkin workspace. Note that these instructions assume that you followed the [machine setup guide](https://www.personalrobotics.ri.cmu.edu/software/machine-setup) on your computer or virtual machine.

In summary, you should be able to execute these commands to download and build the code:
```shell
$ cd my-workspace/src
$ touch .rosinstall
$ wstool set magi --git https://github.com/personalrobotics/magi -y
$ wstool merge https://raw.githubusercontent.com/personalrobotics/pr-rosinstalls/master/herb-minimal-sim.rosinstall
$ wstool set pr-ordata --git https://github.com/personalrobotics/pr-ordata -y
$ wstool update
$ cd ..
$ rosdep update
$ find src -name manifest.xml -delete
$ rosdep install -y -r --ignore-src --rosdistro=indigo --from-paths src
$ catkin build
$ . devel/setup.bash
```

To run an example script, try
```
rosrun magi grasp_glass_demo.py --viewer rviz
```

## Action
`Action` defines what needs to be planned. Each `Action` has `plan` method through which `MAGI` generates a `Solution`, which contains geometrically feasible path that can be `postprocess`ed to be executed on the robot. `MAGI` offers many actions, including the following:
- `MoveHandAction`: Moves hand DOFs. `OpenHandAction`, `CloseHandAction`, `GrabObjectAction`, `ReleaseObjectAction` inherit from this.
- `PlaceObjectAction`: Places an object onto another object.
- `PlanAction`: Uses `PrPy`'s `PlanTo...` methods to generate paths. `PlanToTSRAction`, `PlanEndEffectorStraight` inherit from this.
- `MoveUntilTouchAction`: Uses `PrPy` planner to move along a direction until the endeffector touches an object.


## Meta-Action 
`Action`s can be combined to create a multistep action. `MAGI` has several meta-actions. 
- `SequenceAction` takes a list of `Action` and creates one action sequence.   
- `ParallelAction` takes a list of `Action` and creates a tree where each `Action` is a branching node.
- `DisableAction`: Disables links of an object while planning an `Action`. Useful when the wrapped `Action` results in collision with the padding of an object, which many `PrPy` planners will not succeed unless the padding is disabled.

## Action Graph
An action graph is one meta-action in which all necessary sequences and branches of actions are defined. Use a combination of `Action` and meta-actions to construct an action graph for the multistep task you intend to perform. For example, an action graph for moving an object can be written as a `SequenceAction` as the following:
```python
actions = [ 
    OpenHandAction(...),
    
    # Plan to somewhere near object
    PlanToTSRAction(...), 
    
    # Move until the robot touches the object
    MoveUntilTouchAction(...)
    
    GrabObjectAction(...)
]

SequenceAction(actions, name="GrabObjectSequence")
```

## Searching through an Action Graph 
MAGI offers two backtracking strategies to search through an action graph.
- `DepthFirstPlanner` tries each non-deterministic node multiple times before backtracking.
- `RestartPlanner` restarts from the root whenever an action node fails.

If it succeeds, `Planner` returns a solution which can be executed.
```python
planner = RestartPlanner()
with env:
    solution = planner.plan_action(env, action)
execute_pipeline(env, solution)
```

