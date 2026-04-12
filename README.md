# WebGPU SLAM - a cross-platform performant solution for navigation and spatial reasoning

## TODO:
- fill pyramid pass is dispatching extra workgroups due to not considering the decreasing texture size of subsequent levels of detail
- implement zero-ing out memory after use in gpu buffer bindings 
    - may be only needed in debug builds
- implement debug logs for assigning / unassigning gpu buffer bindings
