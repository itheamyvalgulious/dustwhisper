# Project Instructions

- Do not reduce simulation density, resolution, active region size, solver coverage, realtime budget thresholds, visual fidelity, or configured behavior to claim a performance win.
- `enginedemo` performance target: 1920x1080 viewport with full active random materials must sustain 60 FPS on the GPU path.
- Performance work must start from pass-level profiling evidence and optimize the measured hot passes directly. Do not skip solver stages or lower configuration as a substitute for optimization.
- Do not write benchmark artifacts, timing outputs, or temporary work files to `/tmp`. Use the project-local `./tmp/` directory for temporary output.
