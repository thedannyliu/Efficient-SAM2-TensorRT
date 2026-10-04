# Jetson Thor deployment guide

The original all-in-one guide is split by task. Commands remain relative to
the repository root and retain the recorded Thor environment assumptions.

| Step | Guide |
| --- | --- |
| 1. Install dependencies, pin the environment, obtain model inputs | [Setup](thor/setup.md) |
| 2. Export graphs, build engines and apply accuracy gates | [Export and validation](thor/export.md) |
| 3. Build the native runtime and run video/camera input | [ROS and camera](thor/camera.md) |
| 4. Record timing scopes, test matrix and provenance | [Measurement](thor/measurement.md) |
| 5. Diagnose failures and check acceptance | [Troubleshooting](thor/troubleshooting.md) |

For a shorter operational sequence, use the [runbook](thor_runbook.md).
For implementation changes, start with the [architecture](architecture.md).
[Historical optimizations](thor_optimization_experiments.md) are evidence for
specific runs, not default settings or a replacement for target validation.
