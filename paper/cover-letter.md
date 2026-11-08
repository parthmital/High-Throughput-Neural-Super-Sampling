Dear Editor-in-Chief,

I submit the manuscript "One network for renders and video: vendor-neutral neural temporal super-resolution and low-latency frame generation trained on commodity cloud GPUs" for consideration as an original research article in the Journal of Real-Time Image Processing. It has not been published and is not under review elsewhere.

The manuscript presents NeuralSS, an open system with one temporal super-resolution network and one frame-generation network, each serving both game-engine renders with motion vectors and plain video without them, trained on two NVIDIA T4 GPUs from public datasets.

Real-time issues addressed:

1. **Cost-tiered architectures.** Every network has tiers sized in multiply-accumulates per 1080p output frame (6.2 to 156 GMAC for super-resolution, 9.4 to 77 GMAC for frame generation), built only from convolutions, pixel shuffles and bilinear warps so they fuse and export to ONNX for vendor-neutral runtimes.
2. **Latency rather than frame rate.** Frame generation supports extrapolation, which holds no rendered frame back, and a late-latched camera prior. A pacing model reports displayed frame rate, game latency and camera latency for interpolation, extrapolation and late-latched extrapolation.
3. **Work splitting.** The super-resolution motion estimator runs only when the engine supplies no motion vectors, and the frame-generation encoder runs once per real frame rather than per generated frame.
4. **Measured timings and their gap to budgets.** We report T4 latencies per tier and resolution and state plainly that, in eager execution, every tier except the smallest super-resolution tier exceeds a 16.7 ms frame budget, with the analysis of why and what runtime work remains.

All code, notebooks, metrics and weights are public. The author declares no competing interests.

Sincerely,
Parth Mital
School of Computer Science and Engineering, Vellore Institute of Technology, Vellore 632014, Tamil Nadu, India
parth.mital2023@vitstudent.ac.in
