Wavefront Path Tracer

This is where I will take my cpu and gpu versions of my path tracer
and convert them to wavefront tracers.

At this starting point, I have developed a GPU and CPU version of a ray tracer
based on Peter Shirley's excellent first book (though I implemented a BVH tree
from book 2).

Progress:
- split megakernel into generate_rays, extend, shade, and miss kernels

To do:
- implement extension rays
- currently don't understand the workgroup sizing
- split shade into by-material shade kernels
- split rendering of image into chunks so that the buffers aren't so big

Future updates:
- add other geometric shapes
- add textures
- start loading in obj files for cooler scenes to render
- start increasing complexity by implementing some features from Physically Based Rendering