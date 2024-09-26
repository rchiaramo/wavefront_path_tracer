# Wavefront Path Tracer

This is where I will take my cpu and gpu versions of my path tracer
and convert them to wavefront tracers.

At this starting point, I have developed a GPU and CPU version of a ray tracer
based on Peter Shirley's excellent first book (though I implemented a BVH tree
from book 2).  This is located in my path_tracer repository.

### Progress:
- split megakernel into generate_rays, extend, shade, miss, and accumulate kernels
- created kernel structure to avoid all of the initial duplicate code
- moved timestamping into kernels
- wavefront path tracer now working

### To do:
- contemplate whether this runs fast enough...doesn't seem like it
- currently don't understand the workgroup sizing
- split shade into by-material shade kernels
- split rendering of image into chunks so that the buffers aren't so big

### Future updates:
- add other geometric shapes
- add textures
- start loading in obj files for cooler scenes to render
- start increasing complexity by implementing some features from Physically Based Rendering