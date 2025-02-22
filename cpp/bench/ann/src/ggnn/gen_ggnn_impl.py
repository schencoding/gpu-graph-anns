def gen_ggnn_impl(dim, kbuild, kquery, segment_size):
    lines = f"""
#include "ggnn_impl-inl.cuh"
namespace cuvs::bench {{
INSTANTIATE({dim}, {kbuild}, {kquery}, {segment_size});
}}
"""
    return lines


for dim in [96, 100, 128, 200, 784, 960]:
    for kquery in [10, 100]:
        for kbuild, segment_size in [
            (24, 32),
            (24, 64),
            (48, 32),
            (48, 64),
            (64, 64),
            (96, 64),
        ]:
            with open(f"ggnn_impl_{dim}_{kbuild}_{kquery}_{segment_size}.cu", "w") as f:
                f.write(gen_ggnn_impl(dim, kbuild, kquery, segment_size))
