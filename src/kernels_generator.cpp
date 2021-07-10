#include "kernels_generator.h"

HALIDE_REGISTER_GENERATOR(halide_gnne_matmul, halide_gnne_matmul)

HALIDE_REGISTER_GENERATOR(halide_matmul, halide_matmul)

HALIDE_REGISTER_GENERATOR(halide_gnne_conv2d, halide_gnne_conv2d)

HALIDE_REGISTER_GENERATOR(halide_conv2d, halide_conv2d)

HALIDE_REGISTER_GENERATOR(halide_conv2d_depthwise, halide_conv2d_depthwise)
