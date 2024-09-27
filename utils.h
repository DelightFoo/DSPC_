//utils.h
#pragma once

__device__ inline int clamp(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}