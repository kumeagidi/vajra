#pragma once
//==============================================================================
// C headers
#include <stdio.h>
#include <sys/time.h>
#include <errno.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <cmath>
//==============================================================================
// C++ headers
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <array>
#include <memory>
#include <optional>
#include <unordered_map>
#include <set>
//==============================================================================
namespace std {
    template<>
    struct hash<std::set<int>> {
        size_t operator()(const std::set<int>& s) const {
            size_t hash = 0;
            for (int x : s) {
                hash ^= std::hash<int>{}(x) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };
}
//==============================================================================