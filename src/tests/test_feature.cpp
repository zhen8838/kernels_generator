#include "hkg/export/target.h"
#include <gtest/gtest.h>

TEST(test, feature)
{
    std::cout << internal::host_target << std::endl;
}