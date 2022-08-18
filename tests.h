#pragma once

#include "math.h"

#ifdef _DEBUG
#define RUN_TESTS
#endif

#ifdef RUN_TESTS
#define VERIFY // stub

void runTests();
void test_matrix_initialization();
void test_mpv_matrices();
void test_operator_bitshiftleft();

#endif