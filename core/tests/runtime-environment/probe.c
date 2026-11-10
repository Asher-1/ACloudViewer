#include <stdio.h>

extern int dependency_value(void);

int main(void) {
    const int value = dependency_value();
    fprintf(stderr, "Dependency value: %d (expected %d)\n", value, EXPECTED_VALUE);
    return value == EXPECTED_VALUE ? 0 : 1;
}
