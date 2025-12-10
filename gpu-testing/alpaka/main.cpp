#include <alpaka/alpaka.hpp>
#include "rambo_alpaka.h"

int main(int argc, char **argv) {
    Dev device = alpaka::getDevByIdx<Dev>(0);
    Queue queue(device);
    ramboAlpakaMain(device, queue);
    return 0;
}