#include "rambo_alpaka.h"
#include "integrator.h"
#include "kernels.h"

void runRamboAlpaka() {
    // Initialize Alpaka environment
    alpaka::dev::Dev<alpaka::dev::Acc<alpaka::dev::DevProp<alpaka::dev::DevType::Cpu>>> device;
    alpaka::queue::Queue<alpaka::dev::Acc<alpaka::dev::DevProp<alpaka::dev::DevType::Cpu>>> queue(device);

    // Create an instance of the Integrator
    Integrator integrator;

    // Set up parameters for integration
    integrator.setParameters(/* parameters */);

    // Execute the integration using Alpaka
    integrator.execute(queue);

    // Perform any necessary cleanup
}

int main() {
    runRamboAlpaka();
    return 0;
}