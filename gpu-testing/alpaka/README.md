# Rambo Alpaka

## Overview
Rambo Alpaka is a high-performance computing project that leverages the Alpaka library to perform complex computations efficiently on various hardware architectures. This project aims to provide an easy-to-use interface for executing parallel algorithms and simulations.

## Features
- Utilizes the Alpaka library for device-agnostic parallel computing.
- Implements integration tasks through the `Integrator` class.
- Provides kernel functions for executing computations on the device.
- Includes examples and benchmarks to demonstrate functionality and performance.

## Project Structure
```
rambo-alpaka
├── CMakeLists.txt          # Main configuration file for CMake
├── cmake                   # CMake modules for finding dependencies
│   ├── FindAlpaka.cmake
│   └── modules
│       └── FindAlpakaExtras.cmake
├── include                 # Header files for the project
│   ├── rambo_alpaka.h
│   ├── integrator.h
│   └── kernels.h
├── src                     # Source files for the project
│   ├── main.cpp
│   ├── rambo_alpaka.cpp
│   ├── integrator.cpp
│   └── kernels.cpp
├── examples                # Example applications demonstrating usage
│   └── simple-run
│       ├── CMakeLists.txt
│       └── main.cpp
├── tests                   # Unit tests for the project
│   ├── CMakeLists.txt
│   └── test_integrator.cpp
├── benchmarks              # Benchmark tests for performance measurement
│   └── CMakeLists.txt
│   └── benchmark_rambo.cpp
├── scripts                 # Utility scripts for building and running
│   ├── build.sh
│   ├── run_example.sh
│   └── ci_build.sh
├── docker                  # Docker configuration for the project
│   └── Dockerfile
├── .github                 # GitHub workflows for CI/CD
│   └── workflows
│       └── ci.yml
├── .gitignore              # Files and directories to ignore in Git
├── LICENSE                 # Licensing information
└── README.md               # Project documentation
```

## Getting Started

### Prerequisites
- CMake (version 3.10 or higher)
- Alpaka library
- A compatible C++ compiler (e.g., GCC, Clang, MSVC)

### Building the Project
1. Clone the repository:
   ```
   git clone <repository-url>
   cd rambo-alpaka
   ```

2. Create a build directory:
   ```
   mkdir build
   cd build
   ```

3. Configure the project using CMake:
   ```
   cmake ..
   ```

4. Build the project:
   ```
   make
   ```

### Running Examples
To run the simple example, navigate to the `examples/simple-run` directory and follow the build instructions in its `CMakeLists.txt`.

### Running Tests
To run the tests, navigate to the `tests` directory and follow the build instructions in its `CMakeLists.txt`.

### License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.