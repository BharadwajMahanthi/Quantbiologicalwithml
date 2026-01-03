# OpenMC Installation Guide for Windows (Native Build)

This guide documents the complete process for installing OpenMC from source on Windows with MSVC, as successfully completed for the QuantumBioSim project.

## System Requirements

- **OS**: Windows 10/11 (64-bit)
- **Compiler**: Microsoft Visual Studio 2022 Build Tools (or full VS 2022)
- **Python**: 3.11 or higher (tested with 3.13.7)
- **CMake**: 3.10 or higher
- **Git**: For cloning the OpenMC repository
- **HDF5**: Required dependency for OpenMC
- **Disk Space**: ~10 GB (3 GB for source/build, 2 GB for nuclear data, rest for dependencies)

## Installation Steps

### 1. Install Prerequisites

#### Visual Studio Build Tools

```powershell
# Download and install VS 2022 Build Tools
# From: https://visualstudio.microsoft.com/downloads/

# Required workloads:
# - Desktop development with C++
# - Windows 10/11 SDK
```

#### CMake

```powershell
# Download from: https://cmake.org/download/
# Add to PATH during installation
```

#### HDF5

```powershell
# Option 1: Using vcpkg (recommended)
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install hdf5:x64-windows

# Set environment variable
setx HDF5_ROOT "C:\path\to\vcpkg\installed\x64-windows"

# Option 2: Download pre-built binaries
# From: https://www.hdfgroup.org/downloads/hdf5/
```

### 2. Clone OpenMC Repository

```powershell
cd C:\Users\<username>\Downloads\Quantbiologicalwithml\QuantumBioSim
git clone --recurse-submodules https://github.com/openmc-dev/openmc.git
cd openmc
```

### 3. Apply Windows Compatibility Patches

The following patches are required for MSVC compatibility:

#### Patch 1: Fix M_PI Math Constant

**Files to patch:**

- `openmc/src/external/quartic_solver.cpp`
- `openmc/src/plot.cpp`
- `openmc/src/mesh.cpp`

**Action:** Add `#define _USE_MATH_DEFINES` at line 1 (before any includes)

```cpp
#define _USE_MATH_DEFINES
#include <cmath>
// ... rest of includes
```

#### Patch 2: Fix OpenMP Member Variable Reduction

**File:** `openmc/src/random_ray/flat_source_domain.cpp`

**Location:** ~line 1031, function `count_external_source_regions()`

**Change:**

```cpp
// BEFORE (causes C7660 error)
#pragma omp parallel for reduction(+ : n_external_source_regions_)
for (int64_t sr = 0; sr < n_source_regions(); sr++) {
    if (source_regions_.external_source_present(sr)) {
        n_external_source_regions_++;
    }
}

// AFTER (use local variable)
int64_t n_ext = 0;
#pragma omp parallel for reduction(+ : n_ext)
for (int64_t sr = 0; sr < n_source_regions(); sr++) {
    if (source_regions_.external_source_present(sr)) {
        n_ext++;
    }
}
n_external_source_regions_ = n_ext;
```

**Also add includes at top of file:**

```cpp
#include <cstdint>
#include <vector>
```

#### Patch 3: Add Missing Standard Headers

**Files to patch:**

1. `openmc/include/openmc/random_ray/source_region.h`

   ```cpp
   #include <cstdint>
   #include <vector>
   #include <unordered_set>
   ```

2. `openmc/include/openmc/random_ray/flat_source_domain.h`

   ```cpp
   #include <cstdint>
   #include <vector>
   ```

3. `openmc/include/openmc/random_ray/random_ray.h`

   ```cpp
   #include <cstdint>
   #include <vector>
   ```

4. `openmc/include/openmc/random_ray/random_ray_simulation.h`

   ```cpp
   #include <cstdint>
   #include <memory>
   ```

5. `openmc/include/openmc/random_ray/linear_source_domain.h`
   ```cpp
   #include <cstdint>
   ```

### 4. Configure CMake Build

Create `install_openmc.ps1`:

```powershell
# OpenMC Native Installer (Windows/MSVC)

Write-Host "Starting OpenMC Build..." -ForegroundColor Cyan

# Create build directory
if (!(Test-Path "openmc/build")) {
    mkdir "openmc/build"
}

# Navigate to build directory
cd openmc/build

# Configure with MSVC-specific flags
cmake .. `
    -DCMAKE_BUILD_TYPE=Release `
    -DBUILD_SHARED_LIBS=OFF `
    -DOPENMC_USE_OPENMP=OFF `
    -DCMAKE_CXX_FLAGS="/permissive- /std:c++17 /Zc:__cplusplus /EHsc /wd4244 /wd4267 /wd4297"

# Build (use all cores)
Write-Host "Compiling... (This may take 10-20 minutes)" -ForegroundColor Yellow
cmake --build . --config Release --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build Failed!"
    exit 1
}

# Install Python bindings
Write-Host "Installing Python bindings..." -ForegroundColor Green
cd ..
pip install .

Write-Host "OpenMC Installation Complete!" -ForegroundColor Green
python -c "import openmc; print('OpenMC Version:', openmc.__version__)"
```

**Explanation of CMake flags:**

- `/permissive-`: Strict C++ standard compliance
- `/std:c++17`: Use C++17 standard
- `/Zc:__cplusplus`: Report correct `__cplusplus` macro value
- `/EHsc`: Enable C++ exception handling
- `/wd4244 /wd4267 /wd4297`: Suppress harmless cross-platform conversion warnings
- `OPENMC_USE_OPENMP=OFF`: Disable OpenMP (MSVC has compatibility issues)

### 5. Run the Build

```powershell
# Execute the build script
.\install_openmc.ps1

# Build takes 10-20 minutes
# On success, you'll see: "OpenMC Version: 0.15.4.dev26+..."
```

### 6. Download Nuclear Cross Section Data

**Option 1: Using Python Script (Recommended)**

```powershell
# Copy the download_nuclear_data.py script to your project
python download_nuclear_data.py

# Downloads ~1.6 GB compressed file
# Extracts to: C:\Users\<username>\.openmc\endfb-viii.0\
# Takes 5-15 minutes depending on connection
```

**Option 2: Manual Download**

1. Visit: https://openmc.org/official-data-libraries/
2. Download ENDF/B-VIII.0 (HDF5 format)
3. Extract to: `C:\Users\<username>\.openmc\endfb-viii.0\`
4. Verify `cross_sections.xml` exists in that directory

### 7. Configure Environment Variable

```powershell
# Set permanent environment variable
setx OPENMC_CROSS_SECTIONS "C:\Users\<username>\.openmc\endfb-viii.0\cross_sections.xml"

# Restart terminal for changes to take effect
```

### 8. Verification

```powershell
# Test 1: Import OpenMC
python -c "import openmc; print(f'OpenMC {openmc.__version__}')"

# Test 2: Check cross sections
python -c "import openmc; print(openmc.config['cross_sections'])"

# Test 3: Run integration test
python test_openmc_integration.py

# All tests should pass without warnings
```

## Python Virtual Environment Setup

### Create Virtual Environment (Python 3.11+)

```powershell
# Create venv with Python 3.13
python -m venv .venv_evo

# Activate
.venv_evo\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install OpenMC
pip install ./openmc
```

### Requirements File

```txt
numpy
pandas
scipy
matplotlib
statsmodels
faker
deap
torch
pygad
```

## Troubleshooting

### Common Issues

#### Issue 1: `M_PI` undeclared identifier

**Error:**

```
error C2065: 'M_PI': undeclared identifier
```

**Fix:** Add `#define _USE_MATH_DEFINES` at the very top of the file (line 1), before any `#include` statements.

#### Issue 2: OpenMP reduction error C7660

**Error:**

```
error C7660: 'member_var_': only a variable or static data member can be used
```

**Fix:** Use a local variable for the reduction, then assign to member variable after the parallel region.

#### Issue 3: Unknown type name 'int64_t' or 'vector'

**Error:**

```
error: unknown type name 'int64_t'
error: no template named 'vector'
```

**Fix:** Add explicit includes:

```cpp
#include <cstdint>
#include <vector>
```

#### Issue 4: Cross sections warning

**Warning:**

```
UserWarning: Path 'C:\...\cross_sections.xml' does not exist.
```

**Fix:**

1. Download nuclear data using `python download_nuclear_data.py`
2. Set `OPENMC_CROSS_SECTIONS` environment variable
3. Restart terminal

#### Issue 5: Python version mismatch

**Error:**

```
ERROR: Package 'openmc' requires a different Python: 3.10.11 not in '>=3.11'
```

**Fix:**

1. Install Python 3.11 or higher
2. Recreate virtual environment
3. Reinstall dependencies

## Build Artifacts

After successful build, you'll have:

```
openmc/
├── build/
│   ├── libopenmc.vcxproj       # C++ library
│   ├── lib/Release/
│   │   ├── openmc.lib
│   │   ├── fmt.lib
│   │   └── pugixml.lib
│   └── bin/Release/            # Executables (optional)
├── openmc/                     # Python package
│   ├── __init__.py
│   ├── filter.py
│   └── ...
└── setup.py                    # Python installer

C:\Users\<username>\.openmc\
└── endfb-viii.0/
    ├── cross_sections.xml      # Nuclear data index
    ├── 1001.h5                 # Hydrogen-1 data
    └── ...                     # ~400 nuclide files
```

## Performance Notes

- **Build Time**: 10-20 minutes (depends on CPU cores)
- **Nuclear Data Download**: 5-15 minutes (depends on internet speed)
- **Disk Space**: ~10 GB total
- **RAM**: 8 GB minimum recommended for compilation

## Integration with QuantumBioSim

The `src/real_openmc_model.py` module provides the interface:

```python
from src.real_openmc_model import RealOpenMCModel

# Create model
model = RealOpenMCModel()

# Generate XML files
model.build_model()

# Run simulation (requires cross sections)
heating = model.run()
```

## Additional Resources

- **OpenMC Documentation**: https://docs.openmc.org
- **Build Guide**: https://docs.openmc.org/en/stable/usersguide/install.html
- **Nuclear Data**: https://openmc.org/official-data-libraries/
- **GitHub**: https://github.com/openmc-dev/openmc
- **Support**: https://openmc.discourse.group/

## Version Information

Successfully tested with:

- Windows 11 (Build 26200)
- Python 3.13.7
- MSVC 14.34.31933 (Visual Studio 2022)
- CMake 3.28+
- OpenMC 0.15.4.dev26
- HDF5 1.14+

## Acknowledgments

This installation guide was developed during the QuantumBioSim project integration, which required native Windows compilation of OpenMC for radiation shielding simulations in biological evolution modeling.

---

**Last Updated**: 2026-01-02  
**Tested Platform**: Windows 11 x64  
**Build Status**: ✅ Successful
