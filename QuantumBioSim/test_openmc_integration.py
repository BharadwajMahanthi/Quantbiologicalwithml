#!/usr/bin/env python3
"""
OpenMC Integration Test for QuantumBioSim
Tests that the OpenMC API is working and can generate model files.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_openmc_import():
    """Test that OpenMC can be imported."""
    try:
        import openmc
        logger.info(f"✓ OpenMC imported successfully (version {openmc.__version__})")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import OpenMC: {e}")
        return False

def test_model_generation():
    """Test that we can generate OpenMC XML files."""
    try:
        from src.real_openmc_model import RealOpenMCModel
        
        model = RealOpenMCModel()
        model.build_model()
        
        # Check that XML files were created
        required_files = ['materials.xml', 'geometry.xml', 'settings.xml', 'tallies.xml']
        missing = [f for f in required_files if not Path(f).exists()]
        
        if missing:
            logger.warning(f"✗ Missing XML files: {missing}")
            return False
        
        logger.info("✓ All OpenMC XML files generated successfully")
        for f in required_files:
            size = Path(f).stat().st_size
            logger.info(f"  - {f} ({size} bytes)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_sections():
    """Check if nuclear data is available."""
    import openmc
    
    try:
        # Try to get the cross sections path
        xs_path = openmc.config.get('cross_sections')
        if xs_path and Path(xs_path).exists():
            logger.info(f"✓ Cross sections found at: {xs_path}")
            return True
        else:
            logger.warning("✗ Cross sections not configured or file doesn't exist")
            logger.info("\nTo enable full Monte Carlo transport, download nuclear data:")
            logger.info("1. Visit: https://openmc.org/official-data-libraries/")
            logger.info("2. Download ENDF/B-VIII.0 library (recommended)")
            logger.info("3. Extract and set OPENMC_CROSS_SECTIONS environment variable")
            logger.info("   Example: set OPENMC_CROSS_SECTIONS=C:\\path\\to\\cross_sections.xml")
            return False
    except Exception as e:
        logger.warning(f"Could not check cross sections: {e}")
        return False

def test_openmc_executable():
    """Test if the OpenMC executable can run."""
    import subprocess
    
    try:
        result = subprocess.run(['openmc', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info(f"✓ OpenMC executable found: {version}")
            return True
        else:
            logger.warning("✗ OpenMC executable returned error")
            return False
            
    except FileNotFoundError:
        logger.warning("✗ OpenMC executable not found in PATH")
        logger.info("  Note: Python API works, but you may need to add openmc/build/bin to PATH")
        logger.info("        for running simulations via openmc.run()")
        return False
    except Exception as e:
        logger.warning(f"Could not test executable: {e}")
        return False

def main():
    logger.info("=" * 60)
    logger.info("OpenMC Integration Test for QuantumBioSim")
    logger.info("=" * 60)
    
    tests = [
        ("Import Test", test_openmc_import),
        ("Model Generation", test_model_generation),
        ("Executable Check", test_openmc_executable),
        ("Cross Sections", test_cross_sections),
    ]
    
    results = {}
    for name, test_func in tests:
        logger.info(f"\n--- {name} ---")
        results[name] = test_func()
    
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary:")
    logger.info("=" * 60)
    
    for name, passed in results.items():
        status = "PASS" if passed else "WARN"
        logger.info(f"{status:6s} | {name}")
    
    critical_tests = ["Import Test", "Model Generation"]
    critical_passed = all(results.get(t, False) for t in critical_tests)
    
    if critical_passed:
        logger.info("\n✓ OpenMC integration is working!")
        logger.info("  You can now use the real OpenMC API in QuantumBioSim.")
        logger.info("  For full transport simulations, configure cross sections.")
        return 0
    else:
        logger.error("\n✗ Critical tests failed. OpenMC integration incomplete.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
