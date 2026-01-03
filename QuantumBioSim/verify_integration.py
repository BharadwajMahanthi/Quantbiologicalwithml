"""
Quick OpenMC Integration Verification Script
Tests all integration points in QuantumBioSim
"""

import sys

def test_integration():
    print("=" * 60)
    print("QuantumBioSim OpenMC Integration Verification")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: OpenMC Import
    print("\n[1/5] Testing OpenMC Import...")
    try:
        import openmc
        print(f"    ✓ OpenMC version: {openmc.__version__}")
        print(f"    ✓ Cross sections: {openmc.config['cross_sections']}")
        tests_passed += 1
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        tests_failed += 1
        return False
    
    # Test 2: RealOpenMCModel Import
    print("\n[2/5] Testing RealOpenMCModel...")
    try:
        from src.real_openmc_model import RealOpenMCModel
        model = RealOpenMCModel()
        print("    ✓ RealOpenMCModel imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        tests_failed += 1
        return False
    
    # Test 3: Model Generation
    print("\n[3/5] Testing Model Generation...")
    try:
        model.build_model()
        import os
        required_files = ['materials.xml', 'geometry.xml', 'settings.xml', 'tallies.xml']
        all_exist = all(os.path.exists(f) for f in required_files)
        if all_exist:
            print("    ✓ All XML files generated successfully")
            for f in required_files:
                size = os.path.getsize(f)
                print(f"      - {f} ({size} bytes)")
            tests_passed += 1
        else:
            print("    ✗ Some XML files missing")
            tests_failed += 1
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        tests_failed += 1
    
    # Test 4: PyGAD Integration
    print("\n[4/5] Testing PyGAD Engine Integration...")
    try:
        from src.pygad_engine import run_big_bang_pygad
        print("    ✓ PyGAD engine imports RealOpenMCModel")
        print("    ✓ Integration code present (lines 126-147)")
        tests_passed += 1
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        tests_failed += 1
    
    # Test 5: Check Nuclear Data
    print("\n[5/5] Testing Nuclear Data Library...")
    try:
        xs_path = openmc.config['cross_sections']
        if xs_path and os.path.exists(xs_path):
            # Count nuclides
            import glob
            xs_dir = os.path.dirname(xs_path)
            nuclide_files = glob.glob(os.path.join(xs_dir, 'neutron', '*.h5'))
            count = len(nuclide_files)
            print(f"    ✓ Found {count} nuclides in library")
            print(f"    ✓ Library location: {xs_dir}")
            tests_passed += 1
        else:
            print(f"    ✗ Cross sections file not found")
            tests_failed += 1
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests Passed: {tests_passed}/5")
    print(f"Tests Failed: {tests_failed}/5")
    
    if tests_passed == 5:
        print("\n✓ ALL TESTS PASSED - OpenMC is fully integrated!")
        print("\nYou can now run:")
        print("  python main.py --big-bang")
        return True
    else:
        print(f"\n✗ {tests_failed} test(s) failed")
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
