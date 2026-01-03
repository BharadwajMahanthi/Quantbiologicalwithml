"""
Nuclear Cross Section Data Downloader for OpenMC
Downloads ENDF/B-VIII.0 data from official OpenMC sources
"""

import urllib.request
import tarfile
import os
import sys
from pathlib import Path

def download_nuclear_data():
    """Download and extract ENDF/B-VIII.0 nuclear data."""
    
    # Create directory
    data_dir = Path.home() / '.openmc' / 'endfb-viii.0'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("OpenMC Nuclear Data Downloader")
    print("=" * 60)
    print(f"\nTarget directory: {data_dir}")
    print("\nDownloading ENDF/B-VIII.0 nuclear cross section data...")
    print("Size: ~2 GB (compressed: ~500 MB)")
    print("This will take 5-15 minutes depending on your connection.\n")
    
    # Official OpenMC data URL
    url = "https://anl.box.com/shared/static/9igk353zpy8fn9ttvtrqgzvw1vtejoz6.xz"
    tar_path = data_dir.parent / "endfb-viii.0.tar.xz"
    
    # Download with progress
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100.0 * downloaded / total_size, 100.0)
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        
        print(f'\r[{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', end='')
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, tar_path, show_progress)
        print("\n\n✓ Download complete!")
        
        print("\nExtracting data (this may take 5-10 minutes)...")
        with tarfile.open(tar_path, 'r:xz') as tar:
            tar.extractall(data_dir.parent)
        
        print("✓ Extraction complete!")
        
        # Verify cross_sections.xml exists
        xs_file = data_dir / 'cross_sections.xml'
        if xs_file.exists():
            print(f"\n✓ Verification successful!")
            print(f"  Cross sections file found: {xs_file}")
            print(f"  File size: {xs_file.stat().st_size / (1024*1024):.1f} MB")
            
            # Clean up tar file
            tar_path.unlink()
            print(f"\n✓ Cleaned up temporary file: {tar_path.name}")
            
            # Automatically set environment variable
            print("\n" + "=" * 60)
            print("Setting Environment Variable")
            print("=" * 60)
            
            import subprocess
            try:
                # Use setx to set permanent environment variable
                cmd = f'setx OPENMC_CROSS_SECTIONS "{xs_file}"'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✓ Environment variable set successfully!")
                    print(f"  OPENMC_CROSS_SECTIONS = {xs_file}")
                    print("\n⚠️  IMPORTANT: Close and reopen your terminal for changes to take effect!")
                else:
                    print(f"⚠️  Could not set environment variable automatically.")
                    print(f"  Please run manually:")
                    print(f'    setx OPENMC_CROSS_SECTIONS "{xs_file}"')
            except Exception as e:
                print(f"⚠️  Error setting environment variable: {e}")
                print(f"  Please run manually:")
                print(f'    setx OPENMC_CROSS_SECTIONS "{xs_file}"')
            
            print("\n" + "=" * 60)
            print("Installation Complete!")
            print("=" * 60)
            print("\nNext Steps:")
            print("1. Close and restart your terminal")
            print("2. Verify installation:")
            print('   python -c "import openmc; print(openmc.config[\'cross_sections\'])"')
            print("3. Run integration test:")
            print("   python test_openmc_integration.py")
            
            return True
        else:
            print(f"\n✗ Error: cross_sections.xml not found at {xs_file}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during download/extraction: {e}")
        if tar_path.exists():
            print(f"Partial download saved at: {tar_path}")
            print("You can try resuming by re-running this script.")
        return False

if __name__ == "__main__":
    success = download_nuclear_data()
    sys.exit(0 if success else 1)
