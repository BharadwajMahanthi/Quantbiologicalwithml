import openmc
import openmc.stats  # Explicit import to avoid linter warnings
import numpy as np
import logging

logger = logging.getLogger('quantum_bio_system')

class RealOpenMCModel:
    """
    Wraps the OFFICIAL OpenMC Python API to generate and run 
    Monte Carlo neutron/photon transport simulations.
    
    Requires: openmc python package and binary installed.
    """
    def __init__(self, materials_xml="materials.xml"):
        self.materials_xml = materials_xml
        
    def build_model(self):
        """
        Defines the Biological Shielding Model using REAL OpenMC classes.
        """
        logger.info("Building OpenMC Model (Real API)...")
        
        # 1. Materials
        # Air
        air = openmc.Material(name='air')
        air.set_density('g/cm3', 0.001205)
        air.add_nuclide('N14', 0.78)
        air.add_nuclide('O16', 0.21)
        air.add_nuclide('Ar40', 0.01)
        
        # Soil (Shield) - Using specific isotopes instead of elements
        # This allows XML generation without requiring cross_sections.xml
        soil = openmc.Material(name='soil')
        soil.set_density('g/cm3', 1.5)
        # Silicon-28 (most abundant Si isotope)
        soil.add_nuclide('Si28', 0.3)
        # Oxygen-16
        soil.add_nuclide('O16', 0.5)
        # Aluminum-27
        soil.add_nuclide('Al27', 0.1)
        # Trace Radionuclides
        soil.add_nuclide('Cs137', 1e-6)
        
        materials = openmc.Materials([air, soil])
        materials.export_to_xml()
        
        # 2. Geometry
        # Surfaces
        ground = openmc.ZPlane(z0=0.0, boundary_type='vacuum')
        interface = openmc.ZPlane(z0=100.0) # 1 meter of air
        top = openmc.ZPlane(z0=150.0, boundary_type='vacuum') # 50cm soil
        
        # Cells
        air_region = +ground & -interface
        soil_region = +interface & -top
        
        cell_air = openmc.Cell(fill=air, region=air_region)
        cell_soil = openmc.Cell(fill=soil, region=soil_region)
        
        geometry = openmc.Geometry([cell_air, cell_soil])
        geometry.export_to_xml()
        
        # 3. Settings
        settings = openmc.Settings()
        settings.batches = 10
        settings.inactive = 2
        settings.particles = 1000
        settings.run_mode = 'fixed source'
        
        # Source: Cosmic Rays / Gamma from INSIDE the air region
        # Position must be within geometry bounds: between z=0 and z=100 (air region)
        source = openmc.Source()
        source.space = openmc.stats.Point((0, 0, 50))  # Middle of air region (was 160, outside geometry!)
        source.angle = openmc.stats.Isotropic()
        source.energy = openmc.stats.Discrete([1.0e6], [1.0])  # 1 MeV (in eV)
        source.particle = 'neutron'
        settings.source = source
        
        settings.export_to_xml()
        
        # 4. Tallies
        tally = openmc.Tally(name='heating')
        cell_filter = openmc.CellFilter(cell_soil)
        tally.filters = [cell_filter]
        tally.scores = ['heating']
        
        tallies = openmc.Tallies([tally])
        tallies.export_to_xml()
        
        return True

    def run(self):
        try:
            logger.info("Running OpenMC Simulation...")
            openmc.run()
            
            # Load Results
            with openmc.StatePoint('statepoint.10.h5') as sp:
                t = sp.get_tally(name='heating')
                mean = t.mean.ravel()[0]
                std_dev = t.std_dev.ravel()[0]
                
                logger.info(f"OpenMC Heating Result: {mean:.4e} +/- {std_dev:.4e} eV/source")
                return mean
                
        except Exception as e:
            logger.error(f"OpenMC Execution Failed: {e}")
            logger.error("Ensure 'openmc' executable is in PATH.")
            return None
