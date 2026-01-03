import xml.etree.ElementTree as ET
import os

class OpenMCAdapter:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.materials = {}
        
    def load_materials(self):
        """Parses materials.xml to extract density and composition."""
        if not os.path.exists(self.xml_path):
            print(f"OpenMC Warning: {self.xml_path} not found.")
            return {}
            
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            
            for mat in root.findall('material'):
                name = mat.get('name')
                if name is None:
                    continue  # Skip materials without names
                
                density_elem = mat.find('density')
                if density_elem is None:
                    continue  # Skip materials without density
                
                density_value = density_elem.get('value')
                if density_value is None:
                    continue  # Skip if density value is missing
                density = float(density_value)
                
                # Composition
                composition = {}
                for nuclide in mat.findall('nuclide'):
                    iso_name = nuclide.get('name')
                    wo_value = nuclide.get('wo')
                    if iso_name and wo_value:  # Only add if both exist
                        composition[iso_name] = float(wo_value)
                        
                self.materials[name] = {
                    'density': density,
                    'composition': composition
                }
                
            return self.materials
            
        except Exception as e:
            print(f"Error parsing OpenMC XML: {e}")
            return {}

    def get_environmental_impact(self):
        """
        Derives simulation scalars from material properties.
        Example: 
        - High Density -> Shielding (Lower Radiation)
        - Cs137 -> Radiation Source
        - O16 -> Oxygen Source
        """
        impact = {'radiation_shielding': 0.0, 'radiation_source': 0.0, 'oxygen_level': 0.0}
        
        for name, props in self.materials.items():
            # Density acts as shielding
            impact['radiation_shielding'] += props['density'] * 0.1
            
            # Check isotopes
            for iso, amt in props['composition'].items():
                if 'Cs' in iso or 'U' in iso: # Radioactive
                    impact['radiation_source'] += amt * 1000.0
                if 'O' in iso: # Oxygen
                    impact['oxygen_level'] += amt * props['density'] * 10.0
                    
        return impact
