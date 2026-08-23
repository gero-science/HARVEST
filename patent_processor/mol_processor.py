#!/usr/bin/env python3
"""
Compact module for reading and extracting data from MOL files.
Fixes UnicodeDecodeError and extracts chemical data.
"""

import os
import sys
from typing import Dict, Any, Optional, List
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from chemistry_rdkit import get_molecular_weight  # type: ignore
    CHEM_UTILS_AVAILABLE = True
except ImportError:
    CHEM_UTILS_AVAILABLE = False
    # Fallback: use direct RDKit import
    pass


def read_mol_file(file_path: str) -> str:
    """
    Read a MOL file with automatic encoding detection.

    Args:
        file_path: Path to the MOL file

    Returns:
        File contents as a string
    """
    encodings = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    # Fallback with error replacement
    with open(file_path, 'rb') as f:
        content_bytes = f.read()
    return content_bytes.decode('utf-8', errors='replace')


def has_dummy_atoms(mol) -> bool:
    """
    Check whether the molecule contains dummy atoms (placeholder atoms such as * or R).

    Args:
        mol: RDKit Mol object

    Returns:
        True if the molecule contains dummy atoms, False otherwise
    """
    if mol is None:
        return False

    for atom in mol.GetAtoms():
        # Check atomic number 0 (dummy atom) or symbol * or R
        if atom.GetAtomicNum() == 0 or atom.GetSymbol() in ['*', 'R']:
            return True

    return False


def is_scaffold_mol_file(file_path: str) -> bool:
    """
    Check whether a MOL file is a scaffold (incomplete molecule) by analyzing text content.
    Looks for atoms such as * or R-group placeholders (R1, R2, R1R2N, etc.) in the MOL file atom block.

    Scaffold = incomplete molecule with attachment points.

    Args:
        file_path: Path to the MOL file

    Returns:
        True if the file contains placeholder atoms (incomplete molecule), False otherwise
    """
    try:
        content = read_mol_file(file_path)
        if not content:
            return False

        lines = content.splitlines()

        # Find counts line (line 4 in a MOL file)
        if len(lines) < 4:
            return False

        counts_line = lines[3]
        parts = counts_line.split()

        if len(parts) < 2:
            return False

        try:
            num_atoms = int(parts[0])
        except (ValueError, IndexError):
            return False

        # Atom block starts at line 4 (index 3) and contains num_atoms rows
        atom_block_start = 4
        atom_block_end = atom_block_start + num_atoms

        if len(lines) < atom_block_end:
            return False

        # Check each atom block row for * or R-group atoms
        for i in range(atom_block_start, atom_block_end):
            line = lines[i]
            # In a MOL file, atom type is at positions 31-33 (0-indexed: 30-32)
            # But for longer types (R1R2N) it may be longer
            if len(line) >= 34:
                # Extract atom type (may be longer than 3 characters)
                atom_type = line[31:].split()[0] if len(line[31:].split()) > 0 else ''

                # Check for dummy atoms and R-group placeholders
                if atom_type == '*' or atom_type.startswith('R'):
                    return True

        return False

    except Exception:
        return False


def detect_attachment_points(mol) -> Dict[str, Any]:
    """
    Detect attachment points in an incomplete molecule (scaffold).

    Args:
        mol: RDKit Mol object

    Returns:
        Dictionary with attachment point information:
        - count: number of attachment points
        - indices: list of dummy atom indices
        - neighbor_atoms: list of neighboring atom indices (where fragments attach)
    """
    if mol is None:
        return {'count': 0, 'indices': [], 'neighbor_atoms': []}

    dummy_indices = []
    neighbor_atoms = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 or atom.GetSymbol() in ['*', 'R']:
            idx = atom.GetIdx()
            dummy_indices.append(idx)

            # Find neighboring atoms
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in neighbor_atoms:
                    neighbor_atoms.append(neighbor_idx)

    return {
        'count': len(dummy_indices),
        'indices': dummy_indices,
        'neighbor_atoms': neighbor_atoms
    }


def extract_chemical_data(mol_file_path: str) -> Dict[str, Any]:
    """
    Extract chemical data from a MOL file.
    
    Args:
        mol_file_path: Path to the MOL file
    
    Returns:
        Dictionary with extracted data, including scaffold (incomplete molecule) information
    """
    try:
        # First check whether the file is a scaffold/incomplete molecule (contains * atoms or R-groups)
        is_scaffold = is_scaffold_mol_file(mol_file_path)

        # For scaffolds (incomplete molecules): load with sanitize=False and do NOT process with RDKit functions
        if is_scaffold:
            mol = Chem.MolFromMolFile(mol_file_path, sanitize=False)

            if mol is None:
                return {
                    'success': False,
                    'error': 'Failed to load scaffold (incomplete molecule) from MOL file',
                    'is_scaffold': True,
                    'has_attachment_points': True
                }

            # Detect attachment points
            attachment_info = detect_attachment_points(mol)

            # For scaffolds (incomplete molecules), return minimal information without RDKit processing
            return {
                'success': True,
                'smiles': None,  # Do not generate SMILES for incomplete molecules
                'inchi_key': None,  # Do not generate InChI for incomplete molecules
                'inchi': None,
                'molecular_formula': None,  # Do not compute formula for incomplete molecules
                'molecular_weight': None,  # Do not compute weight for incomplete molecules
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'has_stereochemistry': False,
                'stereocenters': 0,
                'is_scaffold': True,
                'has_attachment_points': True,
                'attachment_point_count': attachment_info['count'],
                'scaffold_info': {
                    'attachment_point_count': attachment_info['count'],
                    'attachment_indices': attachment_info['indices'],
                    'neighbor_atoms': attachment_info['neighbor_atoms']
                }
            }

        # For full molecules: load normally and process with all RDKit functions
        mol = Chem.MolFromMolFile(mol_file_path)
        
        if mol is None:
            return {
                'success': False,
                'error': 'Failed to load molecule from MOL file',
                'is_scaffold': False
            }

        # Detect attachment points (in case they exist but were not detected by text parser)
        attachment_info = detect_attachment_points(mol)
        has_attachment_points = attachment_info['count'] > 0

        # If attachment points are detected after loading, this is also a scaffold (incomplete molecule)
        if has_attachment_points:
            return {
                'success': True,
                'smiles': None,
                'inchi_key': None,
                'inchi': None,
                'molecular_formula': None,
                'molecular_weight': None,
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'has_stereochemistry': False,
                'stereocenters': 0,
                'is_scaffold': True,
                'has_attachment_points': True,
                'attachment_point_count': attachment_info['count'],
                'scaffold_info': {
                    'attachment_point_count': attachment_info['count'],
                    'attachment_indices': attachment_info['indices'],
                    'neighbor_atoms': attachment_info['neighbor_atoms']
                }
            }
        
        # Full molecules only: extract all chemical data
        try:
            smiles = Chem.MolToSmiles(mol)
        except Exception as e:
            smiles = None

        try:
            inchi_key = Chem.MolToInchiKey(mol)
        except Exception:
            inchi_key = None

        try:
            inchi = Chem.MolToInchi(mol)
        except Exception:
            inchi = None
        
        # Extract additional data
        try:
            formula = rdMolDescriptors.CalcMolFormula(mol)
        except Exception:
            formula = None

        # Use utility for exact molecular weight
        if CHEM_UTILS_AVAILABLE:
            try:
                mol_weight = get_molecular_weight(mol, exact=True)
            except Exception:
                mol_weight = None
        else:
            # Fallback: direct RDKit call
            mol_weight = rdMolDescriptors.CalcExactMolWt(mol)

        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        
        try:
            has_stereo = len(Chem.FindMolChiralCenters(mol)) > 0
            stereocenters = len(Chem.FindMolChiralCenters(mol))
        except Exception:
            has_stereo = False
            stereocenters = 0
        
        return {
            'success': True,
            'smiles': smiles,
            'inchi_key': inchi_key,
            'inchi': inchi,
            'molecular_formula': formula,
            'molecular_weight': round(mol_weight, 2) if mol_weight is not None else None,
            'num_atoms': num_atoms,
            'num_bonds': num_bonds,
            'has_stereochemistry': has_stereo,
            'stereocenters': stereocenters,
            'is_scaffold': False,
            'has_attachment_points': False,
            'attachment_point_count': 0,
            'scaffold_info': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'is_scaffold': False,
            'has_attachment_points': False
        }


def process_mol_file(file_path: str, mark_scaffolds: bool = True) -> Dict[str, Any]:
    """
    Full MOL file processing: read + extract data.
    
    Args:
        file_path: Path to the MOL file
        mark_scaffolds: If True, mark scaffolds (incomplete molecules) with a special marker (default True)

    Returns:
        Dictionary with processing results, including scaffold (incomplete molecule) information
    """
    if not os.path.exists(file_path):
        return {
            'success': False,
            'error': f'File not found: {file_path}',
            'file_path': file_path
        }
    
    # Read file
    try:
        content = read_mol_file(file_path)
        file_size = len(content)
        num_lines = len(content.splitlines())
    except Exception as e:
        return {
            'success': False,
            'error': f'Error reading file: {e}',
            'file_path': file_path
        }
    
    # Extract chemical data
    chemical_data = extract_chemical_data(file_path)
    
    # Merge results
    result = {
        'file_path': file_path,
        'file_size': file_size,
        'num_lines': num_lines,
        **chemical_data
    }
    
    # Mark scaffolds (incomplete molecules) if requested
    if mark_scaffolds and (chemical_data.get('is_scaffold', False) or chemical_data.get('has_attachment_points', False)):
        result['scaffold_marker'] = 'SCAFFOLD_DETECTED'  # Incomplete molecule detected

    return result


def extract_smiles_and_inchi_key(mol_file_path: str) -> Dict[str, Any]:
    """
    Quickly extract only SMILES and InChI Key.
    
    Args:
        mol_file_path: Path to the MOL file
    
    Returns:
        Dictionary with SMILES and InChI Key
    """
    try:
        mol = Chem.MolFromMolFile(mol_file_path)
        
        if mol is None:
            return {
                'smiles': None,
                'inchi_key': None,
                'success': False,
                'error': 'Failed to load molecule from MOL file'
            }
        
        smiles = Chem.MolToSmiles(mol)
        inchi_key = Chem.MolToInchiKey(mol)
        
        return {
            'smiles': smiles,
            'inchi_key': inchi_key,
            'success': True
        }
        
    except Exception as e:
        return {
            'smiles': None,
            'inchi_key': None,
            'success': False,
            'error': str(e)
        }


def batch_process_mol_files(source_dir: str) -> list:
    """
    Process all MOL files in the specified directory.
    
    Args:
        source_dir: Path to the directory with MOL files
    
    Returns:
        List of processing results
    """
    import glob
    
    mol_files = glob.glob(os.path.join(source_dir, "*.MOL"))
    
    if not mol_files:
        return []
    
    results = []
    for mol_file in mol_files:
        result = process_mol_file(mol_file)
        results.append(result)
    
    return results


def save_skipped_scaffolds(scaffold_files: List[str], output_file: str) -> None:
    """
    Save the list of skipped scaffolds (incomplete molecules) to a file.

    Args:
        scaffold_files: List of paths to scaffold (incomplete molecule) files
        output_file: Output file path
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Skipped Scaffold MOL Files (Incomplete Molecules)\n")
            f.write(f"# Total: {len(scaffold_files)}\n")
            f.write("# These files contain scaffolds/fragments with attachment points (*) or R-groups\n")
            f.write("# Scaffold = incomplete molecule with attachment points\n")
            f.write("# They require manual processing or LLM interpretation\n\n")

            for file_path in sorted(scaffold_files):
                # Extract filename only
                filename = os.path.basename(file_path)
                f.write(f"{filename}\n")

        return True
    except Exception as e:
        print(f"Warning: Failed to save skipped scaffolds: {e}")
        return False


def print_mol_summary(result: Dict[str, Any]):
    """
    Print a brief summary of MOL file processing results.
    
    Args:
        result: MOL file processing result
    """
    if result['success']:
        print("✅ File processed successfully!")
        print(f"File size: {result['file_size']} characters")
        print(f"Number of lines: {result['num_lines']}")
        print()

        # Check whether the molecule is a scaffold (incomplete molecule)
        if result.get('is_scaffold', False):
            print("⚠️  SCAFFOLD DETECTED (incomplete molecule)")
            print(f"Attachment points: {result.get('attachment_point_count', 0)}")
            if result.get('scaffold_info'):
                print(f"Attachment point indices: {result['scaffold_info'].get('attachment_indices', [])}")
            if result.get('scaffold_marker'):
                print(f"Marker: {result['scaffold_marker']}")
            print()

        print("=== Chemical data ===")
        print(f"SMILES: {result['smiles']}")
        print(f"InChI Key: {result['inchi_key']}")
        print(f"Molecular formula: {result['molecular_formula']}")
        print(f"Molecular weight: {result['molecular_weight']}")
        print(f"Number of atoms: {result['num_atoms']}")
        print(f"Number of bonds: {result['num_bonds']}")
        print(f"Stereochemistry: {'Yes' if result['has_stereochemistry'] else 'No'}")
        if result.get('stereocenters', 0) > 0:
            print(f"Stereocenters: {result['stereocenters']}")
    else:
        print(f"❌ Error: {result['error']}")


# Usage example
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m patent_processor.mol_processor <path_to.MOL>")
        sys.exit(1)
    test_file = sys.argv[1]
    
    print("🧪 MOL PROCESSOR TEST")
    print("=" * 50)
    
    # Full processing
    result = process_mol_file(test_file)
    print_mol_summary(result)
    
    print("\n" + "=" * 50)
    
    # Quick extraction of SMILES and InChI Key only
    quick_result = extract_smiles_and_inchi_key(test_file)
    if quick_result['success']:
        print("🚀 Quick extraction:")
        print(f"SMILES: {quick_result['smiles']}")
        print(f"InChI Key: {quick_result['inchi_key']}")
    else:
        print(f"❌ Quick extraction error: {quick_result['error']}")
