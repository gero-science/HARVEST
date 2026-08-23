"""
Module for resolving protein names to amino acid sequences.

Contains:
- FastaGeneResolver: Simple resolver by gene + species directly from FASTA
"""

from .fasta_gene_resolver import FastaGeneResolver

__all__ = ['FastaGeneResolver']
__version__ = '2.1.0'
