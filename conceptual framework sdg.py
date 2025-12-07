# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 09:19:57 2025

@author: marina
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set professional style with white background
fig, ax = plt.subplots(1, 1, figsize=(16, 12), facecolor='white')
fig.patch.set_facecolor('white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor('white')

# Professional color palette - muted and academic
colonial_color = '#2C3E50'     # Dark blue-gray
infrastructure_color = '#34495E' # Medium gray-blue  
sdg_color = '#7F8C8D'          # Light gray
violence_color = '#95A5A6'     # Very light gray
accent_color = '#E74C3C'       # Subtle red accent
text_color = '#2C3E50'         # Dark text

# Title and explanation
plt.text(5, 9.5, 'Figure 1. Conceptual Framework: Violence, Epistemic Infrastructures & SDGs', 
         fontsize=18, fontweight='bold', ha='center', color=text_color)

# Brief explanation box
explanation_box = FancyBboxPatch((0.5, 8.6), 9, 0.6, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#F8F9FA', 
                                alpha=0.8, 
                                edgecolor='#DEE2E6',
                                linewidth=1)
ax.add_patch(explanation_box)

plt.text(5, 9.0, 'This framework illustrates how colonial underpinnings shape epistemic infrastructures and intersect', 
         fontsize=11, ha='center', color=text_color)
plt.text(5, 8.8, 'with forms of violence (slow and epistemic) that remain hidden in SDG measurement systems.', 
         fontsize=11, ha='center', color=text_color)

# Foundation: Colonial underpinnings (bottom layer)
foundation = FancyBboxPatch((0.5, 0.5), 9, 1.5, 
                           boxstyle="round,pad=0.1", 
                           facecolor=colonial_color, 
                           alpha=0.8, 
                           edgecolor='black', 
                           linewidth=1.5)
ax.add_patch(foundation)
plt.text(5, 1.25, 'COLONIAL UNDERPINNINGS', fontsize=16, fontweight='bold', 
         ha='center', color='white')
plt.text(5, 0.9, 'Modernity-coloniality • Universality assumptions • Single framing systems', 
         fontsize=11, ha='center', color='white')

# Three main pillars
# 1. Epistemic Infrastructures
pillar1 = FancyBboxPatch((0.5, 3), 2.8, 4.5, 
                        boxstyle="round,pad=0.1", 
                        facecolor=infrastructure_color, 
                        alpha=0.3, 
                        edgecolor=infrastructure_color, 
                        linewidth=2)
ax.add_patch(pillar1)
plt.text(1.9, 7, 'EPISTEMIC', fontsize=14, fontweight='bold', ha='center', color=infrastructure_color)
plt.text(1.9, 6.7, 'INFRASTRUCTURES', fontsize=14, fontweight='bold', ha='center', color=infrastructure_color)
plt.text(1.9, 6.3, '(Bandola-Gill et al. 2022)', fontsize=10, ha='center', color=infrastructure_color, style='italic')

# Sub-components
levels = ['1. Materialities of\nMeasurement', 
          '2. Epistemic Communities\n& Networks', 
          '3. Global Public\nPolicy Paradigm']
for i, level in enumerate(levels):
    y_pos = 5.5 - i*0.8
    level_box = FancyBboxPatch((0.7, y_pos-0.3), 2.4, 0.6, 
                              boxstyle="round,pad=0.05", 
                              facecolor='white', 
                              alpha=0.9, 
                              edgecolor=infrastructure_color)
    ax.add_patch(level_box)
    plt.text(1.9, y_pos, level, fontsize=10, ha='center', va='center')

# 2. SDG Framework (center)
pillar2 = FancyBboxPatch((3.6, 3), 2.8, 4.5, 
                        boxstyle="round,pad=0.1", 
                        facecolor=sdg_color, 
                        alpha=0.3, 
                        edgecolor=sdg_color, 
                        linewidth=2)
ax.add_patch(pillar2)
plt.text(5, 6.5, 'SUSTAINABLE', fontsize=14, fontweight='bold', ha='center', color=sdg_color)
plt.text(5, 6.2, 'DEVELOPMENT GOALS', fontsize=14, fontweight='bold', ha='center', color=sdg_color)
plt.text(5, 5.8, 'Index & Spillover Index', fontsize=11, ha='center', color=sdg_color)

# SDG components
sdg_components = ['What is measured', 'How it is measured', 'Who measures', 'What is excluded']
for i, comp in enumerate(sdg_components):
    y_pos = 5.2 - i*0.4
    plt.text(5, y_pos, f'• {comp}', fontsize=10, ha='center', color=text_color)

# 3. Violence Types (right)
pillar3 = FancyBboxPatch((6.7, 3), 2.8, 4.5, 
                        boxstyle="round,pad=0.1", 
                        facecolor=violence_color, 
                        alpha=0.3, 
                        edgecolor=violence_color, 
                        linewidth=2)
ax.add_patch(pillar3)
plt.text(8.1, 7, 'VIOLENCE', fontsize=14, fontweight='bold', ha='center', color=violence_color)
plt.text(8.1, 6.7, 'MANIFESTATIONS', fontsize=14, fontweight='bold', ha='center', color=violence_color)

# Violence types
violence_box1 = FancyBboxPatch((6.9, 5.8), 2.4, 0.8, 
                              boxstyle="round,pad=0.05", 
                              facecolor='white', 
                              alpha=0.9, 
                              edgecolor=accent_color)
ax.add_patch(violence_box1)
plt.text(8.1, 6.2, 'SLOW VIOLENCE', fontsize=11, fontweight='bold', ha='center', color=accent_color)
plt.text(8.1, 5.95, 'Gradual, hidden impacts', fontsize=9, ha='center', color=text_color)

violence_box2 = FancyBboxPatch((6.9, 4.7), 2.4, 0.8, 
                              boxstyle="round,pad=0.05", 
                              facecolor='white', 
                              alpha=0.9, 
                              edgecolor=accent_color)
ax.add_patch(violence_box2)
plt.text(8.1, 5.1, 'EPISTEMIC VIOLENCE', fontsize=11, fontweight='bold', ha='center', color=accent_color)
plt.text(8.1, 4.85, 'Knowledge exclusion', fontsize=9, ha='center', color=text_color)

# Intersection zones with clean lines
# Central intersection area
intersection = FancyBboxPatch((2.5, 2.2), 5, 0.6, 
                             boxstyle="round,pad=0.05", 
                             facecolor=accent_color, 
                             alpha=0.2, 
                             edgecolor=accent_color)
ax.add_patch(intersection)
plt.text(5, 2.5, 'CRITICAL INTERSECTIONS', fontsize=12, fontweight='bold', ha='center', color=accent_color)

# Key manifestations
manifestations = ['Hidden environmental impacts', 'Excluded knowledge systems', 
                 'Measurement biases', 'Offshored consequences']
for i, manifest in enumerate(manifestations):
    x_pos = 1.5 + i*2
    plt.text(x_pos, 2.2, f'• {manifest}', fontsize=12, ha='center', color=text_color, 
             bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.8))

# Clean connecting arrows
arrows = [
    ((5, 2), (1.9, 3)),      # Foundation to Infrastructure
    ((5, 2), (5, 3)),        # Foundation to SDG  
    ((5, 2), (8.1, 3)),      # Foundation to Violence
    ((3.3, 5.2), (3.7, 5.2)), # Infrastructure to SDG
    ((6.3, 5.2), (6.7, 5.2))  # SDG to Violence
]

for start, end in arrows:
    arrow = ConnectionPatch(start, end, "data", "data", 
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=15, fc=text_color, ec=text_color, linewidth=1.5)
    ax.add_artist(arrow)

plt.tight_layout()
plt.savefig('conceptual_framework.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("Professional academic conceptual framework created!")
print("Features:")
print("- Clean, minimal design suitable for journal publication")
print("- Professional color palette (muted blues/grays)")
print("- Clear hierarchy and relationships")
print("- Academic typography and layout")
print("- Structured presentation of theoretical components")