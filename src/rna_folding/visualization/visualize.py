"""
Module for visualizing RNA sequences and structures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
from pathlib import Path
import py3Dmol
from IPython.display import display

# Define color scheme for nucleotides
NUCLEOTIDE_COLORS = {
    'A': '#32CD32',  # Lime Green
    'C': '#1E90FF',  # Dodger Blue
    'G': '#FF8C00',  # Dark Orange
    'U': '#DC143C',  # Crimson
    'T': '#DC143C',  # Crimson (same as U)
    'N': '#808080',  # Gray for unknown
}

def plot_rna_sequence(sequence, figsize=(10, 2), title=None):
    """
    Plot RNA sequence as colored blocks.
    
    Args:
        sequence (str): RNA sequence string.
        figsize (tuple): Figure size (width, height).
        title (str, optional): Plot title.
        
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create colored blocks for each nucleotide
    for i, nt in enumerate(sequence):
        color = NUCLEOTIDE_COLORS.get(nt, '#808080')  # Default to gray for unknown nucleotides
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
        ax.text(i + 0.5, 0.5, nt, ha='center', va='center', color='white', fontweight='bold')
    
    # Set plot limits and labels
    ax.set_xlim(0, len(sequence))
    ax.set_ylim(0, 1)
    ax.set_xlabel('Position')
    ax.set_yticks([])
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'RNA Sequence (Length: {len(sequence)})')
    
    plt.tight_layout()
    return fig

def plot_rna_structure_2d(coordinates, figsize=(10, 10), projection='xy', title=None):
    """
    Plot RNA structure in 2D.
    
    Args:
        coordinates (numpy.ndarray): Array of shape (n, 3) containing 3D coordinates.
        figsize (tuple): Figure size (width, height).
        projection (str): Projection plane ('xy', 'xz', or 'yz').
        title (str, optional): Plot title.
        
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Select coordinates based on projection
    if projection == 'xy':
        x, y = coordinates[:, 0], coordinates[:, 1]
        xlabel, ylabel = 'X', 'Y'
    elif projection == 'xz':
        x, y = coordinates[:, 0], coordinates[:, 2]
        xlabel, ylabel = 'X', 'Z'
    elif projection == 'yz':
        x, y = coordinates[:, 1], coordinates[:, 2]
        xlabel, ylabel = 'Y', 'Z'
    else:
        raise ValueError(f"Invalid projection: {projection}. Choose from 'xy', 'xz', or 'yz'.")
    
    # Plot backbone as a line
    ax.plot(x, y, '-', color='gray', alpha=0.7, linewidth=1)
    
    # Plot nucleotides as points
    ax.scatter(x, y, s=50, c=range(len(x)), cmap='viridis', alpha=0.8)
    
    # Add labels for some nucleotides (e.g., every 10th)
    for i in range(0, len(x), 10):
        ax.text(x[i], y[i], str(i+1), fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Set plot limits and labels
    ax.set_xlabel(xlabel + ' (Å)')
    ax.set_ylabel(ylabel + ' (Å)')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'RNA Structure - {projection.upper()} Projection')
    
    plt.tight_layout()
    return fig

def create_rna_structure_3d(coordinates, sequence=None, width=800, height=500):
    """
    Create interactive 3D visualization of RNA structure.
    
    Args:
        coordinates (numpy.ndarray): Array of shape (n, 3) containing 3D coordinates.
        sequence (str, optional): RNA sequence string.
        width (int): Width of the viewer.
        height (int): Height of the viewer.
        
    Returns:
        py3Dmol.view: Interactive 3D viewer object.
    """
    # Create a py3Dmol view
    view = py3Dmol.view(width=width, height=height)
    
    # Add backbone as a line
    backbone = []
    for i in range(len(coordinates)):
        x, y, z = coordinates[i]
        backbone.append({'x': float(x), 'y': float(y), 'z': float(z)})
    
    view.addLine({
        'start': {'x': backbone[0]['x'], 'y': backbone[0]['y'], 'z': backbone[0]['z']},
        'end': {'x': backbone[-1]['x'], 'y': backbone[-1]['y'], 'z': backbone[-1]['z']},
        'dashed': True,
        'color': 'gray',
        'opacity': 0.5
    })
    
    # Add nucleotides as spheres
    for i in range(len(coordinates)):
        x, y, z = coordinates[i]
        
        # Get color based on nucleotide type if sequence is provided
        color = '#808080'  # Default gray
        if sequence and i < len(sequence):
            color = NUCLEOTIDE_COLORS.get(sequence[i], '#808080')
        
        view.addSphere({
            'center': {'x': float(x), 'y': float(y), 'z': float(z)},
            'radius': 0.8,
            'color': color,
            'opacity': 0.8
        })
        
        # Add labels for some nucleotides (e.g., every 10th)
        if i % 10 == 0:
            view.addLabel(str(i+1), {
                'position': {'x': float(x), 'y': float(y), 'z': float(z)},
                'backgroundColor': 'white',
                'fontColor': 'black',
                'fontSize': 12,
                'backgroundOpacity': 0.7
            })
    
    # Set view options
    view.zoomTo()
    view.setStyle({}, {'stick': {}})
    
    return view

def save_visualization(fig, output_path, dpi=300):
    """
    Save visualization to file.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        output_path (str or Path): Path to save the figure.
        dpi (int): Resolution in dots per inch.
        
    Returns:
        str: Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    return str(output_path)

def visualize_rna_sample(sequence_data, labels_data, output_dir=None):
    """
    Visualize a sample RNA structure.
    
    Args:
        sequence_data (pandas.Series): RNA sequence data.
        labels_data (pandas.DataFrame): RNA structure coordinates.
        output_dir (str or Path, optional): Directory to save visualizations.
        
    Returns:
        dict: Dictionary containing visualization objects.
    """
    sequence = sequence_data['sequence']
    target_id = sequence_data['target_id']
    
    # Extract coordinates
    coords = np.array([
        labels_data['x_1'].values,
        labels_data['y_1'].values,
        labels_data['z_1'].values
    ]).T
    
    # Create visualizations
    seq_fig = plot_rna_sequence(sequence, title=f"RNA Sequence: {target_id}")
    xy_fig = plot_rna_structure_2d(coords, projection='xy', title=f"RNA Structure: {target_id} (XY Projection)")
    xz_fig = plot_rna_structure_2d(coords, projection='xz', title=f"RNA Structure: {target_id} (XZ Projection)")
    yz_fig = plot_rna_structure_2d(coords, projection='yz', title=f"RNA Structure: {target_id} (YZ Projection)")
    
    # Create 3D visualization
    view_3d = create_rna_structure_3d(coords, sequence)
    
    # Save visualizations if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_visualization(seq_fig, output_dir / f"{target_id}_sequence.png")
        save_visualization(xy_fig, output_dir / f"{target_id}_structure_xy.png")
        save_visualization(xz_fig, output_dir / f"{target_id}_structure_xz.png")
        save_visualization(yz_fig, output_dir / f"{target_id}_structure_yz.png")
    
    return {
        'sequence_fig': seq_fig,
        'xy_fig': xy_fig,
        'xz_fig': xz_fig,
        'yz_fig': yz_fig,
        '3d_view': view_3d
    }

def main():
    """Main function to demonstrate visualization capabilities."""
    # Load sample data
    data_dir = Path("data/raw")
    train_sequences = pd.read_csv(data_dir / "train_sequences.csv")
    train_labels = pd.read_csv(data_dir / "train_labels.csv")
    
    # Get the first sequence and its corresponding labels
    sample_seq = train_sequences.iloc[0]
    sample_id = sample_seq['target_id']
    sample_labels = train_labels[train_labels['ID'].str.startswith(f"{sample_id}_")]
    
    # Visualize the sample
    output_dir = Path("data/visualizations")
    visualizations = visualize_rna_sample(sample_seq, sample_labels, output_dir)
    
    print(f"Visualizations saved to {output_dir}")
    
    # Display the 3D view if in a notebook environment
    try:
        display(visualizations['3d_view'])
    except:
        print("3D visualization can only be displayed in a notebook environment.")

if __name__ == "__main__":
    main()
