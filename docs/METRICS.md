# Evaluation Metrics for RNA 3D Structure Prediction

This document explains the metrics used to evaluate the quality of predicted RNA 3D structures.

## Overview

Evaluating the quality of predicted RNA 3D structures requires specialized metrics that capture both local and global structural similarity. The primary metrics used in this project are:

1. **Root Mean Square Deviation (RMSD)**: Measures local structural similarity
2. **Template Modeling Score (TM-score)**: Measures global structural similarity
3. **Physics-Based Energy Terms**: Evaluate the physical plausibility of structures

## Root Mean Square Deviation (RMSD)

RMSD measures the average distance between corresponding atoms in two structures after optimal superposition.

### Mathematical Definition

For two sets of coordinates $X = \{x_1, x_2, ..., x_N\}$ and $Y = \{y_1, y_2, ..., y_N\}$, the RMSD is defined as:

$$\text{RMSD}(X, Y) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} ||x_i - y_i||^2}$$

where $||x_i - y_i||$ is the Euclidean distance between corresponding points.

### Implementation

```python
def rmsd(coords1, coords2):
    """
    Calculate Root Mean Square Deviation (RMSD) between two sets of coordinates.
    
    Args:
        coords1 (numpy.ndarray): First set of coordinates with shape (N, 3).
        coords2 (numpy.ndarray): Second set of coordinates with shape (N, 3).
        
    Returns:
        float: RMSD value.
    """
    # Ensure inputs are numpy arrays
    if isinstance(coords1, torch.Tensor):
        coords1 = coords1.detach().cpu().numpy()
    if isinstance(coords2, torch.Tensor):
        coords2 = coords2.detach().cpu().numpy()
    
    # Check that shapes match
    if coords1.shape != coords2.shape:
        raise ValueError(f"Coordinate shapes do not match: {coords1.shape} vs {coords2.shape}")
    
    # Calculate RMSD
    squared_diff = np.sum((coords1 - coords2) ** 2, axis=1)
    return np.sqrt(np.mean(squared_diff))
```

### Interpretation

- **Lower values are better**: RMSD = 0 means perfect match
- **Scale-dependent**: Values depend on the size of the structure
- **Sensitive to outliers**: A few large deviations can significantly increase RMSD
- **Typical values**: 
  - < 2Å: Very good prediction
  - 2-5Å: Good prediction
  - 5-10Å: Moderate prediction
  - > 10Å: Poor prediction

## Template Modeling Score (TM-score)

TM-score measures the global structural similarity between two structures, with a focus on the overall fold rather than local deviations.

### Mathematical Definition

For two structures $X$ and $Y$ with length $L$, the TM-score is defined as:

$$\text{TM-score}(X, Y) = \frac{1}{L} \sum_{i=1}^{L} \frac{1}{1 + (d_i / d_0)^2}$$

where:
- $d_i$ is the distance between the $i$-th pair of residues after optimal superposition
- $d_0 = 1.24 \sqrt[3]{L - 15} - 1.8$ is a normalization factor that makes the score length-independent

### Implementation

```python
def tm_score(coords1, coords2, sequence_length=None):
    """
    Calculate Template Modeling Score (TM-score) between two sets of coordinates.
    
    Args:
        coords1 (numpy.ndarray): First set of coordinates with shape (N, 3).
        coords2 (numpy.ndarray): Second set of coordinates with shape (N, 3).
        sequence_length (int, optional): Length of the sequence. If None, uses the
            number of coordinates.
            
    Returns:
        float: TM-score value.
    """
    # Get sequence length
    L = sequence_length if sequence_length is not None else coords1.shape[0]
    
    # Calculate d0 (normalization factor)
    if L <= 15:
        d0 = 0.5  # Minimum value for d0
    else:
        d0 = 1.24 * (L - 15) ** (1/3) - 1.8
        d0 = max(d0, 0.5)  # Ensure d0 is at least 0.5
    
    # Calculate distances between all pairs of points
    distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))
    
    # Calculate TM-score
    tm_sum = np.sum(1.0 / (1.0 + (distances / d0) ** 2))
    tm = tm_sum / L
    
    return tm
```

### Interpretation

- **Range**: [0, 1], where 1 indicates a perfect match
- **Length-independent**: Comparable across structures of different sizes
- **Emphasis on global fold**: Less sensitive to local deviations
- **Typical values**:
  - > 0.5: Structures share the same fold
  - 0.3-0.5: Structures have similar folds
  - < 0.3: Structures have different folds

## Physics-Based Energy Terms

Physics-based energy terms evaluate the physical plausibility of predicted structures based on known physical constraints.

### Bond Length Energy

Measures how well the predicted structure satisfies bond length constraints:

```python
def bond_length_energy(coords):
    # Calculate distances between consecutive C1' atoms
    consecutive_coords = coords[:, :-1, :]
    next_coords = coords[:, 1:, :]
    
    # Calculate squared distances
    squared_distances = torch.sum((consecutive_coords - next_coords) ** 2, dim=-1)
    
    # Calculate energy term (harmonic potential)
    energy = torch.mean((distances - c1_c1_mean_distance) ** 2)
    
    return energy
```

### Steric Clash Energy

Measures how well the predicted structure avoids steric clashes (atoms occupying the same space):

```python
def steric_clash_energy(coords):
    # Calculate all pairwise distances
    coords_expanded1 = coords.unsqueeze(2)
    coords_expanded2 = coords.unsqueeze(1)
    
    # Calculate squared distances
    squared_distances = torch.sum((coords_expanded1 - coords_expanded2) ** 2, dim=-1)
    
    # Calculate energy term for steric clashes
    steric_energy = torch.sum(
        torch.relu(min_distance - masked_distances) ** 2,
        dim=(1, 2)
    )
    
    return torch.mean(steric_energy)
```

### Base Pairing Energy

Measures how well the predicted structure satisfies base pairing constraints:

```python
def base_pairing_energy(coords, sequence):
    # Process each sequence in the batch
    for b in range(batch_size):
        seq = sequence[b]
        
        # Find potential base pairs
        for i in range(seq_len - 3):
            for j in range(i + 3, seq_len):
                base_i = seq[i]
                base_j = seq[j]
                
                # Check if bases can form a pair
                if (base_i, base_j) in base_pair_distances:
                    # Get target distance
                    target_distance = base_pair_distances[(base_i, base_j)]
                    
                    # Calculate actual distance
                    actual_distance = torch.sqrt(torch.sum((coords[b, i, :] - coords[b, j, :]) ** 2))
                    
                    # Add to energy term
                    energy = energy + (actual_distance - target_distance) ** 2
```

### Interpretation

- **Lower values are better**: Energy = 0 means perfect satisfaction of constraints
- **Relative importance**: Different energy terms may have different scales
- **Weighting**: Energy terms are weighted to balance their contributions
- **Trade-offs**: Optimizing one energy term may worsen others

## Combined Metrics

To get a comprehensive evaluation of predicted structures, we combine multiple metrics:

```python
def calculate_all_metrics(predicted_coords, true_coords, sequence=None):
    """
    Calculate all metrics for evaluating RNA 3D structure prediction.
    
    Args:
        predicted_coords (numpy.ndarray): Predicted coordinates with shape (N, 3).
        true_coords (numpy.ndarray): True coordinates with shape (N, 3).
        sequence (str, optional): RNA sequence.
        
    Returns:
        dict: Dictionary containing all metrics.
    """
    # Calculate metrics
    metrics = {
        'rmsd': rmsd(predicted_coords, true_coords),
        'tm_score': tm_score(predicted_coords, true_coords, len(sequence) if sequence else None)
    }
    
    return metrics
```

## Visualization

In addition to numerical metrics, visualization is an important tool for evaluating predicted structures:

1. **3D Visualization**: Render predicted and true structures in 3D
2. **Distance Maps**: Visualize pairwise distances between nucleotides
3. **Error Maps**: Visualize local errors in predicted structures
4. **Ensemble Visualization**: Visualize multiple predictions to assess uncertainty

## Conclusion

Evaluating RNA 3D structure predictions requires a combination of metrics that capture different aspects of structural similarity and physical plausibility. By using RMSD, TM-score, and physics-based energy terms, we can comprehensively assess the quality of predicted structures and guide the development of better prediction models.
