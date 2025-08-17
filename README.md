# HurricaneGenomes

## Getting Started

From the project root, run:
cd frontend/
npm run dev:all


## Project Overview - Claude

HurricaneGenomes analyzes hurricane track data from HURDAT2 to create "family trees" of storm patterns. The project treats each hurricane track as a unique "genome" and uses hierarchical clustering to identify relationships between storms based on their geographical paths.

### Key Features

- **Track Comparison**: Uses great circle distance calculations to measure similarity between hurricane paths
- **Hierarchical Clustering**: Applies average-linkage clustering to group similar storm tracks
- **Visual Analysis**: Generates dendrograms and interactive maps showing storm relationships
- **Parent-Child Trees**: Creates family tree structures showing how storms relate to each other

### How It Works

1. **Data Processing**: Loads hurricane track data from HURDAT2 format
2. **Pairwise Comparison**: Calculates distances between all storm track pairs
3. **Clustering**: Groups similar tracks using hierarchical clustering algorithms
4. **Visualization**: Maps clusters with color-coded families and generates dendrograms

The system identifies which storms follow similar paths and at what points they diverge, revealing patterns in hurricane behavior and potential "genetic" relationships between storm systems.
