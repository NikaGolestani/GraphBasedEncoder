
# GraphEncoder

The `GraphEncoder` class processes data representing entities and their attributes, calculates cosine similarity between entities based on their attributes, creates a graph with entities as nodes, encodes the shortest path through the graph using the Traveling Salesman Problem (TSP) algorithm, and then applies one-hot encoding to ensure that similar entities are encoded with nearby values.

## Requirements

- Python 3.x
- NumPy
- Pandas
- NetworkX
- scikit-learn

## Installation

You can install the required packages using pip:

```sh
pip install numpy pandas networkx scikit-learn
````

## Usage

### Step-by-Step Instructions

1. **Prepare the Data**:
   Prepare a pandas DataFrame with entities and their attributes.

2. **Create an Instance of `GraphEncoder`**:
   Instantiate the `GraphEncoder` class with your DataFrame and specify the columns for entity identifiers and attributes.

3. **Preprocess Data**:
   Call the `preprocess_data` method to binarize the attributes and create a DataFrame suitable for similarity calculations.

4. **Calculate Cosine Similarity**:
   Call the `calculate_cosine_similarity` method to compute the similarity scores between entities.

5. **Create the Graph**:
   Call the `create_graph` method to create a graph with entities as nodes and cosine similarity scores as weighted edges.

6. **Encode the Shortest Path**:
   Call the `encode_shortest_path` method to encode the shortest path that visits all entities without returning to the starting point. This method updates the DataFrame with the encoded path.

### Methods

- `preprocess_data()`: Preprocesses data and returns a binarized DataFrame and entity labels.
- `calculate_cosine_similarity(data)`: Calculates cosine similarity scores between entity vectors.
- `create_graph()`: Creates a graph with nodes representing entities and weighted edges representing cosine similarity scores.
- `encode_shortest_path(df)`: Encodes the shortest path that starts from the first entity, visits all other entities, and returns the updated DataFrame with the encoded path.

## Logging

The class uses Python's built-in `logging` module to provide information about the process steps. By default, the logging level is set to `INFO`, but this can be adjusted as needed.

```python
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
```

## Example Data Format

Ensure your data is in the following format:

```python
data = {
    'entity': ['entity1', 'entity2', 'entity3'],
    'attributes': [['a', 'b'], ['b', 'c'], ['x', 'y']]
}
df = pd.DataFrame(data)
```

Replace `'entity'` and `'attributes'` with the appropriate column names from your DataFrame when creating an instance of `GraphEncoder`.

## Author

Nika Golestani

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

