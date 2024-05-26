import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class CustomerGraph:
    def __init__(self, data,col,base):
        self.data = data[[col,base]]
        self.colToEncode = col
        self.colEncodeBasedOn = base
        self.customer_labels = None
        self.similarity_matrix = None
        self.graph = None

    def preprocess_data(self):
        """
        Preprocesses data and returns binarized DataFrame and customer labels.
        """
        logger.info("Preprocessing data...")
        # Convert data dictionary to DataFrame
        df = pd.DataFrame(self.data)

        # Use MultiLabelBinarizer to binarize item lists
        mlb = MultiLabelBinarizer()
        binarized_items = mlb.fit_transform(df[self.colEncodeBasedOn])

        # Convert binarized items into a DataFrame
        binarized_df = pd.DataFrame(binarized_items, columns=mlb.classes_)

        # Concatenate original DataFrame with binarized DataFrame
        df = pd.concat([df, binarized_df], axis=1)

        # Get customer labels
        self.customer_labels = df[self.colToEncode].to_dict()

        return df

    def calculate_cosine_similarity(self, data):
        """
        Calculates cosine similarity scores between customer vectors.
        """
        logger.info("Calculating cosine similarity scores...")
        customer_vectors = data.drop(columns=[self.colToEncode, self.colEncodeBasedOn]).values
        similarity_matrix = cosine_similarity(customer_vectors)

        # Calculate scores as 1 divided by cosine similarity
        with np.errstate(divide='ignore'):
            # Calculate scores as 1 divided by cosine similarity, handling division by zero
            scores = np.where(similarity_matrix == 0, np.inf, 1 / similarity_matrix)
        np.fill_diagonal(scores, 0)  # Set diagonal elements to 0 to exclude self-similarity
        self.similarity_matrix = scores



    def create_graph(self):
        """
        Creates a graph with nodes representing customers and weighted edges representing cosine similarity scores.
        """
        logger.info("Creating graph...")
        G = nx.Graph()
        num_customers = len(self.customer_labels)

        # Add nodes
        for customer, label in self.customer_labels.items():
            G.add_node(customer, label=label)

        # Add weighted edges based on cosine similarity scores
        for i in range(num_customers):
            for j in range(i+1, num_customers):
                customer1, customer2 = list(self.customer_labels.keys())[i], list(self.customer_labels.keys())[j]
                similarity_score = self.similarity_matrix[i, j]
                G.add_edge(customer1, customer2, weight=similarity_score)

        self.graph = G

    def encode_shortest_path(self, df):
        """
        Encodes the shortest path that starts from point A, visits all other points, and returns to the starting point.
        """
        logger.info("Encoding shortest path that visits all points and returns to the starting point...")

        # Get the starting point
        start_point = list(self.customer_labels.keys())[0]

        # Check if start point is in the graph
        if start_point not in self.graph.nodes:
            logger.error(f"Start point {start_point} is not in the graph.")
            return df

        all_shortest_paths = nx.shortest_path(self.graph, source=start_point, weight='weight')

        # Find the longest shortest path among all shortest paths
        shortest_path = max(all_shortest_paths.values(), key=len)
        count = 0
        for node in shortest_path:
            # Find the row corresponding to the current node
            if node in self.customer_labels:  # Check if customer exists in the graph
                df.loc[node, 'encoded_tour'] = count
                count += 1
            else:
                logger.warning(f"{node} is not in the graph.")

        return df


def main():
    # Dummy data in a pandas DataFrame
    data = {
        'customer': ['customer1', 'customer2', 'customer3'],
        'items_purchased': [['a', 'b'], ['d', 'c'], ['a', 'd']]
    }
    df = pd.DataFrame(data)
    # Specify the column to encode and the base column
    col = 'customer'
    base = 'items_purchased'

    # Create CustomerGraph instance
    customer_graph = CustomerGraph(df, col, base)

    # Preprocess data
    df = customer_graph.preprocess_data()

    # Calculate cosine similarity scores
    customer_graph.calculate_cosine_similarity(df)

    # Create a graph
    customer_graph.create_graph()

    # Encode shortest paths for all customers
    df = customer_graph.encode_shortest_path(df)

    print(df)


if __name__ == "__main__":
    main()
