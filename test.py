import unittest
import pandas as pd
from Encoder import GraphEncoder

class TestGraphEncoder(unittest.TestCase):
    def run_test(self, data, expected):
        # Input data
        df = pd.DataFrame(data)
        # Specify the column to encode and the base column
        col = 'customer'
        base = 'items_purchased'

        # Create GraphEncoder instance
        customer_graph = GraphEncoder(df, col, base)
        # Preprocess data
        df_processed = customer_graph.preprocess_data()
        # Calculate cosine similarity scores
        customer_graph.calculate_cosine_similarity(df_processed)
        # Create a graph
        customer_graph.create_graph()
        # Encode shortest paths for all customers
        df_encoded = customer_graph.encode_shortest_path(df_processed)
        # Expected encoded tour
        expected_encoded_tour = expected
        print(df_encoded)
        # Check if the 'encoded' values match the expected values
        self.assertListEqual(df_encoded['encoded'].tolist(), expected_encoded_tour)

    def test_one(self):
        # Input data and expected output
        data = {
            'customer': ['customer1', 'customer2', 'customer3'],
            'items_purchased': [['a', 'b'], ['d', 'c'], ['a', 'd']]
        }
        expected_output = [2.0, 0.0, 1.0]
        self.run_test(data, expected_output)

    def test_two(self):
        # Input data and expected output
        data = {
            'customer': ['customer1', 'customer2', 'customer3'],
            'items_purchased': [['a', 'b','c'], ['d', 'c'], ['x', 'y']]
        }
        expected_output = [2, 1, 0]
        self.run_test(data, expected_output)
    
    def test_three(self):
        # Input data and expected output
        data = {
            'customer': ['customer1', 'customer2', 'customer3','customer4'],
            'items_purchased': [['a', 'b','c'], ['d', 'c'], ['x', 'y'],['x','z']]
        }
        expected_output = [3.0,2.0, 0.0, 1.0]
        self.run_test(data, expected_output)

if __name__ == '__main__':
    unittest.main()
