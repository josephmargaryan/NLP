import unittest
import torch
from transformer import TransformerModel

class TestTransformerModel(unittest.TestCase):
    def setUp(self):
        # Define test parameters
        self.ntoken = 1000
        self.d_model = 64
        self.nhead = 4
        self.d_hid = 128
        self.nlayers = 2
        self.dropout = 0.1
        self.num_classes = 1

        # Instantiate the model
        self.model = TransformerModel(self.ntoken, self.d_model, self.nhead, self.d_hid,
                                      self.nlayers, self.dropout, self.num_classes)

    def test_forward_pass(self):
        # Prepare test data
        batch_size = 8
        seq_len = 20
        input_tensor = torch.randint(0, self.ntoken, (seq_len, batch_size))

        # Forward pass
        output = self.model(input_tensor)

        # Check output shape
        self.assertEqual(output.shape, torch.Size([seq_len, batch_size, self.num_classes]))

    def test_output_range_binary_classification(self):
        # Prepare test data
        batch_size = 8
        seq_len = 20
        input_tensor = torch.randint(0, self.ntoken, (seq_len, batch_size))

        # Forward pass
        output = self.model(input_tensor)

        # Check if output values are within [0, 1] for binary classification
        self.assertTrue((output >= 0).all() and (output <= 1).all())

if __name__ == '__main__':
    unittest.main()
