import unittest
import torch
import torch.nn.functional as F
from transformer import TransformerModel

class TestMulticlassClassification(unittest.TestCase):
    def setUp(self):
        # Define test parameters
        self.ntoken = 1000
        self.d_model = 64
        self.nhead = 4
        self.d_hid = 128
        self.nlayers = 2
        self.dropout = 0.1
        self.num_classes = 5  # Set the number of classes for multiclass

        # Instantiate the model
        self.model = TransformerModel(self.ntoken, self.d_model, self.nhead, self.d_hid,
                                      self.nlayers, self.dropout, self.num_classes)

    def test_multiclass_output_shape(self):
        # Prepare test data
        batch_size = 8
        seq_len = 20
        input_tensor = torch.randint(0, self.ntoken, (seq_len, batch_size))

        # Forward pass
        output = self.model(input_tensor)

        # Check output shape
        self.assertEqual(output.shape, torch.Size([seq_len, batch_size, self.num_classes]))

    def test_multiclass_output_probabilities(self):
        # Prepare test data
        batch_size = 8
        seq_len = 20
        input_tensor = torch.randint(0, self.ntoken, (seq_len, batch_size))

        # Forward pass
        output = self.model(input_tensor)

        # Apply softmax along the class dimension
        softmax_output = F.softmax(output, dim=-1)

        # Check if output probabilities sum up to 1 along the class dimension
        probabilities_sum = softmax_output.sum(dim=-1)
        expected_sum = torch.ones_like(probabilities_sum)
        self.assertTrue(torch.allclose(probabilities_sum, expected_sum))


if __name__ == '__main__':
    unittest.main()
