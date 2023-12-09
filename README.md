# TabMT-SyntheticNetworkIntrusionData
A from-scratch implementation of TabMT for tabular network intrusion data.

# Experiments
1) Replace ordered embeddings for continuous values with linear layers.
2) Attempt to learn simple embeddings for each unique port number.
3) Set masking probability of all labels to 0 during training for better conditional generation.
