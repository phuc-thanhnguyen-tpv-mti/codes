import torch
import torch.nn as nn
from transformers import AutoConfig, DistilBertModel

import torch
import torch.nn as nn
from transformers import AutoConfig, DistilBertModel

class SchemaItemClassifier(nn.Module):
    def __init__(self, model_name_or_path, mode):
        super(SchemaItemClassifier, self).__init__()
        if mode in ["eval", "test"]:
            # load config
            config = AutoConfig.from_pretrained(model_name_or_path)
            # randomly initialize model's parameters according to the config
            self.plm_encoder = DistilBertModel(config)
        elif mode == "train":
            self.plm_encoder = DistilBertModel.from_pretrained(model_name_or_path)
        else:
            raise ValueError()

        self.plm_hidden_size = self.plm_encoder.config.hidden_size  # Use the original hidden size (768)

        # column cls head
        self.column_info_cls_head = nn.Linear(self.plm_hidden_size, 2)

        # column 1D convolutional layer
        self.column_info_conv = nn.Conv1d(in_channels=self.plm_hidden_size, out_channels=self.plm_hidden_size, kernel_size=3, padding=1)

        # table cls head
        self.table_name_cls_head = nn.Linear(self.plm_hidden_size, 2)

        # table 1D convolutional layer
        self.table_name_conv = nn.Conv1d(in_channels=self.plm_hidden_size, out_channels=self.plm_hidden_size, kernel_size=3, padding=1)

        # activation function
        self.relu = nn.ReLU()

        # dropout function, p=0.1 to reduce complexity
        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        batch_aligned_column_info_ids,
        batch_aligned_table_name_ids,
        batch_column_number_in_each_table,
    ):
        batch_size = encoder_input_ids.shape[0]

        encoder_output = self.plm_encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            return_dict=True
        )  # encoder_output["last_hidden_state"].shape = (batch_size x seq_length x hidden_size)

        batch_table_name_cls_logits, batch_column_info_cls_logits = [], []

        # handle each data in current batch
        for batch_id in range(batch_size):
            column_number_in_each_table = batch_column_number_in_each_table[batch_id]
            sequence_embeddings = encoder_output["last_hidden_state"][batch_id, :, :]  # (seq_length x hidden_size)

            # obtain table ids for each table
            aligned_table_name_ids = batch_aligned_table_name_ids[batch_id]
            # obtain column ids for each column
            aligned_column_info_ids = batch_aligned_column_info_ids[batch_id]

            table_name_embedding_list, column_info_embedding_list = [], []

            # obtain table embedding via 1D conv + ReLU
            for table_name_ids in aligned_table_name_ids:
                table_name_embeddings = sequence_embeddings[table_name_ids, :]
                table_name_embeddings = table_name_embeddings.transpose(0, 1).unsqueeze(0)  # (1, hidden_size, seq_length)
                table_name_embedding = self.relu(self.table_name_conv(table_name_embeddings)).squeeze(0).transpose(0, 1)
                table_name_embedding_list.append(torch.mean(table_name_embedding, dim=0, keepdim=True))

            table_name_embeddings_in_one_db = torch.cat(table_name_embedding_list, dim=0)

            # obtain column embedding via 1D conv + ReLU
            for column_info_ids in aligned_column_info_ids:
                column_info_embeddings = sequence_embeddings[column_info_ids, :]
                column_info_embeddings = column_info_embeddings.transpose(0, 1).unsqueeze(0)  # (1, hidden_size, seq_length)
                column_info_embedding = self.relu(self.column_info_conv(column_info_embeddings)).squeeze(0).transpose(0, 1)
                column_info_embedding_list.append(torch.mean(column_info_embedding, dim=0, keepdim=True))

            column_info_embeddings_in_one_db = torch.cat(column_info_embedding_list, dim=0)

            # calculate table 0-1 logits
            table_name_cls_logits = self.dropout(self.relu(self.table_name_cls_head(table_name_embeddings_in_one_db)))

            # calculate column 0-1 logits
            column_info_cls_logits = self.dropout(self.relu(self.column_info_cls_head(column_info_embeddings_in_one_db)))

            batch_table_name_cls_logits.append(table_name_cls_logits)
            batch_column_info_cls_logits.append(column_info_cls_logits)

        return {
            "batch_table_name_cls_logits": batch_table_name_cls_logits,
            "batch_column_info_cls_logits": batch_column_info_cls_logits
        }


# Instantiate the model
model = SchemaItemClassifier("distilbert-base-uncased", "train")

# Calculate the total number of parameters
total_params = sum(p.numel() for p in model.parameters())

# Print the total number of parameters
print(f"Total number of parameters: {total_params}")