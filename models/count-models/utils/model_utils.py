"""Custom model architectures for count extraction."""

import torch
import numpy as np
from torch.nn import Linear, PoissonNLLLoss
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class PoissonRegressionModel(torch.nn.Module):
    """
    DistilBERT-based Poisson regression model for count prediction.
    
    Uses Poisson NLL loss which is appropriate for modeling count data.
    """
    
    def __init__(self, pretrained_model_name, num_labels=1):
        """
        Initialize the model.
        
        Args:
            pretrained_model_name: HuggingFace model identifier (e.g., 'distilbert-base-cased')
            num_labels: Number of output labels (default: 1 for single count prediction)
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.regressor = Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask for padded sequences
            labels: True count values (for computing loss during training)
            
        Returns:
            SequenceClassifierOutput with loss and logits
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0][:, 0, :]  # [CLS] token
        # Predict log-rate (log of Poisson mean) for numerical stability
        log_rate = self.regressor(sequence_output).squeeze(-1)
        # Convert to mean count (mu) for outputs
        mu = torch.exp(log_rate)
        
        loss = None
        if labels is not None:
            # Use Poisson loss with log_input=True (inputs are log of mean)
            loss_fct = PoissonNLLLoss(log_input=True)
            loss = loss_fct(log_rate, labels.float())
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=mu,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None
        )


def extract_qa_answer(start_logits, end_logits, input_ids, tokenizer, n_best=1):
    """
    Extract answer text from QA model predictions.
    
    Given start and end logits from a QA model, this function finds the most likely
    answer span and decodes it to text.
    
    Args:
        start_logits: Start position logits from QA model (shape: [batch_size, seq_len])
        end_logits: End position logits from QA model (shape: [batch_size, seq_len])
        input_ids: Input token IDs (shape: [batch_size, seq_len])
        tokenizer: HuggingFace tokenizer used for encoding
        n_best: Number of best answers to consider
        
    Returns:
        List of answer texts (one per example in batch)
    """
    # Convert to numpy for easier manipulation
    start_logits = start_logits.cpu().numpy() if torch.is_tensor(start_logits) else start_logits
    end_logits = end_logits.cpu().numpy() if torch.is_tensor(end_logits) else end_logits
    input_ids = input_ids.cpu().numpy() if torch.is_tensor(input_ids) else input_ids
    
    answers = []
    batch_size = start_logits.shape[0]
    
    for i in range(batch_size):
        # Get top n_best start and end positions
        start_indexes = np.argsort(start_logits[i])[-n_best:][::-1]
        end_indexes = np.argsort(end_logits[i])[-n_best:][::-1]
        
        # Find best valid span (end >= start)
        best_score = float('-inf')
        best_start = 0
        best_end = 0
        
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Check if span is valid
                if end_index >= start_index and (end_index - start_index) <= 30:  # Max answer length
                    score = start_logits[i][start_index] + end_logits[i][end_index]
                    if score > best_score:
                        best_score = score
                        best_start = start_index
                        best_end = end_index
        
        # Extract answer tokens
        answer_tokens = input_ids[i][best_start:best_end + 1]
        answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        answers.append(answer_text)
    
    return answers

