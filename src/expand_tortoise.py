# Expands the Number of Tokens in a tortoise model, script still in testing

import torch
import torch.nn as nn
from dlas.models.audio.tts.unified_voice2 import UnifiedVoice

def expand_pretrained_model(pretrained_model_path, expanded_model_path, expanded_number_text_tokens):
    # Load the pretrained model
    pretrained_state_dict = torch.load(pretrained_model_path)
    pretrained_model = UnifiedVoice(
        layers=30,
        model_dim=1024,
        heads=16,
        max_text_tokens=402,
        max_mel_tokens=604,
        max_conditioning_inputs=2,
        mel_length_compression=1024,
        number_text_tokens=256,
        start_text_token=255,
        stop_text_token=0,
        number_mel_codes=8194,
        start_mel_token=8192,
        stop_mel_token=8193,
        train_solo_embeddings=False,
        use_mel_codes_as_input=True,
        checkpointing=True,
        average_conditioning_embeddings=False,
        freeze_everything_but_position_embeddings=False,
        tortoise_compat=True
    )
    pretrained_model.load_state_dict(pretrained_state_dict)

    # Create a new instance of the UnifiedVoice class with the expanded number of text tokens
    expanded_model = UnifiedVoice(
        layers=30,
        model_dim=1024,
        heads=16,
        max_text_tokens=402,
        max_mel_tokens=604,
        max_conditioning_inputs=2,
        mel_length_compression=1024,
        number_text_tokens=expanded_number_text_tokens,
        start_text_token=255,
        stop_text_token=0,
        number_mel_codes=8194,
        start_mel_token=8192,
        stop_mel_token=8193,
        train_solo_embeddings=False,
        use_mel_codes_as_input=True,
        checkpointing=True,
        average_conditioning_embeddings=False,
        freeze_everything_but_position_embeddings=False,
        tortoise_compat=True
    )
    # Update dimensions of relevant layers and tensors
    expanded_model.text_embedding = nn.Embedding(expanded_number_text_tokens, expanded_model.text_embedding.embedding_dim)
    expanded_model.text_head = nn.Linear(expanded_model.text_head.in_features, expanded_number_text_tokens)
    
    # Copy the weights from the pretrained model
    expanded_model.text_embedding.weight.data[:pretrained_model.text_embedding.weight.shape[0], :] = pretrained_model.text_embedding.weight.data
    expanded_model.text_head.weight.data[:pretrained_model.text_head.weight.shape[0], :] = pretrained_model.text_head.weight.data
    expanded_model.text_head.bias.data[:pretrained_model.text_head.bias.shape[0]] = pretrained_model.text_head.bias.data

    # Initialize the expanded portion of the weights
    expanded_model.text_embedding.weight.data[pretrained_model.text_embedding.weight.shape[0]:].normal_(mean=0.0, std=.02)
    expanded_model.text_head.weight.data[pretrained_model.text_head.weight.shape[0]:].normal_(mean=0.0, std=.02)
    expanded_model.text_head.bias.data[pretrained_model.text_head.bias.shape[0]:].normal_(mean=0.0, std=.02)
    
    # Load the matching weights from the pretrained model
    expanded_model_state_dict = expanded_model.state_dict()
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in expanded_model_state_dict and 'text_embedding' not in k and 'text_head' not in k}
    expanded_model_state_dict.update(pretrained_state_dict)
    expanded_model.load_state_dict(expanded_model_state_dict)

    req_grad = True
    # Freeze the loaded weights of the expanded model, including the original text embedding and text head weights
    for name, param in expanded_model.named_parameters():
        if 'text_embedding' not in name and 'text_head' not in name:
            param.requires_grad = req_grad
        elif 'text_embedding' in name:
            param.requires_grad = req_grad
        elif 'text_head' in name:
            param.requires_grad = req_grad

    # Initialize the expanded text embeddings
    expanded_text_embeddings = nn.Embedding(expanded_number_text_tokens, 1024)
    expanded_text_embeddings.weight.data[:256] = expanded_model.text_embedding.weight.data[:256]
    expanded_text_embeddings.weight.requires_grad = req_grad  # Freeze the original text embedding weights
    expanded_model.text_embedding = expanded_text_embeddings

    # Initialize the expanded text head parameters
    expanded_text_head_weight = nn.Parameter(torch.randn(expanded_number_text_tokens, 1024))
    expanded_text_head_weight.data[:256] = expanded_model.text_head.weight.data[:256]
    expanded_text_head_weight.data[:256].requires_grad = req_grad  # Freeze the original text head weights
    expanded_model.text_head.weight = expanded_text_head_weight

    expanded_text_head_bias = nn.Parameter(torch.randn(expanded_number_text_tokens))
    expanded_text_head_bias.data[:256] = expanded_model.text_head.bias.data[:256]
    expanded_text_head_bias.data[:256].requires_grad = req_grad  # Freeze the original text head bias
    expanded_model.text_head.bias = expanded_text_head_bias

    
    frozen_params = 0
    trainable_params = 0
    for name, param in expanded_model.named_parameters():
        if not param.requires_grad:
            frozen_params += param.numel()
        else:
            trainable_params += param.numel()

    print(f"Number of frozen parameters: {frozen_params}")
    print(f"Number of trainable parameters: {trainable_params}")
    
    # Save the expanded model
    torch.save(expanded_model.state_dict(), expanded_model_path)
    

pretrained_model_path = "models/tortoise/autoregressive.pth"
expanded_number_text_tokens = 512
expanded_model_path = f"models/finetunes/autoregressive_expanded_{expanded_number_text_tokens}.pth"

expand_pretrained_model(pretrained_model_path, expanded_model_path, expanded_number_text_tokens)