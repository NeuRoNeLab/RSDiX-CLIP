from enum import Enum
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.nn import functional as nnf
from transformers import GPT2LMHeadModel, AutoConfig


class MappingType(Enum):
    """
    Enum class for different types of mapping layers for the captioning model, either MLP: "mlp" or Transformer:
    "transformer".
    """
    MLP = 'mlp'
    Transformer = 'transformer'


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module.

    Args:
        sizes (Tuple[int, ...]): A tuple of integers representing the input and output sizes of each layer in the MLP.
        bias (bool, optional): Whether to include bias in the linear layers. Defaults to True.
        act: (callable, optional): The activation function to be used between the linear layers. Defaults to nn.Tanh.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the MLP layers.
        """
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        """
        Initialize the MLP.

        Args:
            sizes (Tuple[int, ...]): A tuple of integers representing the input and output sizes of each layer in the MLP.
            bias (bool, optional): Whether to include bias in the linear layers. Defaults to True.
            act: (callable, optional): The activation function to be used between the linear layers. Defaults to nn.Tanh.
        """
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):

    def __init__(self, in_dim: int, h_dim: int, out_d: Optional[int] = None, act=nnf.relu, dropout: float = 0.0):
        """
        Point-Wise Multi-Layer Perceptron (MLP) submodule of the Transformer architecture, with 2 layers (and dropout).

        Args:
            in_dim (int): The input dimension.
            h_dim (int): The hidden dimension of the MLP.
            out_d (int, optional): The output dimension of the MLP. If None, it will be set to in_dim. Defaults to None.
            act: (callable, optional): The activation function to be used between the linear layers. Defaults to
                nnf.relu.
            dropout (float, optional): The dropout rate to be applied in the MLP. Defaults to 0.0.
        """
        self.act = act
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MlpTransformer. Applies the two layers, the corresponding non-linearities and dropout.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the MlpTransformer layers.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self: int, dim_ref: int, num_heads: int, bias: bool = True, dropout: float = 0.0):
        """
    Multi-Head Attention module.

    Args:
        dim_self (int): The dimension of the self-attention input.
        dim_ref (int): The dimension of the reference input (for queries and keys).
        num_heads (int): The number of attention heads.
        bias (bool, optional): Whether to include bias in the linear layers. Defaults to True.
        dropout (float, optional): The dropout rate to be applied in the attention layer. Defaults to 0.0.
    """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        """
        Forward pass of the MultiHeadAttention.

        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor, optional): The reference tensor for queries and keys. If None, y is set to x. Defaults to None.
            mask (torch.Tensor, optional): The mask tensor for masking the attention weights. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output tensor after applying the multi-head attention, and the
                                             attention weights.
        """
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        """
        Forward pass of the TransformerLayer with intermediate attention weights.

        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor, optional): The reference tensor for queries and keys. If None, y is set to x. Defaults to
             None.
            mask (torch.Tensor, optional): The mask tensor for masking the attention weights. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output tensor after applying the multi-head attention and the MLP,
                                               and the attention weights.
        """
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        """
        Forward pass of the TransformerLayer.

        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor, optional): The reference tensor for queries and keys. If None, y is set to x. Defaults to
                None.
            mask (torch.Tensor, optional): The mask tensor for masking the attention weights. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after applying the multi-head attention and the MLP.
        """

        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self,
                 dim_self,
                 dim_ref,
                 num_heads,
                 mlp_ratio=4.0,
                 bias=False,
                 dropout: float = 0.0,
                 act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        """
        Transformer Layer module.

        Args:
            dim_self (int): The dimension of the self-attention input.
            dim_ref (int): The dimension of the reference input (for queries and keys).
            num_heads (int): The number of attention heads.
            mlp_ratio (float, optional): The ratio of the hidden dimension to the self-attention input dimension.
                                         Defaults to 4.0.
            bias (bool, optional): Whether to include bias in the linear layers. Defaults to False.
            dropout (float, optional): The dropout rate to be applied in the transformer layer. Defaults to 0.0.
            act: (callable, optional): The activation function to be used in the MLP layers. Defaults to nnf.relu.
            norm_layer (nn.Module, optional): The normalization layer to be used in the transformer layer.
                                              Defaults to nn.LayerNorm.
        """

        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        """
        Forward pass of the Transformer module with intermediate attention weights.

        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor, optional): The reference tensor for queries and keys. If None, y is set to x. Defaults to
                None.
            mask (torch.Tensor, optional): The mask tensor for masking the attention weights. Defaults to None.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: The output tensor after applying the multi-head attention and the
                MLP, and a list of attention weights for each layer.
        """
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        """
        Forward pass of the Transformer module.

        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor, optional): The reference tensor for queries and keys. If None, y is set to x. Defaults to
                None.
            mask (torch.Tensor, optional): The mask tensor for masking the attention weights. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after applying the multi-head attention and the MLP.
        """
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:  # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self,
                 dim_self: int,
                 num_heads: int,
                 num_layers: int,
                 dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2.0,
                 act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm,
                 enc_dec: bool = False,
                 dropout: float = 0.0):
        """
        Transformer module for the CLIPCap captioning model.

        Args:
            dim_self (int): The dimension of the self-attention input.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of Transformer layers.
            dim_ref (Optional[int], optional): The dimension of the reference input (for queries and keys).
                                              If None, it will be set to dim_self. Defaults to None.
            mlp_ratio (float, optional): The ratio of the hidden dimension to the self-attention input dimension.
                                         Defaults to 2.0.
            act (callable, optional): The activation function to be used in the MLP layers. Defaults to nnf.relu.
            norm_layer (nn.Module, optional): The normalization layer to be used in the Transformer layer.
                                              Defaults to nn.LayerNorm.
            enc_dec (bool, optional): Whether to use the Transformer in encoder-decoder mode. If True, the num_layers
                                      will be multiplied by 2 to account for cross-attention layers. Defaults to False.
            dropout (float, optional): The dropout rate to be applied in the Transformer layers. Defaults to 0.0.
        """
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self,
                                               dim_ref,
                                               num_heads,
                                               mlp_ratio,
                                               act=act,
                                               norm_layer=norm_layer,
                                               dropout=dropout))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self,
                                               dim_self,
                                               num_heads,
                                               mlp_ratio,
                                               act=act,
                                               norm_layer=norm_layer,
                                               dropout=dropout))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self,
                                               dim_ref,
                                               num_heads,
                                               mlp_ratio,
                                               act=act,
                                               norm_layer=norm_layer,
                                               dropout=dropout))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        """
        Forward pass of the TransformerMapper module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the TransformerMapper layers.
        """
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self,
                 dim_clip: int,
                 dim_embedding: int,
                 prefix_length: int,
                 clip_length: int,
                 num_layers: int = 8,
                 dropout: float = 0.0):
        """
        TransformerMapper module for the CLIPCap captioning model.

        Args:
            dim_clip (int): The dimension of the input clip.
            dim_embedding (int): The dimension of the embedding.
            prefix_length (int): The length of the prefix.
            clip_length (int): The length of the clip.
            num_layers (int, optional): The number of Transformer layers. Defaults to 8.
            dropout (float, optional): The dropout rate to be applied in the Transformer layers. Defaults to 0.0.
        """
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers, dropout=dropout)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self,
                        batch_size: int,
                        device: torch.device) -> torch.Tensor:
        """
        Gets a dummy token tensor.

        Args:
            batch_size (int): The batch size.
            device (torch.device): The device on which the tensor will be created.

        Returns:
            torch.Tensor: A tensor of dummy tokens with shape (batch_size, prefix_length).
        """
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self,
                tokens: torch.Tensor,
                prefix: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        """
        Forward pass of the CLIPCap captioning model.

        Args:
            tokens (torch.Tensor): The input tensor containing the tokens.
            prefix (torch.Tensor): The input tensor containing the prefix.
            mask (torch.Tensor, optional): The mask tensor for masking the attention weights. Defaults to None.
            labels (torch.Tensor, optional): The tensor of labels for GPT-2 language modeling. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after applying the GPT-2 language model and the mapping layer.
        """
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self,
                 prefix_length: int,
                 clip_length: Optional[int] = None,
                 prefix_size: int = 512,
                 num_layers: int = 8,
                 mapping_type: MappingType = MappingType.MLP,
                 dropout_transformer: float = 0.0,
                 dropout_gpt2: Optional[float] = None):
        """
        CLIPCap captioning model.

        Args:
            prefix_length (int): The length of the prefix.
            clip_length (Optional[int], optional): The length of the clip. If None, it will be set to the prefix_length.
                                                   Defaults to None.
            prefix_size (int, optional): The size of the prefix. Defaults to 512.
            num_layers (int, optional): The number of Transformer layers. Defaults to 8.
            mapping_type (MappingType, optional): The type of mapping layer to be used. Defaults to MappingType.MLP.
            dropout_transformer (float, optional): The dropout rate to be applied in the Transformer layers. Defaults to
                0.0.
            dropout_gpt2 (float, optional): The dropout rate to be applied in the GPT-2 model. If None, the default
                GPT-2 model will be used. Defaults to None.
        """
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length

        if dropout_gpt2 is not None:
            gpt2_config = AutoConfig.from_pretrained('gpt2')
            gpt2_config.attn_dropout = dropout_gpt2
            gpt2_config.attn_dropout = dropout_gpt2
            # gpt2_config.embd_pdrop = dropout_gpt2
            self.gpt = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt2_config)
        else:
            self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                  clip_length, num_layers, dropout=dropout_transformer)


class ClipCaptionPrefix(ClipCaptionModel):
    """
    CLIPCap captioning model in which the only trainable parameters are the prefix ones.

    Methods:
        parameters(recurse: bool = True) -> Iterator[Tensor]:
            Returns an iterator over the model's parameters.

        train(mode: bool = True) -> None:
            Sets the model in train mode and freezes the GPT-2 model.
    """

    def parameters(self, recurse: bool = True):
        """
        Get an iterator over the model's parameters.

        Args:
            recurse (bool, optional): If True, returns the parameters of this model and its submodules recursively.
                                      Defaults to True.

        Returns:
            Iterator[Tensor]: An iterator over the model's parameters.
        """
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        """
        Sets the model in train mode and freezes the GPT-2 model.

        Args:
            mode (bool, optional): If True, sets the model in train mode. If False, sets the model in evaluation mode.
                                   Defaults to True.
        """
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self
