from .SetBert import BertSetModel
from transformers import AutoConfig
from .graph_base import torch, nn, F, Optional, GCN, Tensor, Adj, Batch



class DynamicEdge(nn.Module):
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        in_channels: Optional[int] = 1041,
        num_nodes: Optional[int] = None,
        out_channels: Optional[int] = 128,
        num_layers: Optional[int] = 3, 
        dropout: Optional[float] = 0.3, 
        act: Optional[F.relu] = F.relu,
    ):
        super().__init__()
        # Define transformer config and model
        if model_path is None:
            model_path = 'bert-base-uncased'
        
        self.config = AutoConfig.from_pretrained(model_path)
        self.config.num_hidden_layers = num_layers # self-define BERT transformer layer
        self.embed_dim = self.config.hidden_size
        self.seq2seq = BertSetModel.from_pretrained(model_path, config=self.config)

        # Self-defined graph position embedding
        self.wpe = nn.Embedding(
            self.config.max_position_embeddings, 
            self.embed_dim,
        )
        
        # Adapt vocabulary to node space
        if num_nodes is not None:
            self.seq2seq.resize_token_embeddings(num_nodes) 

        # Define graph encoder
        self.shared_encoder = GCN(
            in_channels=in_channels,
            hidden_channels=out_channels,
            out_channels=self.embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
        )


    def forward(self, x: Tensor, edge_index: Adj, batch: Batch, num_graphs: int):
        # Graph encode
        e_ = self.shared_encoder(x, edge_index) # |V| X E

        # Temporal-attentive Learning w/ Graph position embedding
        position_ids = torch.arange(0, num_graphs, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).view(-1, num_graphs)
        graph_embed = self.wpe(position_ids).squeeze(0) # |G| X E
        node_embed = torch.stack([graph_embed[graphid] for graphid in batch], dim=0) # |V| X E
        e_ = e_ + node_embed # |V| X E

        # Seq transform (BERT encoder)
        outputs = self.seq2seq.encoder(e_.unsqueeze(0))
        h = outputs.last_hidden_state.squeeze(0) # |V| X E
        return h