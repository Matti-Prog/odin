from contextlib import nullcontext
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from torch import Tensor
from .transformer.transformer_model_full import DenoiserTransformerFull
from .transformer.transformer_model_no_pos_enc import DenoiserTransformerNoPosEnc
from .transformer.transformer_model_no_local import DenoiserTransformerNoLocal

class PointCloudModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        model_type: str = 'transformer_full',
        embed_dim: int = 1024,
        num_points: int = 689,
    ):
        super().__init__()
        self.model_type = model_type
        
        if self.model_type == 'transformer_full':
            self.autocast_context = nullcontext()
            self.model = DenoiserTransformerFull(
                dim_hidden=embed_dim,
                dim_timestep_embed=embed_dim,
                num_points=num_points,
                global_feature_dim=256,
                point_feature_dim=128,
                num_layers=4,
                nhead=8,
                dim_feedforward=512
            )
        
        elif self.model_type == 'transformer_no_pos_enc':
            self.autocast_context = nullcontext()
            self.model = DenoiserTransformerNoPosEnc(
                dim_hidden=embed_dim,
                dim_timestep_embed=embed_dim,
                num_points=num_points,
                global_feature_dim=256,
                point_feature_dim=128,
                num_layers=4,
                nhead=8,
                dim_feedforward=512
            )
        
        elif self.model_type == 'transformer_no_local':
            self.autocast_context = nullcontext()
            self.model = DenoiserTransformerNoLocal(
                dim_hidden=embed_dim,
                dim_timestep_embed=embed_dim,
                num_points=num_points,
                global_feature_dim=256,
                point_feature_dim=128,
                num_layers=4,
                nhead=8,
                dim_feedforward=512
            )
        
        else:
            raise NotImplementedError()
        
    def forward(
        self,
        coords: Tensor,
        global_features: Tensor,
        local_features: Tensor,
        t: Tensor,
    ) -> Tensor:
        with self.autocast_context:
            if self.model_type == "transformer_full":
                return self.model(
                    coords,
                    global_features,
                    local_features,
                    t,
                )
            return self.model(coords, global_features, local_features, t)
