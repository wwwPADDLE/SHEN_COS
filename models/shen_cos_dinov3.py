import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register
from .decoder.shen_cos_transformer import TwoWayTransformer as TwoWayTransformer_MaskDecoder_Edge
from .decoder.shen_cos_mask_decoder import MaskDecoder as MaskDecoder_Edge
from .dinov3_wrapper import DINOv3BackboneWrapper
from .gfcm import GlobalFeatureCorrelationModule as GFCM
from .spmam import SemanticPromptMambaAlignmentModule as SPMAM
from .ifmfm import IntraModalFrequencyDomainMambaFusionModule as IFMFM
from models.ovcamo_loss import edge_dice_loss
from .iou_loss import IOU
from typing import Any, Optional, Tuple
from dassl.utils import load_checkpoint

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

CAMO_PROMPTS = [
    "A photo of the camouflaged {}.",
    "A photo of the concealed {}.",
    "A photo of the {} camouflaged in the background.",
    "A photo of the {} concealed in the background.",
    "A photo of the {} camouflaged to blend in with its surroundings.",
    "A photo of the {} concealed to blend in with its surroundings.",
]

def get_prompt_template_by_name(name):
    if name == "camoprompts":
        template_set = CAMO_PROMPTS
    return template_set

def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)
    return iou.mean()

class BBCEWithLogitLoss(nn.Module):
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()
    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)
        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)
        return loss

class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )
    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    def forward(self, size: int) -> torch.Tensor:
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)


@register("shen_cos_dinov3")
class SHEN_COS(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder_name = encoder_mode.get("name", "dinov3").lower()
        if encoder_name != "dinov3":
            raise ValueError(
                f"This public version only supports DINOv3 backbone, "
                f"but got encoder_mode.name={encoder_name}"
            )

        num_layers = int(encoder_mode.get("num_layers", 24))
        self.image_encoder = DINOv3BackboneWrapper(
            backbone_name=encoder_mode.get("backbone", "dinov3_vitl16"),
            pretrained=True,
            out_chans=encoder_mode["out_chans"],
            img_size=inp_size,
            patch_size=encoder_mode.get("patch_size", 16),
            dinov3_repo_path=encoder_mode.get("dinov3_repo_path", None),
            weights=encoder_mode.get("weights", None),
            freeze_backbone=encoder_mode.get("freeze_backbone", True),
            num_layers=num_layers,
        )
        self.prompt_embed_dim = int(encoder_mode.get('prompt_embed_dim', 256))

        spmam_cfg = encoder_mode.get("spmam", encoder_mode.get("prompt_ofm", None))
        self.use_prompt_ofm = bool(spmam_cfg and spmam_cfg.get("enable", False))
        if self.use_prompt_ofm:
            self.prompt_ofm = SPMAM(
                dim=self.prompt_embed_dim,
                num_blocks=int(spmam_cfg.get("num_blocks", 1)),
                d_state=int(spmam_cfg.get("d_state", 16)),
                d_conv=int(spmam_cfg.get("d_conv", 4)),
                expand=int(spmam_cfg.get("expand", 2)),
                bimamba_type=spmam_cfg.get("bimamba_type", "v3"),
                detach_prompt=bool(spmam_cfg.get("detach_prompt", True)),
                prompt_at_tail=bool(spmam_cfg.get("prompt_at_tail", False)),
            )
        else:
            self.prompt_ofm = None

        ifmfm_cfg = encoder_mode.get("ifmfm", encoder_mode.get("freq_mamba", None))
        self.use_freq_mamba = bool(ifmfm_cfg and ifmfm_cfg.get("enable", False))
        if self.use_freq_mamba:
            self.freq_mamba_pair = IFMFM(
                dim=self.prompt_embed_dim,
                num_layers=num_layers,
                drop_path=float(ifmfm_cfg.get("drop_path", 0.0)),
                ssm_d_state=int(ifmfm_cfg.get("ssm_d_state", 16)),
                ssm_ratio=float(ifmfm_cfg.get("ssm_ratio", 2.0)),
                ssm_dt_rank=ifmfm_cfg.get("ssm_dt_rank", "auto"),
                ssm_act_layer=nn.SiLU,
                ssm_conv=int(ifmfm_cfg.get("ssm_conv", 3)),
                ssm_conv_bias=bool(ifmfm_cfg.get("ssm_conv_bias", True)),
                ssm_drop_rate=float(ifmfm_cfg.get("ssm_drop_rate", 0.0)),
                ssm_simple_init=bool(ifmfm_cfg.get("ssm_simple_init", False)),
                use_checkpoint=bool(ifmfm_cfg.get("use_checkpoint", False)),
                directions=ifmfm_cfg.get("directions", None),
            )
            effective_num_layers = self.freq_mamba_pair.num_pairs
        else:
            self.freq_mamba_pair = None
            effective_num_layers = num_layers

        gfcm_cfg = encoder_mode.get("gfcm", encoder_mode.get("ofm", None))
        self.use_ofm = bool(gfcm_cfg and gfcm_cfg.get("enable", False))
        if self.use_ofm:
            raw_layers = gfcm_cfg.get("layers", "all")
            if isinstance(raw_layers, str) and raw_layers.lower() == "all":
                self.ofm_layer_indices = list(range(effective_num_layers))
            elif isinstance(raw_layers, (list, tuple)):
                idx = []
                for v in raw_layers:
                    v = int(v)
                    if v >= 1 and v <= effective_num_layers:
                        idx.append(v - 1)
                    elif 0 <= v < effective_num_layers:
                        idx.append(v)
                self.ofm_layer_indices = sorted(set(idx))
            else:
                raise ValueError(f"Unsupported gfcm.layers: {raw_layers}")

            self.ofm_module = GFCM(
                dim=self.prompt_embed_dim,
                num_layers=len(self.ofm_layer_indices),
                d_state=int(gfcm_cfg.get("d_state", 16)),
                d_conv=int(gfcm_cfg.get("d_conv", 4)),
                expand=int(gfcm_cfg.get("expand", 2)),
                num_blocks=int(gfcm_cfg.get("num_blocks", 1)),
                bimamba_type=gfcm_cfg.get("bimamba_type", "v3"),
                use_base_z=bool(gfcm_cfg.get("use_base_z", True)),
                z_extra_mode=gfcm_cfg.get("z_extra_mode", "none"),
                z_extra2d_ks=gfcm_cfg.get("z_extra2d_ks", 3),
                z_extra2d_dilation=int(gfcm_cfg.get("z_extra2d_dilation", 1)),
            )
        else:
            self.ofm_layer_indices = []
            self.ofm_module = None

        self.mask_decoder = MaskDecoder_Edge(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer_MaskDecoder_Edge(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()
        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()
        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])
        self.sam_visual_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.LayerNorm(256),
        )
        self.sam_text_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
        )
        self.train_text_features = torch.load(
            "./datasets/ovcamo_info/TrainCamoPromptsTextFeaturesViTB-14-336.pth").to(
            self.device)
        self.test_text_features = torch.load(
            "./datasets/ovcamo_info/TestCamoPromptsTextFeaturesViTB-14-336.pth").to(
            self.device)

    def load_mapleAlphaCLIP(self, maple_clip_model, MaPLeAlphaCLIP_checkpoint=None):
        self.clip_model = maple_clip_model
        self.clip_model = self.clip_model.float()
        for k, p in self.clip_model.named_parameters():
            p.requires_grad = False
        self.clip_model.to(self.device)
        self.clip_model.load_text_features(self.train_text_features, self.test_text_features)
        if MaPLeAlphaCLIP_checkpoint != None:
            checkpoint = load_checkpoint(MaPLeAlphaCLIP_checkpoint)
            state_dict = checkpoint["state_dict"]
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            self.clip_model.load_state_dict(state_dict, strict=False)

    def set_input(self, input, mask, label_id, clip_image, clip_mask):
        self.input = input.to(self.device)
        self.gt_mask = mask.to(self.device)
        self.label_id = label_id.to(self.device)
        self.clip_image = clip_image.to(self.device)
        self.clip_mask = clip_mask.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def alpha_clip_process(self, image, alpha):
        if self.training:
            text_embeddings = self.train_text_features
        else:
            text_embeddings = self.test_text_features
        image_features = self.clip_model.visual(image, alpha)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        score = torch.matmul(image_features, text_embeddings.permute(1, 0))
        pred_1 = score.topk(1, dim=1)[1].squeeze(dim=1)
        smoothing = 0.1
        confidence = 1 - smoothing
        smooth_score = torch.zeros_like(score).to(score.device)
        smooth_score.fill_(smoothing)
        smooth_score.scatter_(1, pred_1.unsqueeze(1), confidence)
        smooth_score = smooth_score.unsqueeze(-1)
        output_text_features = (smooth_score * text_embeddings).sum(dim=1)
        return image_features.unsqueeze(1), output_text_features.unsqueeze(1), pred_1

    def maple_alpha_clip_process(self, image, alpha):
        image_features, text_features, pred_1, score = self.clip_model(image, alpha, self.training)
        return image_features, text_features, pred_1, score

    def _evolve_hierarchical_clues(self, features, interm_embeddings, sparse_embeddings_1):
        if (not self.use_ofm) or (self.ofm_module is None):
            return features, interm_embeddings, sparse_embeddings_1
        if not isinstance(interm_embeddings, (list, tuple)) or len(interm_embeddings) == 0:
            return features, interm_embeddings, sparse_embeddings_1
        layers_24 = list(interm_embeddings)
        k24 = len(layers_24)
        if self.use_prompt_ofm and (self.prompt_ofm is not None):
            selected_indices = list(range(k24))
            selected_layers = [layers_24[i] for i in selected_indices]
            conditioned_layers = self.prompt_ofm(selected_layers, sparse_embeddings_1)
            for i, layer_idx in enumerate(selected_indices):
                layers_24[layer_idx] = conditioned_layers[i]
            interm_embeddings = layers_24
        else:
            conditioned_layers = layers_24
        if self.use_freq_mamba and (self.freq_mamba_pair is not None):
            fused_pair_layers = self.freq_mamba_pair(layers_24)
        else:
            fused_pair_layers = layers_24
        gfcm_inputs = [fused_pair_layers[i] for i in self.ofm_layer_indices]
        fused_features = self.ofm_module(gfcm_inputs)
        fused_features = fused_features + features
        return fused_features, interm_embeddings, sparse_embeddings_1

    def forward(self):
        bs = 1
        image_feat_1, text_feat_1, pred_1, score = self.maple_alpha_clip_process(
            self.clip_image, self.clip_mask
        )
        image_feat_1 = self.sam_visual_proj(image_feat_1)
        text_feat_1 = self.sam_text_proj(text_feat_1)
        sparse_embeddings_1 = torch.cat((image_feat_1, text_feat_1), dim=1)
        features, interm_embeddings = self.image_encoder(self.input, interm=True)
        features, interm_embeddings, sparse_embeddings_1 = self._evolve_hierarchical_clues(
            features, interm_embeddings, sparse_embeddings_1
        )
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        image_pe = self.get_dense_pe()
        if hasattr(self.mask_decoder, "set_debug"):
            self.mask_decoder.set_debug(False)
        low_res_masks, low_res_edges, iou_predictions = self.mask_decoder(
            image_embeddings=features,
            interm_embeddings=interm_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings_1,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        masks1 = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        edges1 = self.postprocess_masks(low_res_edges, self.inp_size, self.inp_size)
        self.pred_mask = masks1
        self.pred_edge = edges1

    def infer(self, input, clip_image, clip_zero_mask):
        bs = 1
        image_feat_1, text_feat_1, pred_1, score = self.maple_alpha_clip_process(clip_image, clip_zero_mask)
        image_feat_1 = self.sam_visual_proj(image_feat_1)
        text_feat_1 = self.sam_text_proj(text_feat_1)
        sparse_embeddings_1 = torch.cat((image_feat_1, text_feat_1), dim=1)
        features, interm_embeddings = self.image_encoder(input, interm=True)
        features, interm_embeddings, sparse_embeddings_1 = self._evolve_hierarchical_clues(
            features, interm_embeddings, sparse_embeddings_1
        )
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        image_pe = self.get_dense_pe()
        if hasattr(self.mask_decoder, "set_debug"):
            self.mask_decoder.set_debug(False)
        md_out = self.mask_decoder(
            image_embeddings=features,
            interm_embeddings=interm_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings_1,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        if isinstance(md_out, tuple) and len(md_out) == 4:
            low_res_masks, low_res_edges, iou_predictions, _ = md_out
        else:
            low_res_masks, low_res_edges, iou_predictions = md_out
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks

    def infer_test(self, input, clip_image, clip_zero_mask):
        bs = input.shape[0]
        image_feat_1, text_feat_1, pred_1, score = self.maple_alpha_clip_process(clip_image, clip_zero_mask)
        image_feat_1 = self.sam_visual_proj(image_feat_1)
        text_feat_1 = self.sam_text_proj(text_feat_1)
        sparse_embeddings_1 = torch.cat((image_feat_1, text_feat_1), dim=1)
        features, interm_embeddings = self.image_encoder(input, interm=True)
        features, interm_embeddings, sparse_embeddings_1 = self._evolve_hierarchical_clues(
            features, interm_embeddings, sparse_embeddings_1
        )
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        image_pe = self.get_dense_pe()
        if hasattr(self.mask_decoder, "set_debug"):
            self.mask_decoder.set_debug(False)
        md_out = self.mask_decoder(
            image_embeddings=features,
            interm_embeddings=interm_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings_1,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        if isinstance(md_out, tuple) and len(md_out) == 4:
            low_res_masks, low_res_edges, iou_predictions, _ = md_out
        else:
            low_res_masks, low_res_edges, iou_predictions = md_out
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def backward_G(self):
        self.loss_dict = {}
        self.loss_G = 0
        self.loss_dict["loss_mask"] = self.criterionBCE(self.pred_mask, self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_dict["loss_mask"] += _iou_loss(self.pred_mask, self.gt_mask)
        self.loss_G += self.loss_dict["loss_mask"]
        with torch.no_grad():
            edge_ks = 5
            eroded_mask = -F.max_pool2d(-self.gt_mask, kernel_size=edge_ks, stride=1, padding=edge_ks // 2)
            dilated_mask = F.max_pool2d(self.gt_mask, kernel_size=edge_ks, stride=1, padding=edge_ks // 2)
            edge = dilated_mask - eroded_mask
            edge = edge.gt(0).float()
        self.gt_edge = edge
        self.loss_dict["loss_edge"] = edge_dice_loss(self.pred_edge, self.gt_edge)
        self.loss_G += self.loss_dict["loss_edge"]
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward_G()
        self.optimizer.step()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
