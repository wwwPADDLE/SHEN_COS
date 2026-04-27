import torch
import torch.nn as nn

class DINOv3BackboneWrapper(nn.Module):
    """
    Wrapper to replace SAM's ImageEncoderViT with DINOv3 backbone.
    - 输出特征仍然是 (B, out_chans, 24, 24)
    - 当 interm=True 时，额外返回所有层的 24x24 特征列表，用于 OFM 融合
    """
    def __init__(self,
                 backbone_name: str = "dinov3_vitl16",
                 pretrained: bool = True,
                 out_chans: int = 256,
                 img_size: int = 384,
                 patch_size: int = 16,
                 dinov3_repo_path: str | None = None,
                 weights: str | None = None,
                 freeze_backbone: bool = True,
                 num_layers: int | None = None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_chans = out_chans

        import os
        if dinov3_repo_path is not None:
            dinov3_repo_path = os.path.abspath(dinov3_repo_path)
        if isinstance(weights, str) \
           and not weights.startswith("http://") \
           and not weights.startswith("https://") \
           and not weights.startswith("file://"):
            weights = os.path.abspath(weights)

        # ---- import dinov3 ----
        if dinov3_repo_path:
            import sys
            if dinov3_repo_path not in sys.path:
                sys.path.append(dinov3_repo_path)
        try:
            from dinov3.hub.backbones import (
                dinov3_vitl16, dinov3_vitl16plus, dinov3_vit7b16, Weights as DWeights
            )
        except Exception as e:
            raise ImportError(
                f"Failed to import dinov3. "
                f"Set dinov3_repo_path or install dinov3 package. Error: {e}"
            )

        name = backbone_name.lower()
        if "7b" in name:
            build = dinov3_vit7b16
            default_weights = DWeights.LVD1689M
        elif "l16plus" in name or "vitl16plus" in name:
            build = dinov3_vitl16plus
            default_weights = DWeights.LVD1689M
        else:
            build = dinov3_vitl16
            default_weights = DWeights.LVD1689M

        kwargs = dict(pretrained=pretrained, weights=(weights or default_weights))
        self.backbone = build(**kwargs)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # ---- 维度信息 ----
        C = getattr(self.backbone, "embed_dim", None)
        if C is None:
            C = self.backbone.model.embed_dim if hasattr(self.backbone, "model") else None
        if C is None:
            raise RuntimeError("Cannot infer embed_dim from DINOv3 backbone.")
        self.in_chans = C

        # 统一 1x1 投影到 out_chans（256），供所有层共用
        self.proj = nn.Sequential(
            nn.Conv2d(C, out_chans, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_chans)
        )

        # 记录总层数（用于 get_intermediate_layers）
        if num_layers is None:
            num_layers = getattr(self.backbone, "num_layers", None)
            if num_layers is None and hasattr(self.backbone, "blocks"):
                num_layers = len(self.backbone.blocks)
        if num_layers is None:
            num_layers = 24   # 兜底：ViT-L 默认 24 层
        self.num_layers = num_layers

    # ---------- 工具函数 ----------

    def _extract_patch_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, L, C)，可能包含 cls / 其它 token
        返回：只包含 patch 的 (B, HW, C)，HW = (img_size//patch_size)^2
        """
        B, L, C = tokens.shape
        H = W = int(self.img_size // self.patch_size)
        HW = H * W  # 384//16=24 → 576

        if L == HW:
            # 已经是纯 patch tokens
            return tokens

        if L > HW:
            # 典型情况： [CLS] + [patch tokens] (+ 可能末尾还有别的)
            start = 1
            end = 1 + HW
            if end <= L:
                return tokens[:, start:end, :]
            else:
                raise RuntimeError(
                    f"Cannot slice {HW} patch tokens from length {L} "
                    f"(start={start}, end={end})."
                )

        # L < HW：连 24×24 都凑不齐，肯定异常
        raise RuntimeError(
            f"Not enough tokens for patch map: L={L}, required HW={HW}"
        )

    @torch.no_grad()
    def _tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, HW, C)
        return: (B, C, H, W)  with H=W=img_size/patch_size
        """
        B, HW, C = tokens.shape
        H = W = int(self.img_size // self.patch_size)
        assert HW == H * W, f"tokens HW={HW}, but H*W={H*W}"
        x = tokens.transpose(1, 2).contiguous().view(B, C, H, W)
        return x

    def _layer_tokens_to_feat(self, layer_tokens: torch.Tensor) -> torch.Tensor:
        """
        layer_tokens: (B, L, C) → (B, out_chans, 24,24)
        """
        patch_tokens = self._extract_patch_tokens(layer_tokens)  # (B, HW, C)
        fmap = self._tokens_to_map(patch_tokens)                 # (B, C, 24,24)
        out = self.proj(fmap)                                    # (B, 256,24,24)
        return out

    # ---------- 正式前向 ----------

    def forward(self, x: torch.Tensor, interm: bool = False):
        """
        Return:
          - features: (B, out_chans, 24,24)
          - interm_embeddings:
              * interm=False: []
              * interm=True : List[Tensor]，长度 = num_layers，
                              每个 (B, out_chans, 24,24)，从浅到深
        """
        if not interm:
            feats = self.backbone.forward_features(x)

            if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
                # 官方给的 patch tokens
                patch_tokens = feats["x_norm_patchtokens"]  # (B, HW, C) 或 (B,L,C)
                if patch_tokens.dim() == 3:
                    patch_tokens = self._extract_patch_tokens(patch_tokens)
            else:
                # 兜底：用 get_intermediate_layers 拿最后一层 tokens 自己抽 patch
                (last_tokens,) = self.backbone.get_intermediate_layers(
                    x, n=1, reshape=False, norm=True
                )  # (B, L, C)
                patch_tokens = self._extract_patch_tokens(last_tokens)

            fmap = self._tokens_to_map(patch_tokens)
            out = self.proj(fmap)
            return out, []
        else:
            tokens_all = self.backbone.get_intermediate_layers(
                x, n=self.num_layers, reshape=False, norm=True
            )  # List[(B, L, C)]

            all_feats = [self._layer_tokens_to_feat(t) for t in tokens_all]
            features = all_feats[-1]   # 最深一层
            return features, all_feats

