ncut visualization:
在MyFastSAM的init中最后加上这段：

        from segment_anything.modeling.image_encoder import (
            window_partition,
            window_unpartition,
        )
        def new_block_forward(self, x: torch.Tensor) -> torch.Tensor:
            shortcut = x
            x = self.norm1(x)
            # Window partition
            if self.window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, self.window_size)

            x = self.attn(x)
            # Reverse window partition
            if self.window_size > 0:
                x = window_unpartition(x, self.window_size, pad_hw, (H, W))
            self.attn_output = x.clone()

            x = shortcut + x
            mlp_outout = self.mlp(self.norm2(x))
            self.mlp_output = mlp_outout.clone()
            x = x + mlp_outout
            self.block_output = x.clone()

            return x
        setattr(self.lora_sam.image_encoder.blocks[0].__class__, "forward", new_block_forward)

forward改成：

    def forward(self, batched_input):
        x = torch.stack([self.lora_sam.preprocess(x) for x in batched_input])
        out = self.lora_sam.image_encoder(x)

        attn_outputs, mlp_outputs, block_outputs = [], [], []
        for i, blk in enumerate(self.lora_sam.image_encoder.blocks):
            attn_outputs.append(blk.attn_output)
            mlp_outputs.append(blk.mlp_output)
            block_outputs.append(blk.block_output)
            # print(f"block {i} attn_output shape: {blk.attn_output.shape}")
            # print(f"block {i} mlp_output shape: {blk.mlp_output.shape}")
            # print(f"block {i} block_output shape: {blk.block_output.shape}")
        attn_outputs = torch.stack(attn_outputs)
        mlp_outputs = torch.stack(mlp_outputs)
        block_outputs = torch.stack(block_outputs)
        return attn_outputs, mlp_outputs, block_outputs
