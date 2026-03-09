"""
FHE-traceable pipeline wrapper for CryptoFace inference.

PerImagePipeline wraps N backbones + N linear layers + normalization into
a single on.Module with an FX-traceable forward signature. Orion requires
concrete positional arguments (no *args) on the traced forward, so the
method is generated dynamically per patch count and bound to a per-N subclass.
"""

import orion.nn as on


class PerImagePipeline(on.Module):
    """
    Full pipeline: N patches → [Backbone+Linear]×N → Aggregate → L2Normalize → Embedding.

    Supports arbitrary N (net4=4, net9=9, net16=16).
    FX tracing cannot iterate over *args Proxy objects, so we dynamically generate
    a forward method with exactly N explicit positional args and bind it to a
    per-N subclass at __init__ time. final_level=1 in config replaces dummy multiply.
    """

    # Placeholder satisfies on.Module's abstract-forward requirement at class definition time.
    # The real forward (with N explicit args) is generated and bound in __init__.
    def forward(self):
        raise NotImplementedError("forward is generated dynamically in __init__")

    def __init__(self, backbones, linears, normalization):
        super().__init__()
        N = len(backbones)
        for i, (backbone, linear) in enumerate(zip(backbones, linears)):
            setattr(self, f'backbone{i}', backbone)
            setattr(self, f'linear{i}', linear)
        self.normalization = normalization

        # Generate forward(self, patch0, ..., patch{N-1}) with N explicit args.
        # FX tracing cannot iterate over *args Proxy objects, so explicit positional
        # params are required. We bind the generated method to a per-N subclass so
        # the tracer sees the concrete signature on the class (not just the instance).
        args_str = ", ".join(f"patch{i}" for i in range(N))
        body = "\n        ".join(
            f"feat{i} = self.linear{i}(self.backbone{i}(patch{i}))"
            for i in range(N)
        )
        feat_sum = " + ".join(f"feat{i}" for i in range(N))
        src = (
            f"def forward(self, {args_str}):\n"
            f"        {body}\n"
            f"        return self.normalization({feat_sum})"
        )
        ns = {}
        exec(src, ns)
        self.__class__ = type(
            f'PerImagePipeline{N}',
            (PerImagePipeline,),
            {'forward': ns['forward']}
        )
