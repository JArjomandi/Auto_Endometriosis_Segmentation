from transformers import SegformerForSemanticSegmentation


def build_segformer_model(
    pretrained_model_name: str = "nvidia/mit-b2",
    num_labels: int = 1,
):
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    return model