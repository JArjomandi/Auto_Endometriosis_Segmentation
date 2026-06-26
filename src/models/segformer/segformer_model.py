from transformers import SegformerForSemanticSegmentation


def build_segformer_model(
    pretrained_model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels: int = 1,
):
    id2label = {0: "lesion"}
    label2id = {"lesion": 0}

    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    return model