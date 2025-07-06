from lavis import registry


def load_model(
    name="blip2_t5_instruct",
    model_type="flant5xl",
):
    model_cls = registry.get_model_class(name)
    model = model_cls.from_pretrained(model_type=model_type)
    return model_cls
