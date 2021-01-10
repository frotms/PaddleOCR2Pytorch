def build_backbone(config, model_type):
    if model_type == 'det':
        from .det_mobilenet_v3 import MobileNetV3
        from .det_resnet_vd import ResNet
        # from .det_resnet_vd_sast import ResNet_SAST
        support_dict = ['MobileNetV3', 'ResNet', 'ResNet_SAST']
    elif model_type == 'rec' or model_type == 'cls':
        from .rec_mobilenet_v3 import MobileNetV3
        from .rec_resnet_vd import ResNet
        support_dict = ['MobileNetV3', 'ResNet', 'ResNet_FPN']
    else:
        raise NotImplementedError

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'when model typs is {}, backbone only support {}'.format(model_type,
                                                                 support_dict))
    module_class = eval(module_name)(**config)
    return module_class