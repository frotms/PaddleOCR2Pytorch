def build_neck(config):
    from .db_fpn import DBFPN
    # from .east_fpn import EASTFPN
    # from .sast_fpn import SASTFPN
    from .rnn import SequenceEncoder
    support_dict = ['DBFPN', 'EASTFPN', 'SASTFPN', 'SequenceEncoder']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('neck only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class