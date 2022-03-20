import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch

from ptstructure.vqa.pytorchnlp.transformers import LayoutXLMModel, LayoutXLMTokenizer, LayoutXLMForRelationExtraction


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    # yapf: disable
    parser.add_argument("--model_name_or_path",
                        default=None, type=str, required=False,)
    parser.add_argument("--ser_model_type",
                        default='LayoutXLM', type=str)
    parser.add_argument("--re_model_name_or_path",
                        default=None, type=str, required=True,)
    parser.add_argument("--train_data_dir", default=None,
                        type=str, required=False,)
    parser.add_argument("--train_label_path", default=None,
                        type=str, required=False,)
    parser.add_argument("--eval_data_dir", default=None,
                        type=str, required=False,)
    parser.add_argument("--eval_label_path", default=None,
                        type=str, required=False,)
    parser.add_argument("--output_dir", default=None, type=str, required=False,)
    parser.add_argument("--max_seq_length", default=512, type=int,)
    parser.add_argument("--evaluate_during_training", action="store_true",)
    parser.add_argument("--num_workers", default=8, type=int,)
    parser.add_argument("--per_gpu_train_batch_size", default=8,
                        type=int, help="Batch size per GPU/CPU for training.",)
    parser.add_argument("--per_gpu_eval_batch_size", default=8,
                        type=int, help="Batch size per GPU/CPU for eval.",)
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.",)
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="Weight decay if we apply some.",)
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.",)
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.",)
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.",)
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.",)
    parser.add_argument("--eval_steps", type=int, default=10,
                        help="eval every X updates steps.",)
    parser.add_argument("--seed", type=int, default=2048,
                        help="random seed for initialization",)

    parser.add_argument("--rec_model_dir", default=None, type=str, )
    parser.add_argument("--det_model_dir", default=None, type=str, )
    parser.add_argument(
        "--label_map_path", default="./labels/labels_ser.txt", type=str, required=False, )
    parser.add_argument("--infer_imgs", default=None, type=str, required=False)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--ocr_json_path", default=None,
                        type=str, required=False, help="ocr prediction results")

    # yapf: enable
    args = parser.parse_args()
    return args


class LayoutXLMREConverter:
    def __init__(self, args):
        self.args = args
        # args["init_class"] = 'LayoutLMModel'
        self.max_seq_length = args.max_seq_length

        self.net = LayoutXLMForRelationExtraction.from_pretrained(
            args.re_model_name_or_path)
        self.net.eval()

        save_name = os.path.join(args.re_model_name_or_path, 'model_state.pth')
        self._save_pytorch_weights(save_name)

    def _save_pytorch_weights(self, weights_path):
        try:
            torch.save(self.net.state_dict(), weights_path, _use_new_zipfile_serialization=False)
        except:
            torch.save(self.net.state_dict(), weights_path) # _use_new_zipfile_serialization=False for torch>=1.6.0
        print('model is saved: {}'.format(weights_path))



if __name__ == '__main__':
    args = parse_args()
    # loop for infer
    layoutXLM_ser = LayoutXLMREConverter(args)

    print('done.')