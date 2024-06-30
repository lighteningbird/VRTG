import json, numpy as np, random, os, torch, hashlib, argparse, pickle, copy, sklearn, string, time, re, shlex
from tqdm import tqdm
from typing import Union, List
from torch.utils.data.dataset import Dataset
from tree_sitter import Language, Parser
from sklearn.metrics import recall_score, precision_score, f1_score
from transformers import RobertaForMaskedLM, RobertaTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from python_parser.run_parser import get_identifiers, remove_comments_and_docstrings, is_valid_variable_name, get_example, get_example_batch
from src.parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript, DFG_c, remove_comments_and_docstrings, tree_to_token_index, index_to_code_token


_available_name_path_list = ['microsoft/graphcodebert-base', 'microsoft/codebert-base', 'roberta-base', 'microsoft/codebert-base-mlm', 'Salesforce/codet5-base', 'Salesforce/codet5-base-multi-sum']
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def count_file_line(_filename):
    if not os.path.exists(_filename):
        return 0
    with open(_filename, 'r', buffering=1) as f:
        num_lines = sum(1 for line in f)
    return num_lines
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_status_map = {
    'truth label mismatch': -1,
    'fail': -2,
    'skip in reatk': -3,
    'success': 1,
}
_task_lang_map = {
    'authorship_attribution': 'python',
    'clone_detection': 'java',
    'code_classification': 'java',
    'vulnerability_detection': 'c'
}
_lang2parser = {}
_dfg_lang2function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript,
    'c': DFG_c
}
for lang in _dfg_lang2function:
    # LANGUAGE = Language(f'{os.path.dirname(os.path.abspath(__file__))}/../parser/my-languages.so', lang)
    # os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'parser', 'my-languages.so'))
    LANGUAGE = Language(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'parser', 'my-languages.so')), lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, _dfg_lang2function[lang]]
    _lang2parser[lang] = parser

class InputCodeFeature(object):
    '''single code feature for model input'''
    def __init__(self, _input_id_list: list, _label: int, _idx_info: Union[int, List]):
        '''args and token is useless'''
        self._input_id_list = _input_id_list
        self._idx_info = _idx_info
        self._label = _label
class InputCodeGraphFeature(object):
    '''single code graph feature for model input'''
    def __init__(self, _input_id_list: list, _label: int, 
                 _idx_info: Union[int, List],
                 _position_idx_info, _dfg2code_info, _dfg2dfg_info):
        self._input_id_list = _input_id_list
        self._idx_info = _idx_info
        self._label = _label
        self._position_idx_info = _position_idx_info
        self._dfg2code_info = _dfg2code_info
        self._dfg2dfg_info = _dfg2dfg_info
class InputDataset:
    @staticmethod
    def load_code_list(_args, _logger, _path_prefix=''):
        if _args.task_type in ['authorship_attribution', 'clone_detection']:
            _attack_source_file_path = _args.code_txt_path
        elif _args.task_type in ['code_classification', 'vulnerability_detection']:
            _attack_source_file_path = _args.code_jsonl_path
        return InputDataset.load_code_list_from_file(_args, _attack_source_file_path, _logger, _path_prefix)
    @staticmethod
    def split_retraining_dataset(_args, _logger):
        # target split file: tg/datasets/shared/{task_type}/retraining_{split}_index.json, split=train/test
        _split_train_path = _args.retraining_train_index_path
        _split_test_path = _args.retraining_test_index_path
        # skip if both exists
        if os.path.exists(_split_train_path) and os.path.exists(_split_test_path):
            # check len, if both not empty
            _train_index = json.loads(open(_split_train_path, 'r').read())
            _test_index = json.loads(open(_split_test_path, 'r').read())
            if len(_train_index) > 0 and len(_test_index) > 0:
                _logger.info(f'skip split retraining dataset: {_args.task_type}')
                return
        if _args.task_type == 'authorship_attribution':
            # get line of code_txt_path
            _line_count = count_file_line(_args.code_txt_path)
        elif _args.task_type == 'clone_detection':
            # count total_jsonl_path
            _line_count = count_file_line(_args.total_jsonl_path)
        elif _args.task_type in ['code_classification', 'vulnerability_detection']:
            # count by iterate meta size
            _dataset_meta = UtilsMeta.get_dataset_meta(_args.mode, _args.task_type)
            _size = _dataset_meta['size']
            _step = _dataset_meta['step']
            _step_count = _size // _step
            if _size % _step != 0:
                _step_count += 1
            # example file tg/datasets/shared/code_classification/test_0_500.jsonl
            _line_count = 0
            for _idx in range(_step_count):
                _tmp_path = f'tg/datasets/shared/{_args.task_type}/test_{_idx * _step}_{(_idx + 1) * _step}.jsonl'
                _line_count += count_file_line(_tmp_path)
        # split by 1:1 random
        _train_size = _line_count // 2
        _test_size = _line_count - _train_size
        _train_index = random.sample(range(_line_count), _train_size)
        # sort asc
        _train_index.sort()
        _test_index = list(set(range(_line_count)) - set(_train_index))
        _test_index.sort()
        # save to json
        with open(_split_train_path, 'w+') as _file:
            json.dump(_train_index, _file)
        with open(_split_test_path, 'w+') as _file:
            json.dump(_test_index, _file)
        _logger.info(f'line count: {_line_count}')
    @staticmethod
    def load_code_list_from_file(_args, _file_path, _logger, _path_prefix=''):
        _code_info_list = []
        if _args.task_type == 'authorship_attribution':
            _line_idx = 0
            with open(f'{_path_prefix}{_file_path}', 'r') as _file:
                for _line in _file:
                    _code_str = _line.split(" <CODESPLIT> ")[0]
                    _code_str = _code_str.replace("\\n", "\n").replace('\"','"')
                    _label_val = _line.split(" <CODESPLIT> ")[1]
                    _code_info_list.append({
                        'code': _code_str,
                        'label': int(_label_val),
                        'idx': _line_idx
                    })
                    _line_idx += 1
        elif _args.task_type == 'clone_detection':
            _idx2code = {}
            with open(_args.total_jsonl_path, 'r') as _file:
                _line_index = 0
                _file_loader = tqdm(_file)
                for _line in _file_loader:
                    _tmp_item = json.loads(_line.strip())
                    _idx2code[_tmp_item['idx']] = _tmp_item['func']
                    _file_loader.set_description(f'load code: {_line_index}')
                    _line_index += 1
            with open(_file_path, 'r') as _file:
                for _line in _file:
                    _line_str = _line.strip()
                    try:
                        _idx1, _idx2, _label_val = _line_str.split('\t')
                    except:
                        _logger.error(f'error line: {_line_str}')
                    if _idx1 not in _idx2code or _idx2 not in _idx2code:
                        _logger.error(f'skip no-key line: {_line_str}')
                        continue
                    _label_val = 0 if _label_val == '0' else 1
                    _code1 = _idx2code[_idx1]
                    _code2 = _idx2code[_idx2]
                    _code_info_list.append({
                        'code': _code1,
                        'code2': _code2,
                        'label': _label_val,
                        'idx': _idx1,
                        'idx2': _idx2
                    })
        elif _args.task_type in ['code_classification', 'vulnerability_detection']:
            with open(_file_path, 'r') as _file:
                for _line in _file:
                    _tmp_item = json.loads(_line.strip())
                    _code_str = _tmp_item['func']
                    _label_val = int(_tmp_item['target'])
                    _idx = _tmp_item['idx']
                    _code_info_list.append({
                        'code': _code_str,
                        'label': _label_val,
                        'idx': _idx
                    })
        # log info len
        _logger.info(f'load code list from file: {_file_path}, len: {len(_code_info_list)}')
        return _code_info_list
    @staticmethod
    def load_retrain_code_list(_args, _logger):
        # load code part 1: original data, code part 2: retraining train data, code part 3: retraining train data adversarial attack code
        # part 1: author: train.txt, clone: train_sampled.txt, classification/vulnerability: train.jsonl
        if _args.task_type in ['authorship_attribution', 'clone_detection']:
            _train_file_path = _args.train_code_txt_path
        elif _args.task_type in ['code_classification', 'vulnerability_detection']:
            _train_file_path = _args.train_code_jsonl_path
        _code_info_list_1 = InputDataset.load_code_list_from_file(_args, _train_file_path, _logger)
        # part 2/3, need to filter by retraining index train/test
        _dataset_meta = UtilsMeta.get_dataset_meta(_args.mode, _args.task_type)
        _size = _dataset_meta['size']
        _step = _dataset_meta['step']
        _step_count = _size // _step
        if _size % _step != 0:
            _step_count += 1
        _retrain_train_index = json.loads(open(_args.retraining_train_index_path, 'r').read())
        _retrain_test_index = json.loads(open(_args.retraining_test_index_path, 'r').read())
        _code_info_list_2 = []
        _code_info_list_3 = []
        # part 4 retrain test index for evaluate
        _code_info_list_4 = []
        # part 5 retrain test adv code
        _code_info_list_5 = []
        for _step_idx in range(_step_count):
            _index_start = _step_idx * _step
            _index_end = (_step_idx + 1) * _step
            _tmp_args = copy.deepcopy(_args)
            _tmp_args.index = [_index_start, _index_end]
            # reuse prepare_attack_file, validate_attack_args
            UtilsMeta.prepare_attack_file(_tmp_args)
            _tmp_attack_res_info = UtilsMeta.validate_attack_args(_tmp_args)
            if _tmp_args.task_type in ['authorship_attribution', 'clone_detection']:
                _step_attack_file_path = _tmp_args.code_txt_path
            elif _tmp_args.task_type in ['code_classification', 'vulnerability_detection']:
                _step_attack_file_path = _tmp_args.code_jsonl_path
            assert os.path.exists(_tmp_attack_res_info['path'])
            assert os.path.exists(_step_attack_file_path)
            _logger.info(f'load from attack res info: {_tmp_attack_res_info["path"]}')
            _tmp_code_info_list = InputDataset.load_code_list_from_file(_tmp_args, _step_attack_file_path, _logger)
            # _tmp_retrain_index_train from _retrain_index_train and >= _index_start and < _index_end
            _tmp_retrain_index_train = [_x - _index_start for _x in _retrain_train_index if _x >= _index_start and _x < _index_end]
            _tmp_retrain_index_test = [_x - _index_start for _x in _retrain_test_index if _x >= _index_start and _x < _index_end]
            for _tmp_train_index in _tmp_retrain_index_train:
                _code_info_list_2.append(copy.deepcopy(_tmp_code_info_list[_tmp_train_index]))
            for _tmp_test_index in _tmp_retrain_index_test:
                _code_info_list_4.append(copy.deepcopy(_tmp_code_info_list[_tmp_test_index]))
            # read attach res info path, keep only status=1 and index in train
            _tmp_index = 0
            with open(_tmp_attack_res_info['path'], 'r') as _file:
                for _line in _file:
                    _tmp_item = json.loads(_line.strip())
                    if _tmp_item['status'] != 1:
                        _tmp_index += 1
                        continue
                    # replace code value in _tmp_code_info_list target index
                    _tmp_code_info = copy.deepcopy(_tmp_code_info_list[_tmp_index])
                    _tmp_code_info['code'] = _tmp_item['new_code_str']
                    if _tmp_index in _tmp_retrain_index_train:
                        _code_info_list_3.append(_tmp_code_info)
                        _tmp_index += 1
                        continue
                    if _tmp_index in _tmp_retrain_index_test:
                        _code_info_list_5.append(_tmp_code_info)
                        _tmp_index += 1
                        continue
                    _tmp_index += 1
        # log 3 len
        _logger.info(f'load retrain code list 1: {len(_code_info_list_1)}, 2: {len(_code_info_list_2)}, 3: {len(_code_info_list_3)}')
        _logger.info(f'load retrain test code list: {len(_code_info_list_4)}, adv code list: {len(_code_info_list_5)}')
        # 2+3 as augmented info list
        return {
            'original_code_info_list': _code_info_list_1,
            'augmented_code_info_list': _code_info_list_2 + _code_info_list_3,
            'test_code_info_list': _code_info_list_4,
            'test_adv_code_info_list': _code_info_list_5
        }
    @staticmethod
    def load_transfer_code_list(_atk_args_list, _atk_res_list, _logger):
        _unatk_code_info_list = []
        for _atk_args in _atk_args_list:
            _tmp_args = copy.deepcopy(_atk_args)
            UtilsMeta.prepare_attack_file(_tmp_args)
            if _tmp_args.task_type in ['authorship_attribution', 'clone_detection']:
                _attack_source_file_path = _tmp_args.code_txt_path
            elif _tmp_args.task_type in ['code_classification', 'vulnerability_detection']:
                _attack_source_file_path = _tmp_args.code_jsonl_path
            _logger.info(f'load from file: {_attack_source_file_path}')
            _tmp_code_info_list = InputDataset.load_code_list_from_file(_tmp_args, _attack_source_file_path, _logger)
            _unatk_code_info_list.extend(_tmp_code_info_list)
        _file_path_list = [x['path'] for x in _atk_res_list]
        _adv_code_info_list = []
        # iterate through _file_path_list
        _global_idx = 0
        for _file_path in _file_path_list:
            _logger.info(f'load from file: {_file_path}')
            with open(_file_path, 'r') as _adv_file:
                for _adv_jsonl_line in _adv_file:
                    _adv_info = json.loads(_adv_jsonl_line.strip())
                    # if empty, skip
                    if len(_adv_info) == 0:
                        _global_idx += 1
                        continue
                    if _adv_info['status'] != _status_map['success']:
                        _global_idx += 1
                        continue
                    _adv_code_str = _adv_info['new_code_str']
                    # update target code info
                    _tmp_code_info = copy.deepcopy(_unatk_code_info_list[_global_idx])
                    _tmp_code_info['code'] = _adv_code_str
                    _adv_code_info_list.append(_tmp_code_info)
                    _global_idx += 1
        _logger.info(f'adv code len: {len(_adv_code_info_list)}')
        return _adv_code_info_list
    @staticmethod
    def get_dataset(_args, _feature_list):
        if _args.model_name in ['CodeBERT', 'CodeT5']:
            return InputCodeDataset(_feature_list, _args)
        elif _args.model_name == 'GraphCodeBERT':
            return InputCodeGraphDataset(_feature_list, _args)
    @staticmethod
    def get_perf(_label_truth, _label_pred):
        _res_info = {}
        for _metric_type in ['macro', 'micro', 'weighted']:
            _acc = sklearn.metrics.accuracy_score(_label_truth, _label_pred)
            _recall = sklearn.metrics.recall_score(_label_truth, _label_pred, average=_metric_type)
            _precision = sklearn.metrics.precision_score(_label_truth, _label_pred, average=_metric_type)
            _f1 = sklearn.metrics.f1_score(_label_truth, _label_pred, average=_metric_type)
            _res_info[f'{_metric_type}_accuracy'] = _acc
            _res_info[f'{_metric_type}_recall'] = _recall
            _res_info[f'{_metric_type}_precision'] = _precision
            _res_info[f'{_metric_type}_f1'] = _f1
        return _res_info
class InputCodeDataset(Dataset):
    def __init__(self, _feature_list, _args):
        self._feature_list = _feature_list
        self._args = _args
    def __len__(self):
        return len(self._feature_list)
    def __getitem__(self, _idx):
        # onvert into tensor
        _feature = self._feature_list[_idx]
        return (torch.tensor(_feature._input_id_list), 
                # torch.tensor([_feature._label]), 
                torch.tensor(_feature._label), 
                _feature._idx_info)
class InputCodeGraphDataset(Dataset):
    def __init__(self, _feature_list, _args):
        self._feature_list = _feature_list
        self._args = _args
    def __len__(self):
        return len(self._feature_list)
    @staticmethod
    def get_graph_attn_mask(_feature, _args):
        # calculate graph-guided masked function
        _attn_mask = np.zeros(
            (
                _args.model_code_length + _args.model_data_flow_length,
                _args.model_code_length + _args.model_data_flow_length
            ),
            dtype=np.bool
        )
        #calculate begin index of node and max length of input
        _node_index = sum([
            i > 1 for i in _feature._position_idx_info
        ])
        _max_length = sum([
            i != 1 for i in _feature._position_idx_info
        ])
        # sequence can attend to sequence
        _attn_mask[:_node_index, :_node_index] = True
        # special tokens attend to all tokens
        for _idx, _id_v in enumerate(_feature._input_id_list):
            if _id_v in [0, 2]:
                _attn_mask[_idx, :_max_length] = True
        # nodes attend to code tokens that are identified from
        for _idx, (_a, _b) in enumerate(_feature._dfg2code_info):
            if _a < _node_index and _b < _node_index:
                _attn_mask[_idx + _node_index, _a:_b] = True
                _attn_mask[_a:_b, _idx + _node_index] = True
        # nodes attend to adjacent nodes 
        for _idx, _nodes in enumerate(_feature._dfg2dfg_info):
            for _a in _nodes:
                if _a + _node_index < len(_feature._position_idx_info):
                    _attn_mask[_idx + _node_index, _a + _node_index] = True
        return _attn_mask
    def __getitem__(self, _item_idx):
        if self._args.task_type == 'clone_detection':
            # pair in input id, idx_info, position idx, dfg info
            _feature_info = self._feature_list[_item_idx]
            _feature_0 = InputCodeGraphFeature(
                _input_id_list = _feature_info._input_id_list[0],
                _idx_info = None,
                _label = _feature_info._label,
                _position_idx_info = _feature_info._position_idx_info[0],
                _dfg2code_info = _feature_info._dfg2code_info[0],
                _dfg2dfg_info = _feature_info._dfg2dfg_info[0]
            )
            _attn_mask_0 = InputCodeGraphDataset.get_graph_attn_mask(_feature_0, self._args)
            _feature_1 = InputCodeGraphFeature(
                _input_id_list = _feature_info._input_id_list[1],
                _idx_info = None,
                _label = _feature_info._label,
                _position_idx_info = _feature_info._position_idx_info[1],
                _dfg2code_info = _feature_info._dfg2code_info[1],
                _dfg2dfg_info = _feature_info._dfg2dfg_info[1]
            )
            _attn_mask_1 = InputCodeGraphDataset.get_graph_attn_mask(_feature_1, self._args)
            return (
                torch.tensor(_feature_0._input_id_list),
                torch.tensor(_feature_0._position_idx_info),
                torch.tensor(_attn_mask_0),
                torch.tensor(_feature_1._input_id_list),
                torch.tensor(_feature_1._position_idx_info),
                torch.tensor(_attn_mask_1),
                torch.tensor(_feature_info._label)
            )
        else:
            _feature = self._feature_list[_item_idx]
            _attn_mask = InputCodeGraphDataset.get_graph_attn_mask(_feature, self._args)
            return (
                torch.tensor(_feature._input_id_list),
                torch.tensor(_feature._position_idx_info),
                torch.tensor(_attn_mask),
                torch.tensor(_feature._label),
            )
class UtilsCodeBERTT5Feature():
    def __init__(self):
        pass
    @staticmethod
    def get_classification_feature(_args, _code_str, _code_info, _tokenizer, _logger):
        # throw error if task_type not in authorship_attribution, code_classification, vulnerability_detection
        if _args.task_type not in ['authorship_attribution', 'code_classification', 'vulnerability_detection']:
            raise ValueError(f'error task_type: {_args.task_type}')
        _token_list = _tokenizer.tokenize(_code_str)[:_args.model_code_length - 2]
        _token_list = [_tokenizer.cls_token] + _token_list + [_tokenizer.sep_token]
        _token_id_list = _tokenizer.convert_tokens_to_ids(_token_list)
        _padding_len = _args.model_code_length - len(_token_id_list)
        _token_id_list += [_tokenizer.pad_token_id] * _padding_len
        return InputCodeFeature(
            _input_id_list = _token_id_list,
            _label = _code_info['label'],
            _idx_info = _code_info['idx']
        )
    @staticmethod
    def get_clone_feature(_args, _code_str, _code_info, _tokenizer, _logger):
        # throw error if task_type not in clone_detection
        if _args.task_type != 'clone_detection':
            raise ValueError(f'error task_type: {_args.task_type}')
        _code_2 = _code_info['code2']
        _code_1_tokenized_info = UtilsTokenizer.tokenize_and_pad(_args, _code_str, _tokenizer)
        _code_2_tokenized_info = UtilsTokenizer.tokenize_and_pad(_args, _code_2, _tokenizer)
        # concate id
        # _token_list = _code_1_tokenized_info['token_list'] + _code_2_tokenized_info['token_list']
        _id_list = _code_1_tokenized_info['id_list'] + _code_2_tokenized_info['id_list']
        return InputCodeFeature(
            _input_id_list = _id_list,
            _label = _code_info['label'],
            _idx_info = [_code_info['idx'], _code_info['idx2']]
        )
class UtilsGraphCodeBERTFeature():
    @staticmethod
    def get_graph_info(_args, _code_str, _tokenizer, _logger):
        _token_list, _, _dfg = UtilsTokenizer.extract_dataflow(_args, _code_str, _logger)
        _token_list = [_tokenizer.tokenize('@ ' + _x)[1:] if _idx != 0 else _tokenizer.tokenize(_x) for _idx, _x in enumerate(_token_list)]
        _ori2cur_pos = {}
        _ori2cur_pos[-1] = (0, 0)
        for _idx in range(len(_token_list)):
            _ori2cur_pos[_idx] = (_ori2cur_pos[_idx - 1][1], _ori2cur_pos[_idx - 1][1] + len(_token_list[_idx]))
        _token_list = [_y for _x in _token_list for _y in _x]
        _token_list = _token_list[:_args.model_code_length + _args.model_data_flow_length - 2 - min(len(_dfg), _args.model_data_flow_length)]
        _source_token_list = [_tokenizer.cls_token] + _token_list + [_tokenizer.sep_token]
        _source_id_list = _tokenizer.convert_tokens_to_ids(_source_token_list)
        _position_idx_info = [_idx + _tokenizer.pad_token_id for _idx in range(len(_source_token_list))]
        _dfg = _dfg[:_args.model_code_length + _args.model_data_flow_length - len(_source_token_list)]
        _source_token_list += [_x[0] for _x in _dfg]
        _position_idx_info += [0 for _ in _dfg]
        _source_id_list += [_tokenizer.unk_token_id for _ in _dfg]
        _padding_len = _args.model_code_length + _args.model_data_flow_length - len(_source_id_list)
        _position_idx_info += [_tokenizer.pad_token_id] * _padding_len
        _source_id_list += [_tokenizer.pad_token_id] * _padding_len
        _reverse_idx = {}
        for _idx, _x in enumerate(_dfg):
            _reverse_idx[_x[1]] = _idx
        for _idx, _x in enumerate(_dfg):
            _dfg[_idx] = _x[:-1] + ([
                _reverse_idx[_i] for _i in _x[-1] if _i in _reverse_idx 
            ],)
        _dfg2dfg = [_x[-1] for _x in _dfg]
        _dfg2code = [_ori2cur_pos[_x[1]] for _x in _dfg]
        _length = len([_tokenizer.cls_token])
        _dfg2code = [(_x[0] + _length, _x[1] + _length) for _x in _dfg2code]
        return {
            'input_id_list': _source_id_list,
            'position_idx_info': _position_idx_info,
            'dfg2code_info': _dfg2code,
            'dfg2dfg_info': _dfg2dfg
        }
    @staticmethod
    def get_classification_feature(_args, _code_str, _code_info, _tokenizer, _logger):
        _code_graph_info = UtilsGraphCodeBERTFeature.get_graph_info(_args, _code_str, _tokenizer, _logger)
        return InputCodeGraphFeature(
            _input_id_list = _code_graph_info['input_id_list'],
            _label = _code_info['label'],
            _idx_info = _code_info['idx'],
            _position_idx_info = _code_graph_info['position_idx_info'],
            _dfg2code_info = _code_graph_info['dfg2code_info'],
            _dfg2dfg_info = _code_graph_info['dfg2dfg_info']
        )
    @staticmethod
    def get_clone_feature(_args, _code_str, _code_info, _tokenizer, _logger):
        _code_0_graph_info = UtilsGraphCodeBERTFeature.get_graph_info(_args, _code_str, _tokenizer, _logger)
        _code_1_graph_info = UtilsGraphCodeBERTFeature.get_graph_info(_args, _code_info['code2'], _tokenizer, _logger)
        return InputCodeGraphFeature(
            _input_id_list =[_code_0_graph_info['input_id_list'],
                _code_1_graph_info['input_id_list']],
            _label = _code_info['label'],
            _idx_info = [_code_info['idx'], _code_info['idx2']],
            _position_idx_info = [_code_0_graph_info['position_idx_info'],
                _code_1_graph_info['position_idx_info']],
            _dfg2code_info = [_code_0_graph_info['dfg2code_info'],
                _code_1_graph_info['dfg2code_info']],
            _dfg2dfg_info = [_code_0_graph_info['dfg2dfg_info'],
                _code_1_graph_info['dfg2dfg_info']]
        )
    
class UtilsFeature():
    def __init__(self):
        pass
    @staticmethod
    def get_code_dataset(_args, _code_info_list: list, _tokenizer, _logger):
        # iterate code_str_list and code_info_list, prepare feature list, create related dataset instance
        _feature_matrix = []
        # use tqdm to show progress
        _code_info_loader = tqdm(_code_info_list)
        for _code_idx, _code_info in enumerate(_code_info_loader):
            _code_str = _code_info['code']
            _tmp_feature = UtilsFeature.get_code_feature(_args, _code_str, _code_info, _tokenizer, _logger)
            _feature_matrix.append(_tmp_feature)
            _code_info_loader.set_description(f'load code: {_code_idx} of {len(_code_info_list)}')
        # for code_info in _code_info_list:
        #     _code_str = code_info['code']
        #     _tmp_feature = UtilsFeature.get_code_feature(_args, _code_str, code_info, _tokenizer, _logger)
        #     _feature_matrix.append(_tmp_feature)
        # prepare dataset
        _dataset = InputDataset.get_dataset(_args, _feature_matrix)
        return _dataset
    @staticmethod
    def get_code_feature(_args, _cur_code_str: str, _code_info: dict, _tokenizer, _logger):
        '''
        model_name: CodeBERT, CodeT5, GraphCodeBERT
        task_type: authorship_attribution, clone_detection, code_classification, vulnerability_detection
        '''
        # if _args.model_name == 'CodeBERT':
        if _args.model_name in ['CodeBERT', 'CodeT5']:
            if _args.task_type in ['authorship_attribution', 'code_classification', 'vulnerability_detection']:
                return UtilsCodeBERTT5Feature.get_classification_feature(_args, _cur_code_str, _code_info, _tokenizer, _logger)
            elif _args.task_type == 'clone_detection':
                return UtilsCodeBERTT5Feature.get_clone_feature(_args, _cur_code_str, _code_info, _tokenizer, _logger)
        elif _args.model_name == 'GraphCodeBERT':
            if _args.task_type in ['authorship_attribution', 'code_classification', 'vulnerability_detection']:
                return UtilsGraphCodeBERTFeature.get_classification_feature(_args, _cur_code_str, _code_info, _tokenizer, _logger)
            elif _args.task_type == 'clone_detection':
                return UtilsGraphCodeBERTFeature.get_clone_feature(_args, _cur_code_str, _code_info, _tokenizer, _logger)
class UtilsTokenizer():
    _predefined_position_replacement = ["a","b","c","de", "fg","hi","jkl","mno","pqr","stuv","wxyz","abcde","fghig","klmno","pqrst","uvwxyz"]
    def __init__(self):
        pass
    @staticmethod
    def get_code_identifier_token(_args, _code_str: str, _logger):
        _lang_val = _task_lang_map[_args.task_type]
        _filtered_code_str = _code_str
        try:
            # try remove_comments_and_docstrings
            _filtered_code_str = remove_comments_and_docstrings(_code_str, _lang_val)
        except Exception as e:
            # _logger.error(f'error remove_comments_and_docstrings: {e}, {_code_str}')
            pass
        _identifier_list, _token_list = get_identifiers(_filtered_code_str, _lang_val)
        return _identifier_list, _token_list
    @staticmethod
    def get_tokenized_info(_args, _code_str: str, _tokenizer, _logger):
        _identifier_list, _token_list = UtilsTokenizer.get_code_identifier_token(_args, _code_str, _logger)
        _new_code_str = ' '.join(_token_list)
        _word_list, _sub_word_list, _sub_word_index = UtilsTokenizer.tokenize_seq_token_list(_new_code_str, _tokenizer)
        _valid_identifier_list = []
        _lang_v = _task_lang_map[_args.task_type]
        for _tmp_list in _identifier_list:
            _tmp_name = _tmp_list[0].strip()
            if ' ' in _tmp_name:
                continue
            # check by is_valid_variable_name
            if not is_valid_variable_name(_tmp_name, _lang_v):
                continue
            _valid_identifier_list.append(_tmp_name)
        return {
            'word_list': _word_list,
            'valid_identifier_list': _valid_identifier_list,
            'sub_word_list': _sub_word_list,
            'sub_word_index': _sub_word_index,
            'token_code_str': _new_code_str
        }
    @staticmethod
    def tokenize_seq_token_list(_seq: str, _tokenizer):
        _seq = _seq.replace('\n', '')
        words = _seq.split(' ')
        sub_words = []
        subword_index_info = []
        index = 0
        for word in words:
            # 并非直接tokenize这句话，而是tokenize了每个splited words.
            sub = _tokenizer.tokenize(word)
            sub_words += sub
            subword_index_info.append([index, index + len(sub)])
            # 将subwords对齐
            index += len(sub)
        return words, sub_words, subword_index_info
    @staticmethod
    def get_identifier_position_name2index(_token_list: list, _name_list: list) -> dict:
        '''
        variable name to index dict
        '''
        _name2index_dict = {}
        for _name in _name_list:
            for _index, _token in enumerate(_token_list):
                if _name == _token:
                    if _name not in _name2index_dict:
                        _name2index_dict[_name] = []
                    _name2index_dict[_name].append(_index)
        return _name2index_dict
    @staticmethod
    def mask_identifier_list_by_unk(_token_list, _name2index_dict):
        '''
        mask each identifier in dict, get masked matrix
        '''
        _masked_token_matrix = []
        _masked_name_list = list(_name2index_dict.keys())
        for _name_key in _masked_name_list:
            _tmp_token_list = copy.deepcopy(_token_list)
            for _pos_idnex in _name2index_dict[_name_key]:
                _tmp_token_list[_pos_idnex] = '<unk>'
            _masked_token_matrix.append(_tmp_token_list)
        return _masked_token_matrix, _masked_name_list
    @staticmethod
    def mask_identifier_list_by_predefined_replacement(_token_list, _name2index_dict):
        '''replace each variable with _predefined_position_replacement'''
        _replaced_token_matrix = []
        _name_list = list(_name2index_dict.keys())
        for _name_key in _name_list:
            for _predefined_name in UtilsTokenizer._predefined_position_replacement:
                _tmp_token_list = copy.deepcopy(_token_list)
                for _pos_index in _name2index_dict[_name_key]:
                    _tmp_token_list[_pos_index] = _predefined_name
                _replaced_token_matrix.append(_tmp_token_list)
        return _replaced_token_matrix, _name_list
    @staticmethod
    def tokenize_and_pad(_args, _code_str: str, _tokenizer):
        _token_list = _tokenizer.tokenize(_code_str)[:_args.model_code_length - 2]
        _token_list = [_tokenizer.cls_token] + _token_list + [_tokenizer.sep_token]
        _token_id_list = _tokenizer.convert_tokens_to_ids(_token_list)
        _padding_length = _args.model_code_length - len(_token_id_list)
        _token_id_list += [_tokenizer.pad_token_id] * _padding_length
        return {
            'token_list': _token_list,
            'id_list': _token_id_list
        }
    @staticmethod
    def extract_dataflow(_args, _code_str: str, _logger):
        _lang_val = _task_lang_map[_args.task_type]
        _lang_parser = _lang2parser[_lang_val]
        _cur_code_str = _code_str.replace('\\n', '\n')
        try:
            _cur_code_str = remove_comments_and_docstrings(_cur_code_str, _lang_val)
        except Exception as e:
            # _logger.error(f'error remove_comments_and_docstrings: {e}, {_cur_code_str}')
            pass
        _index_idx2index = {}
        try:
            _tree = _lang_parser[0].parse(bytes(_cur_code_str, 'utf8'))
            _root_node = _tree.root_node
            _token2index = tree_to_token_index(_root_node)
            _cur_code_str = _cur_code_str.split('\n')
            _token_list = [index_to_code_token(_t, _cur_code_str) for _t in _token2index]
            _index2code = {}
            for _idx, (_index, _tmp_code) in enumerate(zip(_token2index, _token_list)):
                _index2code[_index] = (_idx, _tmp_code)
            for _idx, (_index, _tmp_code) in enumerate(zip(_token2index, _token_list)):
                _index_idx2index[_idx] = _index
            _DFG, _ = _lang_parser[1](_root_node, _index2code, {})
            _DFG = sorted(_DFG, key=lambda x: x[1])
            pass
        except Exception as e:
            _logger.error(f'error parse code: {e}, {_cur_code_str}')
            _DFG = []
        return _token_list, _index_idx2index, _DFG
    @staticmethod
    def get_code_name2idx(_args, _word_list, _name_list):
        _name2idx_dict = {}
        for _name in _name_list:
            for _idx, _word in enumerate(_word_list):
                if _name == _word:
                    if _name not in _name2idx_dict:
                        _name2idx_dict[_name] = []
                    _name2idx_dict[_name].append(_idx)
        return _name2idx_dict
    @staticmethod
    def get_bpe_sub_list(_args, _substitutes, _tokenizer, _mlm_model):
        # limit max candidates
        _subs_list = _substitutes[0:12, 0:4]
        # find all possible candidates
        _subs_total_list = []
        for _idx in range(_subs_list.size(0)):
            if len(_subs_total_list) == 0:
                _lev_i = _subs_list[_idx]
                _subs_total_list = [[int(_c)] for _c in _lev_i]
            else:
                _lev_i = []
                for _all_sub in _subs_total_list[:24]:
                    for _j in _subs_list[_idx]:
                        _lev_i.append(_all_sub + [int(_j)])
                _subs_total_list = _lev_i
        # all substitutes list of list of token-id (all candidates)
        _c_loss = torch.nn.CrossEntropyLoss(reduction='none')
        _word_list = []
        _subs_total_list = torch.tensor(_subs_total_list)
        _subs_total_list = _subs_total_list[:24].to(_args.device)
        _N, _L = _subs_total_list.size()
        _word_pred_list = _mlm_model(_subs_total_list)[0]
        _ppl = _c_loss(_word_pred_list.view(_N * _L, -1), _subs_total_list.view(-1))
        _ppl = torch.exp(torch.mean(_ppl.view(_N, _L), dim=-1))
        _, _word_list = torch.sort(_ppl)
        _word_list = [_subs_total_list[i] for i in _word_list]
        _word_res_list = []
        for _word in _word_list:
            _tokens = [_tokenizer._convert_id_to_token(int(_i)) for _i in _word]
            _text = _tokenizer.convert_tokens_to_string(_tokens)
            _word_res_list.append(_text)
        return _word_res_list
    @staticmethod
    def recover_word_list_from_similar_subs(_args, _similar_subs, _similar_scores, _tokenizer, _mlm_model, _use_bpe, _threshold=3.0):
        _word_list = []
        _sub_len, _k = _similar_subs.size()
        if _sub_len == 0:
            return _word_list
        elif _sub_len == 1:
            for (_i, _j) in zip(_similar_subs[0], _similar_scores[0]):
                if _threshold != 0 and _j < _threshold:
                    continue
                _word_list.append(
                    _tokenizer.decode([int(_i)])
                )
        elif _use_bpe:
            _word_list = UtilsTokenizer.get_bpe_sub_list(_args, _similar_subs, _tokenizer, _mlm_model)
        return _word_list
    @staticmethod
    def recover_word_list_from_similar_subs_no_threshold(_args, _similar_subs, _tokenizer, _mlm_model, _use_bpe):
        _word_list = []
        _sub_len, _k = _similar_subs.size()
        if _sub_len == 0:
            return _word_list
        elif _sub_len == 1:
            # decode similar subs directly
            for _word_idx in _similar_subs[0]:
                _word_list.append(
                    _tokenizer.decode([int(_word_idx)])
                )
        elif _use_bpe:
            _word_list = UtilsTokenizer.get_bpe_sub_list(_args, _similar_subs, _tokenizer, _mlm_model)
        return _word_list
class UtilsPositionScore():
    def __init__(self):
        pass
    @staticmethod
    def get_pos_file_info(_args, _logger):
        '''
        path, existing len
        '''
        if _args.position_type in ['predefined_mask', 'predefined_replacement']:
            
            # path example: tg/datasets/shared/{task_type}/position_{_args.model_name}_{position_type}_{index}_{index}.jsonl
            _res_file = f'tg/datasets/shared/{_args.task_type}/position_{_args.model_name}_{_args.position_type}_{_args.index[0]}_{_args.index[1]}.jsonl'
        elif _args.position_type == 'random':
            _res_file = f'tg/datasets/shared/{_args.task_type}/position_{_args.position_type}_{_args.index[0]}_{_args.index[1]}.jsonl'
        _existing_len = count_file_line(_res_file)
        _logger.info(f'pos res file: {_res_file}, existing len: {_existing_len}')
        return {
            'path': _res_file,
            'existing_len': _existing_len
        }
    @staticmethod
    def calc_identifier_importance_score_by_mask(_args, _model_info: dict, _code_info: dict, _tokenized_info: dict, _logger=None, _cur_name2index_dict: dict = None):
        _tokenizer = _model_info['tokenizer']
        if _cur_name2index_dict != None:
            # reuse name2index dict, maybe for dynamic score calc
            _name2index_dict = _cur_name2index_dict
        else:
            _name2index_dict = UtilsTokenizer.get_identifier_position_name2index(
                _token_list = _tokenized_info['word_list'],
                _name_list = _tokenized_info['valid_identifier_list']
            )
        _masked_token_matrix, _masked_name_list = UtilsTokenizer.mask_identifier_list_by_unk(
            _token_list = _tokenized_info['word_list'],
            _name2index_dict = _name2index_dict
        )
        # return if _masked_token_matrix empty
        if len(_masked_token_matrix) == 0:
            return []
        _masked_feature_matrix = []
        for _token_list in _masked_token_matrix:
            _tmp_code_str = ' '.join(_token_list)
            # vectorize
            _tmp_feature = UtilsFeature.get_code_feature(_args, _tmp_code_str, _code_info, _tokenizer, _logger)
            _masked_feature_matrix.append(_tmp_feature)
        # prepare dataset
        # prepare base logit score
        _base_code_str = ' '.join(_tokenized_info['word_list'])
        _base_code_feature = UtilsFeature.get_code_feature(_args, _base_code_str, _code_info, _tokenizer, _logger)
        if _args.model_name in ['CodeBERT', 'CodeT5']:
            _masked_dataset = InputCodeDataset(_masked_feature_matrix, _args)
            _base_dataset = InputCodeDataset([_base_code_feature], _args)
        elif _args.model_name == 'GraphCodeBERT':
            _masked_dataset = InputCodeGraphDataset(_masked_feature_matrix, _args)
            _base_dataset = InputCodeGraphDataset([_base_code_feature], _args)
        # get dataset results
        # log dataset len
        # _logger.info(f'masked dataset len: {len(_masked_dataset)}, base dataset len: {len(_base_dataset)}')
        _logit_list, _label_pred_list = _model_info['model'].get_dataset_result(
            _dataset=_masked_dataset,
            _batch_size=_args.model_eval_batch_size,
        )
        _base_logit_list, _base_label_pred_list = _model_info['model'].get_dataset_result(
            _dataset=_base_dataset,
            _batch_size=_args.model_eval_batch_size
        )
        # assert _base_logit_list len, get value on label index
        assert len(_base_logit_list) == 1
        _base_logit_v = _base_logit_list[0][_code_info['label']]
        # calc prob delta value on target label index and return
        # rank by prob delta, prepare save format
        _masked_name_score_delta_list = []
        for _logit_idx, _logit_v in enumerate(_logit_list):
            _tmp_logit_v = _logit_v[_code_info['label']]
            _tmp_logit_delta = _base_logit_v - _tmp_logit_v
            _masked_name_score_delta_list.append({
                'name': _masked_name_list[_logit_idx],
                'score': _tmp_logit_delta
            })
        # rank _masked_name_score_delta_list by score, desc order
        _masked_name_score_delta_list = sorted(_masked_name_score_delta_list, key=lambda x: x['score'], reverse=True)
        return _masked_name_score_delta_list
    @staticmethod
    def calc_identifier_importance_score_by_predefined_replacement(_args, _model_info: dict, _code_info: dict, _tokenized_info: dict, _logger=None, _cur_name2index_dict: dict = None):
        '''
        try each identifier replaced by predifined set, calc average prob delta, rank by delta from high to low
        '''
        _tokenizer = _model_info['tokenizer']
        if _cur_name2index_dict != None:
            # reuse name2index dict, maybe for dynamic score calc
            _name2index_dict = _cur_name2index_dict
        else:
            _name2index_dict = UtilsTokenizer.get_identifier_position_name2index(
                _token_list = _tokenized_info['word_list'],
                _name_list = _tokenized_info['valid_identifier_list']
            )
        _replaced_token_matrix, _replaced_name_list = UtilsTokenizer.mask_identifier_list_by_predefined_replacement(
            _token_list = _tokenized_info['word_list'],
            _name2index_dict = _name2index_dict
        )
        # matrix len = len(_replaced_name_list) * len(UtilsTokenizer._predefined_position_replacement)
        if len(_replaced_token_matrix) == 0:
            return []
        _replaced_feature_matrix = []
        for _token_list in _replaced_token_matrix:
            _tmp_code_str = ' '.join(_token_list)
            # vectorize
            _tmp_feature = UtilsFeature.get_code_feature(_args, _tmp_code_str, _code_info, _tokenizer, _logger)
            _replaced_feature_matrix.append(_tmp_feature)
        # prepare dataset
        _base_code_str = ' '.join(_tokenized_info['word_list'])
        _base_code_feature = UtilsFeature.get_code_feature(_args, _base_code_str, _code_info, _tokenizer, _logger)
        if _args.model_name in ['CodeBERT', 'CodeT5']:
            _replaced_dataset = InputCodeDataset(_replaced_feature_matrix, _args)
            _base_dataset = InputCodeDataset([_base_code_feature], _args)
        elif _args.model_name == 'GraphCodeBERT':
            _replaced_dataset = InputCodeGraphDataset(_replaced_feature_matrix, _args)
            _base_dataset = InputCodeGraphDataset([_base_code_feature], _args)
        # _logger.info(f'replaced dataset len: {len(_replaced_dataset)}, base dataset len: {len(_base_dataset)}')
        _logit_list, _label_pred_list = _model_info['model'].get_dataset_result(
            _dataset = _replaced_dataset,
            _batch_size = _args.model_eval_batch_size
        )
        _base_logit_list, _base_label_pred_list = _model_info['model'].get_dataset_result(
            _dataset = _base_dataset,
            _batch_size = _args.model_eval_batch_size
        )
        assert len(_base_logit_list) == 1
        assert len(_logit_list) == len(_replaced_token_matrix)
        assert len(_logit_list) == len(_replaced_name_list) * len(UtilsTokenizer._predefined_position_replacement)
        _base_logit_v = _base_logit_list[0][_code_info['label']]
        # chunk logit list in size len(UtilsTokenizer._predefined_position_replacement)
        _logit_chunk_list = []
        for _idx in range(0, len(_logit_list), len(UtilsTokenizer._predefined_position_replacement)):
            _logit_chunk_list.append(_logit_list[_idx: _idx + len(UtilsTokenizer._predefined_position_replacement)])
        # calc prob delta value on target label index and return
        _replaced_name_score_delta_list = []
        for _logit_chunk, _replaced_name in zip(_logit_chunk_list, _replaced_name_list):
            _tmp_logit_v = np.mean([_x[_code_info['label']] for _x in _logit_chunk])
            _tmp_logit_delta = _base_logit_v - _tmp_logit_v
            _replaced_name_score_delta_list.append({
                'name': _replaced_name,
                'score': _tmp_logit_delta
            })
        # sort
        _replaced_name_score_delta_list = sorted(_replaced_name_score_delta_list, key=lambda x: x['score'], reverse=True)
        return _replaced_name_score_delta_list
    @staticmethod
    def calc_identifier_importance_score_by_random(_args, _model_info: dict, _code_info: dict, _tokenized_info: dict, _logger=None, _cur_name2index_dict: dict = None):
        '''random choose order'''
        if _cur_name2index_dict != None:
            # reuse name2index dict, maybe for dynamic score calc
            _name2index_dict = _cur_name2index_dict
        else:
            _name2index_dict = UtilsTokenizer.get_identifier_position_name2index(
                _token_list = _tokenized_info['word_list'],
                _name_list = _tokenized_info['valid_identifier_list']
            )
        _name_list = list(_name2index_dict.keys())
        # random sort name list
        random.shuffle(_name_list)
        # prepare format: name, score (len-index)
        _name_info_list = [
            {
                'name': _name,
                'score': len(_name_list) - _idx
            } for _idx, _name in enumerate(_name_list)
        ]
        return _name_info_list
class UtilsSubstitutionGeneration():
    '''for each identifier in code, generate a set of new name as candidate'''
    @staticmethod
    def generate_random_identifier_name(_min_len = 4, _max_len = 10):
        _target_len = random.randint(_min_len, _max_len)
        first_char = random.choice(string.ascii_letters)
        remaining_chars = random.choices(string.ascii_letters + string.digits, k=_target_len-1)
        return first_char + ''.join(remaining_chars)
    @staticmethod
    def get_random_identifier_list(_size=1000):
        _random_name_list = [UtilsSubstitutionGeneration.generate_random_identifier_name() for _ in range(_size)]
        # unique, remove duplication
        _random_name_list = list(set(_random_name_list))
        return _random_name_list
    @staticmethod
    def get_subs_file_info(_args, _logger):
        if _args.substitution_type in ['token', 'code']:
            # model related path
            _subs_file = f'tg/datasets/shared/{_args.task_type}/substitution_{_args.model_name}_{_args.substitution_type}_{_args.index[0]}_{_args.index[1]}.jsonl'
        elif _args.substitution_type == 'random':
            _subs_file = f'tg/datasets/shared/{_args.task_type}/substitution_{_args.substitution_type}_{_args.index[0]}_{_args.index[1]}.jsonl'
        # replace .jsonl with _debug.jsonl if is_debug
        if _args.is_debug:
            _subs_file = _subs_file.replace('.jsonl', '_debug.jsonl')
        _existing_len = count_file_line(_subs_file)
        _logger.info(f'subs res file: {_subs_file}, existing len: {_existing_len}')
        return {
            'path': _subs_file,
            'existing_len': _existing_len
        }
    pass
    @staticmethod
    def get_masked_model_class_list(_args):
        '''
        RobertaForMaskedLM, RobertaTokenizer
        '''
        if _args.model_name in ['GraphCodeBERT', 'CodeBERT', 'CodeT5']:
            return (RobertaForMaskedLM, RobertaTokenizer)
    @staticmethod
    def get_id_list_from_sub_word_list(_args, _sub_word_list, _tokenizer, _need_pad = False):
        _sub_word_input_list = [_tokenizer.cls_token] + _sub_word_list[:_args.model_code_length - 2] + [_tokenizer.sep_token]
        _sub_word_id_list = _tokenizer.convert_tokens_to_ids(_sub_word_input_list)
        if _need_pad:
            # pad
            _pad_len = _args.model_code_length - len(_sub_word_id_list)
            _sub_word_id_list += [_tokenizer.pad_token_id] * _pad_len
        _sub_word_id_list = torch.tensor([_sub_word_id_list])
        return _sub_word_id_list
    @staticmethod
    def generate_word_prediction_list(_args: dict, _tokenized_info: dict, _model_info: dict, _top_k: int, _logger, _need_pad=False) -> dict:
        _model = _model_info['model']
        _tokenizer = _model_info['tokenizer']
        _sub_word_list = _tokenized_info['sub_word_list']
        _sub_word_id_list = UtilsSubstitutionGeneration.get_id_list_from_sub_word_list(
            _args=_args,
            _sub_word_list=_sub_word_list,
            _tokenizer=_tokenizer,
            _need_pad=_need_pad
        )
        _word_predict_total_list = _model(_sub_word_id_list.to(_args.device))[0].squeeze()
        _score_predict_list, _word_predict_list = torch.topk(_word_predict_total_list, _top_k, -1)
        # keep 1 to 1+ sub word len
        _word_predict_keep_list = _word_predict_list[1: 1 + len(_sub_word_list), :]
        _score_predict_keep_list = _score_predict_list[1: 1 + len(_sub_word_list), :]
        # log ssize/shape of word,score
        # _logger.info(f'word size: {_word_predict_keep_list.size()}, score size: {_score_predict_keep_list.size()}')
        return {
            'word_list': _word_predict_keep_list,
            'score_list': _score_predict_keep_list,
            'sub_word_id_list': _sub_word_id_list
        }
    @staticmethod
    def generate_mask_subs_dict_by_token_sim(_args, _model_info: dict, _code_info: dict, _tokenized_info: dict, _logger=None):
        _mask_pred_list = UtilsSubstitutionGeneration.generate_word_prediction_list(
            _args=_args,
            _tokenized_info=_tokenized_info,
            _model_info=_model_info,
            _top_k=60,
            _logger=_logger,
            _need_pad=False
        )
        _lang_v = _task_lang_map[_args.task_type]
        _input_id_list = _mask_pred_list['sub_word_id_list']
        with torch.no_grad():
            _input_embedding_list = _model_info['model'].roberta(_input_id_list.to(_args.device))[0]
        _cos_func = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        _name2idx_dict = UtilsTokenizer.get_code_name2idx(
            _args=_args,
            _word_list=_tokenized_info['word_list'],
            _name_list=_tokenized_info['valid_identifier_list']
        )
        _name2subs_dict = {}
        _sub_word_index_info = _tokenized_info['sub_word_index']
        _word_pred_list = _mask_pred_list['word_list']
        _score_pred_list = _mask_pred_list['score_list']
        # _sub_word_index_info format: list, item as [idx_start, idx_end]
        for _name in _name2idx_dict.keys():
            # skip if original name not valid
            if not is_valid_variable_name(_name, _lang_v):
                # _logger.info(f'skip invalid name: {_name}')
                continue
            _name_idx_list = _name2idx_dict[_name]
            _name_candi_list = []
            for _name_pos_idx in _name_idx_list:
                # avoid exceed index start
                _idx_start = _sub_word_index_info[_name_pos_idx][0]
                _idx_end = _sub_word_index_info[_name_pos_idx][1]
                if _idx_start >= _word_pred_list.size()[0]:
                    continue
                # filter out only candi at target index
                _target_word_pred_list = _word_pred_list[_idx_start : _idx_end]
                _target_score_pred_list = _score_pred_list[_idx_start : _idx_end]
                # offset cls token
                _word_vector_before = _input_embedding_list[0][_idx_start + 1 : _idx_end + 1]
                _sub_word_len, _candi_num = _target_word_pred_list.size()
                _delta_sim_list = []
                for _candi_idx in range(_candi_num):
                    _new_id_list = copy.deepcopy(_input_id_list)
                    # update id
                    _new_id_list[0][_idx_start + 1 : _idx_end + 1] = _target_word_pred_list[:, _candi_idx]
                    with torch.no_grad():
                        _new_embedding_list = _model_info['model'].roberta(_new_id_list.to(_args.device))[0]
                    _word_vector_delta = _new_embedding_list[0][_idx_start + 1 : _idx_end + 1]
                    # calc sim
                    _tmp_sim_v = sum(_cos_func(_word_vector_before, _word_vector_delta)) / _sub_word_len
                    _delta_sim_list.append((_candi_idx, _tmp_sim_v))
                _delta_sim_list = sorted(_delta_sim_list, key=lambda x: x[1], reverse=True)
                _similar_subword_pred_list = []
                _similar_score_pred_list = []
                for _sort_idx in range(int(_candi_num / 2)):
                    _original_candi_idx = _delta_sim_list[_sort_idx][0]
                    _similar_subword_pred_list.append(
                        _target_word_pred_list[:, _original_candi_idx].reshape(_sub_word_len, -1))
                    _similar_score_pred_list.append(
                        _target_score_pred_list[:, _original_candi_idx].reshape(_sub_word_len, -1))
                _similar_subword_pred_list = torch.cat(_similar_subword_pred_list, 1)
                _similar_score_pred_list = torch.cat(_similar_score_pred_list, 1)
                # _subs
                _similar_word_pred_list = UtilsTokenizer.recover_word_list_from_similar_subs(
                    _args=_args,
                    _similar_subs=_similar_subword_pred_list,
                    _similar_scores=_similar_score_pred_list,
                    _tokenizer=_model_info['tokenizer'],
                    _mlm_model=_model_info['model'],
                    _use_bpe=True,
                )
                # log max len in word list, need map item in _similar_word_pred_list into len(item)
                # _len_map_list = [len(_x) for _x in _similar_word_pred_list]
                # _logger.info(f'max len in similar word: {max(_len_map_list)}')
                _name_candi_list += _similar_word_pred_list
            _name_candi_list = list(set(_name_candi_list))
            for _name_candi in _name_candi_list:
                _name_candi_strip = _name_candi.strip()
                if _name_candi_strip in _tokenized_info['valid_identifier_list']:
                    continue
                if not is_valid_variable_name(_name_candi_strip, _lang_v):
                    continue
                # set dict
                if _name not in _name2subs_dict:
                    _name2subs_dict[_name] = []
                _name2subs_dict[_name].append(_name_candi_strip)
        # log max of each value len
        # _dict_len_list = [len(_v) for _v in _name2subs_dict.values()]
        # if len(_dict_len_list) > 0:
            # _logger.info(f'max len in name2subs: {max(_dict_len_list)}')
        return _name2subs_dict
    @staticmethod
    def generate_mask_subs_dict_by_code_sim(_args, _model_info: dict, _code_info: dict, _tokenized_info: dict, _logger=None):
        _voc_top_k = 500
        _candi_top_k = 90
        # rank subs by embedding sim of whole code
        _mask_pred_list = UtilsSubstitutionGeneration.generate_word_prediction_list(
            _args=_args,
            _tokenized_info=_tokenized_info,
            _model_info=_model_info,
            _top_k=_voc_top_k,
            _logger=_logger,
            _need_pad=True
        )
        _lang_v = _task_lang_map[_args.task_type]
        _input_id_list = _mask_pred_list['sub_word_id_list']
        with torch.no_grad():
            _input_embedding_list = _model_info['model'].roberta(_input_id_list.to(_args.device))[0]
        _cos_func = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        _name2idx_dict = UtilsTokenizer.get_code_name2idx(
            _args=_args,
            _word_list=_tokenized_info['word_list'],
            _name_list=_tokenized_info['valid_identifier_list']
        )
        _name2subs_dict = {}
        _sub_word_index_info = _tokenized_info['sub_word_index']
        _word_pred_list = _mask_pred_list['word_list']
        # _score_pred_list = _mask_pred_list['score_list']
        for _name in _name2idx_dict.keys():
            # skip if name not valid
            if not is_valid_variable_name(_name, _lang_v):
                # _logger.info(f'skip invalid name: {_name}')
                continue
            _name_idx_list = _name2idx_dict[_name]
            _name_candi_list = []
            for _name_pos_idx in _name_idx_list:
                _idx_start = _sub_word_index_info[_name_pos_idx][0]
                _idx_end = _sub_word_index_info[_name_pos_idx][1]
                # skip if index exceed
                if _idx_start >= _word_pred_list.size()[0]:
                    continue
                _name_word_pred = _word_pred_list[_idx_start : _idx_end]
                # _name_score_pred = _score_pred_list[_idx_start : _idx_end]
                _similar_word_pred_list = UtilsTokenizer.recover_word_list_from_similar_subs_no_threshold(
                    _args=_args,
                    _similar_subs=_name_word_pred,
                    _tokenizer=_model_info['tokenizer'],
                    _mlm_model=_model_info['model'],
                    _use_bpe=True
                )
                _name_candi_list += _similar_word_pred_list
            _name_candi_list = list(set(_name_candi_list))
            _tmp_name_candi_list = []
            for _name_candi in _name_candi_list:
                _name_candi_strip = _name_candi.strip()
                if _name_candi_strip in _tokenized_info['valid_identifier_list']:
                    continue
                if not is_valid_variable_name(_name_candi_strip, _lang_v):
                    continue
                _tmp_name_candi_list.append(_name_candi_strip)
            # log len change
            # _logger.info(f'len change: {len(_name_candi_list)} -> {len(_tmp_name_candi_list)}')
            _name_candi_list = _tmp_name_candi_list
            _name_sim_list = []
            for _name_candi in _tmp_name_candi_list:
                # code str from tokenized_info->token_code_str
                _candi_code_str = get_example(
                    _tokenized_info['token_code_str'],
                    _name,
                    _name_candi,
                    _lang_v
                )
                _candi_tokenized_info = UtilsTokenizer.get_tokenized_info(
                    _args=_args,
                    _code_str=_candi_code_str,
                    _tokenizer=_model_info['tokenizer'],
                    _logger=_logger
                )
                _candi_sub_word_id_list = UtilsSubstitutionGeneration.get_id_list_from_sub_word_list(
                    _args=_args,
                    _sub_word_list=_candi_tokenized_info['sub_word_list'],
                    _tokenizer=_model_info['tokenizer'],
                    _need_pad=True
                )
                _candi_embedding_list = _model_info['model'].roberta(_candi_sub_word_id_list.to(_args.device))[0]
                # log _input_embedding_list, _candi_embedding_list shape
                # _logger.info(f'input embedding shape: {_input_embedding_list.size()}, candi embedding shape: {_candi_embedding_list.size()}')
                # log sud word len of the 2: _tokenized_info,  _candi_tokenized_info
                # _logger.info(f'sub word len: {len(_tokenized_info["sub_word_list"])} -> {len(_candi_tokenized_info["sub_word_list"])}')
                _candi_sim_v = torch.mean(
                    _cos_func(_input_embedding_list, _candi_embedding_list)
                    ).cpu().detach().numpy().tolist()
                _name_sim_list.append(_candi_sim_v)
            _sort_idx_list = np.argsort(_name_sim_list)[:_candi_top_k]
            _name_candi_sort_list = np.array(_name_candi_list)[_sort_idx_list].tolist()
            # log change
            # _logger.info(f'len final change: {len(_name_candi_list)} -> {len(_name_candi_sort_list)}')
            # set dict
            _name2subs_dict[_name] = _name_candi_sort_list
        return _name2subs_dict
    @staticmethod
    def generate_mask_subs_dict_by_random(_args, _model_info: dict, _code_info: dict, _tokenized_info: dict, _logger=None, _size=90):
        _candidate_word_list = UtilsSubstitutionGeneration.get_random_identifier_list()
        # random choose name from valid_identifier_list iterately, random choose new name from candidate_info word_list, form a old_name to new_name list dict
        _name2candidate_dict = {}
        for _ in range(_size):
            # random choose one name
            _old_name = random.choice(_tokenized_info['valid_identifier_list'])
            _new_name = random.choice(_candidate_word_list)
            if _old_name not in _name2candidate_dict:
                _name2candidate_dict[_old_name] = []
            _name2candidate_dict[_old_name].append(_new_name)
            # remove new_name from _candidate_word_list
            _candidate_word_list = [_x for _x in _candidate_word_list if _x != _new_name]
        return _name2candidate_dict
class UtilsMeta():
    _default_search_beam_size = 2
    @staticmethod
    def get_args_obj(_command_str):
        _clean_str = re.sub(r'^python -u [^\s]+.py', '', _command_str).strip()
        _parser = UtilsMeta.get_args_parser()
        _args_obj = _parser.parse_args(shlex.split(_clean_str))
        set_seed(_args_obj.seed)
        return _args_obj
    @staticmethod
    def get_args_parser():
        _parser = argparse.ArgumentParser(description='Start')
        # index
        _parser.add_argument("--index", type=int, nargs='+', default=[0, 500])
        # seed, 123456
        _parser.add_argument("--seed", type=int, default=123456)
        # mode, train/attack/transfer/retraining_split/retraining
        _parser.add_argument("--mode", type=str, default='train', choices=['train', 'rank_position', 'generate_substitution', 'attack', 'fix_attack', 'attack_after_retrain', 'transfer', 'transfer_after_retrain', 'reatk_transfer', 'reatk_transfer_after_retrain', 'retraining_split', 'retraining', 'eval_test'])
        # task_type, authorship_attribution/clone_detection/code_classification/vulnerability_detection
        _parser.add_argument("--task_type", type=str, default='authorship_attribution', choices=['authorship_attribution', 'clone_detection', 'code_classification', 'vulnerability_detection'])
        # position_type: 'predefined_mask', 'predefined_replacement', 'dynamic_mask', 'dynamic_replacement', 'random'
        _parser.add_argument("--position_type", type=str, default='predefined_mask', choices=['predefined_mask', 'predefined_replacement', 'dynamic_mask', 'dynamic_replacement', 'random'])
        # substitution_type, token/code/random
        _parser.add_argument("--substitution_type", type=str, default='token', choices=['token', 'code', 'random'])
        # search_type: beam/greedy
        _parser.add_argument("--search_type", type=str, default='beam', choices=['beam', 'greedy'])
        # search_beam_size
        _parser.add_argument("--search_beam_size", type=int)
        # attack_type: greedy/beam/ga/twostep/mhm/codeattack
        _parser.add_argument("--attack_type", type=str, default='greedy', choices=['greedy', 'beam', 'ga', 'twostep', 'mhm', 'codeattack', 'random'])
        # attack_type_list, only for twostep attack
        _parser.add_argument("--attack_type_list", type=str, nargs='+', default=[])
        # model_name: GraphCodeBERT/CodeBERT/CodeT5
        _parser.add_argument("--model_name", type=str, default='GraphCodeBERT', choices=['GraphCodeBERT', 'CodeBERT', 'CodeT5'])
        # model_name_or_path
        _parser.add_argument("--model_name_or_path", type=str)
        # model_tokenizer_name
        _parser.add_argument("--model_tokenizer_name", type=str)
        # model_num_label
        _parser.add_argument("--model_num_label", type=int)
        # model_code_length
        _parser.add_argument("--model_code_length", type=int)
        # model_data_flow_length
        _parser.add_argument("--model_data_flow_length", type=int)
        # model_train_batch_size
        _parser.add_argument("--model_train_batch_size", type=int)
        # model_eval_batch_size
        _parser.add_argument("--model_eval_batch_size", type=int, default=8)
        # train_grad_accu_step
        _parser.add_argument("--train_grad_accu_step", type=int, default=1)
        # train_lr, default 5e-5
        _parser.add_argument("--train_lr", type=float, default=5e-5)
        # train_weight_decay, default 0
        _parser.add_argument("--train_weight_decay", type=float, default=0.0)
        # train_adam_epsilon, default 1e-8
        _parser.add_argument("--train_adam_epsilon", type=float, default=1e-8)
        # train_max_grad_norm, default 1.0
        _parser.add_argument("--train_max_grad_norm", type=float, default=1.0)
        # train_num_epoch, default 20
        _parser.add_argument("--train_num_epoch", type=int, default=20)
        # train_eval_data_path, train_eval_data_max_size, train_eval_step_size
        _parser.add_argument("--train_eval_data_path", type=str)
        _parser.add_argument("--train_eval_data_max_size", type=int)
        _parser.add_argument("--train_eval_step_size", type=int)
        # code_with_cache, default False
        _parser.add_argument("--code_with_cache", type=str2bool, default=False)
        # rank_with_ptm, default True
        _parser.add_argument("--rank_with_ptm", type=str2bool, default=True)
        # is_debug, default False
        _parser.add_argument("--is_debug", type=str2bool, default=False)
        # use_gpu, default True
        _parser.add_argument("--use_gpu", type=str2bool, default=True)
        # codeattack_sim_threshold, float, default 0.5
        _parser.add_argument('--codeattack_sim_threshold', type=float, default=0.5)
        # codeattack_perturbation_threshold, default 0.1
        _parser.add_argument('--codeattack_perturbation_threshold', type=float, default=0.5)
        return _parser
    @staticmethod
    def get_model_list():
        return ['CodeBERT', 'GraphCodeBERT', 'CodeT5']
    @staticmethod
    def get_task_list():
        return ['authorship_attribution', 'clone_detection', 'code_classification', 'vulnerability_detection']
    @staticmethod
    def get_attack_type_args_list(_model: string, _task: string, _mode='attack', _args=None, _replace_only_rnns=False):
        # group by type
        # return a args obj list with different attack type
        _pos_list = ['predefined_mask', 'predefined_replacement', 'random', 'dynamic_mask', 'dynamic_replacement']
        _subs_list = ['token', 'code', 'random']
        _attack_type_list = ['greedy', 'beam', 'codeattack', 'random', 'ga', 'mhm', 'twostep']
        _args_command_dict = {}
        _model_meta = UtilsMeta.get_model_meta(_model, _task)
        _dataset_meta = UtilsMeta.get_dataset_meta('attack', _task)
        _model_name = _model_meta['model_meta']['model']
        _tokenizer_name = _model_meta['model_meta']['tokenizer']
        _model_num_label = _model_meta['model_param']['model_num_label']
        _model_code_length = _model_meta['model_param']['model_code_length']
        _model_data_flow_length_str = ''
        if 'model_data_flow_length' in _model_meta['model_param']:
            _model_data_flow_length_str = f' --model_data_flow_length={_model_meta["model_param"]["model_data_flow_length"]}'
        _model_eval_batch_size = _model_meta['model_param']['batch_size']
        if _args and _args.is_debug:
            _model_eval_batch_size = _model_eval_batch_size // 8
        else:
            _model_eval_batch_size = _model_eval_batch_size // 2
        _data_size = _dataset_meta['size']
        _data_step = _dataset_meta['step']
        _data_step_count = _data_size // _data_step
        if _data_size % _data_step != 0:
            _data_step_count += 1
        for _atk_type in _attack_type_list:
            _valid_pos_list = _pos_list
            if _atk_type in ['random']:
                _valid_pos_list = ['random']
            elif _atk_type in ['codeattack', 'ga', 'mhm']:
                _valid_pos_list = ['']
            elif _atk_type in ['twostep']:
                _valid_pos_list = ['predefined_mask', 'predefined_replacement', 'random', 'dynamic_mask', 'dynamic_replacement']
            for _pos in _valid_pos_list:
                if _replace_only_rnns and _pos == 'predefined_replacement':
                    if _atk_type != 'greedy':
                        continue
                _pos_str = ''
                if _pos != '':
                    _pos_str = f' --position_type={_pos}'
                _valid_subs_list = _subs_list
                if _atk_type in ['random']:
                    _valid_subs_list = ['random']
                for _subs in _valid_subs_list:
                    _subs_str = f' --substitution_type={_subs}'
                    _search_type_list = []
                    if _atk_type in ['greedy', 'beam']:
                        _search_type_list = [_atk_type]
                    elif _atk_type in ['codeattack', 'ga', 'mhm', 'random']:
                        _search_type_list = ['']
                    elif _atk_type in ['twostep']:
                        _search_type_list = [
                            ['greedy', 'ga'],
                            ['beam', 'ga'],
                            ['mhm', 'ga'],
                            ['codeattack', 'ga']
                        ]
                    else:
                        raise Exception(f'unknown attack type: {_atk_type}')
                    _beam_size_str = ''
                    if _atk_type in ['beam']:
                        _beam_size_str = f' --search_beam_size={UtilsMeta._default_search_beam_size}'
                    for _search_type in _search_type_list:
                        _search_type_str = ''
                        _final_pos_str = _pos_str
                        if _search_type != '' and isinstance(_search_type, str):
                            _search_type_str = f' --search_type={_search_type}'
                        elif isinstance(_search_type, list):
                            _search_type_str = f' --attack_type_list {" ".join(_search_type)}'
                            if all([_search_type_item in ['mhm', 'ga', 'codeattack'] for _search_type_item in _search_type]):
                                _final_pos_str = ''
                        for _step_idx in range(_data_step_count):
                            _step_start = _step_idx * _data_step
                            _step_end = (_step_idx + 1) * _data_step
                            _command_str = f'python -u src/model/start_debug.py --mode={_mode} --task_type={_task}{_final_pos_str}{_subs_str}{_search_type_str} --attack_type={_atk_type} --model_name={_model} --model_name_or_path={_model_name} --model_tokenizer_name={_tokenizer_name} --model_num_label={_model_num_label} --model_code_length={_model_code_length}{_model_data_flow_length_str} --model_eval_batch_size={_model_eval_batch_size}{_beam_size_str} --index {_step_start} {_step_end}'
                            if _args and _args.is_debug:
                                _command_str += ' --is_debug=True'
                            _tmp_args = UtilsMeta.get_args_obj(_command_str)
                            # validate_attack_args
                            _atk_res = UtilsMeta.validate_attack_args(_tmp_args)
                            _prefix = _atk_res['prefix']
                            if _prefix not in _args_command_dict:
                                _args_command_dict[_prefix] = []
                            if _command_str not in _args_command_dict[_prefix]:
                                _args_command_dict[_prefix].append(_command_str)
        _args_list = []
        for _group_command_list in _args_command_dict.values():
            _tmp_group_args_list = [
                UtilsMeta.get_args_obj(_x) for _x in _group_command_list
            ]
            _args_list.append(_tmp_group_args_list)
        return _args_list
    @staticmethod
    def validate_attack_args(_args):
        # check model_name_or_path, model_tokenizer_name in name list
        for _tmp_name in [_args.model_name_or_path, _args.model_tokenizer_name]:
            if _tmp_name not in _available_name_path_list:
                raise ValueError(f"model_name_or_path, model_tokenizer_name must in {_available_name_path_list}, {_args}")
        # throw error if attack type random but position/subs not random
        if _args.attack_type == 'random' and (_args.position_type != 'random' or _args.substitution_type != 'random'):
            raise ValueError(f'attack type random must set position/substitution type random, current: {_args}')
        if _args.search_type in ['greedy']:
            _args.search_beam_size = 1
        _target_dir = f'tg/datasets/attack_res/{_args.model_name}_{_args.task_type}'
        # create if not exist
        if not os.path.exists(_target_dir):
            os.makedirs(_target_dir, exist_ok=True)
        # prepare valid combination, attack: ['greedy', 'beam', 'ga', 'twostep', 'mhm']
        _file_path = None
        _file_prefix = None
        if _args.attack_type in ['mhm', 'codeattack', 'ga']:
            # differ in subs, pos no differ
            # example file path: tg/datasets/attack_res/{model_name}_{task_type}/{attack_type}_{substitution_type}_{index1}_{index2}.jsonl
            # codeattack generate substitue on the fly and bsed on constraint, so it's totally dynamic, similar with mgm
            _file_path = f'{_target_dir}/{_args.attack_type}_{_args.substitution_type}_{_args.index[0]}_{_args.index[1]}.jsonl'
            _file_prefix = f'{_args.attack_type}_{_args.substitution_type}'
        if _args.attack_type in ['random']:
            _file_path = f'{_target_dir}/random_{_args.index[0]}_{_args.index[1]}.jsonl'
            _file_prefix = 'random'
        elif _args.attack_type in ['pwvs']:
            raise ValueError(f'attack type pwvs not valid for now, {_args}')
            # raise error if search type not in ['greedy', 'beam']
            if _args.search_type not in ['greedy', 'beam']:
                raise ValueError(f'search type must in [greedy, beam] for pwvs attack, current: {_args.search_type}')
            # differ in subs, pos no differ, search type ? (pwvs + greedy / beam?) -> beam_{beam_size}
            # example file path: tg/datasets/attack_res/{model_name}_{task_type}/{attack_type}_{substitution_type}_{search_type}_{index1}_{index2}.jsonl, greedy -> beam_1
            _file_path = f'{_target_dir}/{_args.attack_type}_{_args.substitution_type}_beam_{_args.search_beam_size}_{_args.index[0]}_{_args.index[1]}.jsonl'
        elif _args.attack_type in ['greedy', 'beam']:
            # consider pos frist, then subs
            # example file path: tg/datasets/attack_res/{model_name}_{task_type}/{attack_type}_{position_type}_{substitution_type}_{index1}_{index2}.jsonl
            _file_path = f'{_target_dir}/{_args.attack_type}_{_args.position_type}_{_args.substitution_type}_{_args.index[0]}_{_args.index[1]}.jsonl'
            _file_prefix = f'{_args.attack_type}_{_args.position_type}_{_args.substitution_type}'
        elif _args.attack_type in ['twostep']:
            # check attack_type_list len, raise if len not 2
            if len(_args.attack_type_list) != 2:
                raise ValueError(f'attack type twostep must have 2 attack type, current: {_args.attack_type_list}')
            # raise error if sub type in attack_type_list not in ['greedy' / 'beam' + 'mhm' / 'ga' / 'codeattack']
            for _tmp_sub_type in _args.attack_type_list:
                if _tmp_sub_type not in ['greedy', 'beam', 'mhm', 'ga', 'codeattack']:
                    raise ValueError(f'attack type in twostep not valid: {_tmp_sub_type}')
            _attack_type_label = f'{_args.attack_type}_{"_".join(_args.attack_type_list)}'
            # if all sub type in mhm,ga,codeattack
            if all([_x in ['mhm', 'ga', 'codeattack'] for _x in _args.attack_type_list]):
                # ignore pos type in twostep res
                _file_path = f'{_target_dir}/{_attack_type_label}_{_args.substitution_type}_{_args.index[0]}_{_args.index[1]}.jsonl'
                _file_prefix = f'{_attack_type_label}_{_args.substitution_type}'
            else:
                _file_path = f'{_target_dir}/{_attack_type_label}_{_args.position_type}_{_args.substitution_type}_{_args.index[0]}_{_args.index[1]}.jsonl'
                _file_prefix = f'{_attack_type_label}_{_args.position_type}_{_args.substitution_type}'
        # raise error if file path none
        if _file_path == None:
            raise ValueError(f'attack type not valid: {_args.attack_type}')
        # check _file_path exist
        _existing_len = count_file_line(_file_path)
        # calc target len by index diff first, then case size
        _target_end = _args.index[1]
        _dataset_meta = UtilsMeta.get_dataset_meta('attack', _args.task_type)
        if _target_end > _dataset_meta['size']:
           _target_end = _dataset_meta['size']
        _target_len = _target_end - _args.index[0]
        return {
            'path': _file_path,
            'prefix': _file_prefix,
            'existing_len': _existing_len,
            'target_len': _target_len
        }
        # concat unique attack_label
        # attack_label list: 
        # part 1: rank pos: 
        # part 1.1 static model related mask/replacement/random
        # part 1.2 dynamic mask/replacement
        # part 2: substitution:
        # model-free: code/token/random
        # part 3: attach type: ga, greedy, beam, mhm, pwvs
        # pwvs is ranking target old variable by any possible substitues, so it's totally dynamic, only differ in subs (greedy+dynamic)
    @staticmethod
    def validate_reattack_args(_args):
        # set index file into args, example: tg/datasets/shared/authorship_attribution/retraining_test_index.json
        _args.reattack_index_path = f'tg/datasets/shared/{_args.task_type}/retraining_test_index.json'
        # reuse validate_attack_args
        # update path with suffix _reatk.jsonl
        _first_atk_res_info = UtilsMeta.validate_attack_args(_args)
        _path = _first_atk_res_info['path'].replace('.jsonl', '_reatk.jsonl')
        _existing_len = count_file_line(_path)
        # calc target len by reattack_index_path index item in index0, index1
        _target_start = _args.index[0]
        _target_end = _args.index[1]
        _dataset_meta = UtilsMeta.get_dataset_meta('attack', _args.task_type)
        if _target_end > _dataset_meta['size']:
            _target_end = _dataset_meta['size']
        # def root by relative path based on cur file ../../
        _root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        _to_atk_index_list = json.load(open(f'{_root_dir}/{_args.reattack_index_path}', 'r'))
        _target_idx_list = [
            _idx for _idx in _to_atk_index_list if _idx >= _target_start and _idx < _target_end
        ]
        # _target_len = len(_target_idx_list)
        _target_len = _first_atk_res_info['target_len']
        return {
            'path': _path,
            'prefix': _first_atk_res_info['prefix'],
            'existing_len': _existing_len,
            'target_len': _target_len
        }
    @staticmethod
    def validate_retraining_args(_args):
        # set train batch size by divide 16 _gpu_db = UtilsMeta.get_gpu_gb()
        if _args.model_train_batch_size is not None:
            _gpu_db = UtilsMeta.get_gpu_gb()
            _train_batch_size = int (_args.model_train_batch_size * _gpu_db / 16)
            _args.model_train_batch_size = _train_batch_size
        # set retraining index train/test path into args
        _args.retraining_train_index_path = f'tg/datasets/shared/{_args.task_type}/retraining_train_index.json'
        _args.retraining_test_index_path = f'tg/datasets/shared/{_args.task_type}/retraining_test_index.json'
        # train file author: train.txt, clone: train_sampled.txt, classification/vulnerability: train.jsonl
        if _args.task_type == 'authorship_attribution':
            _args.train_code_txt_path = f'tg/datasets/shared/authorship_attribution/train.txt'
        elif _args.task_type == 'clone_detection':
            _args.train_code_txt_path = f'tg/datasets/shared/clone_detection/train_sampled.txt'
            _args.total_jsonl_path = f'tg/datasets/shared/clone_detection/data.jsonl'
        elif _args.task_type in ['code_classification', 'vulnerability_detection']:
            _args.train_code_jsonl_path = f'tg/datasets/shared/{_args.task_type}/train.jsonl'
    @staticmethod
    def get_model_meta(_model_name, _task_type):
        _model_meta = {
            'CodeBERT': {
                'model': 'microsoft/codebert-base',
                'tokenizer': 'microsoft/codebert-base',
                'model_mlm': 'microsoft/codebert-base-mlm'
            },
            'GraphCodeBERT': {
                'model': 'microsoft/graphcodebert-base',
                'tokenizer': 'microsoft/graphcodebert-base'
            },
            'CodeT5': {
                'model': 'Salesforce/codet5-base-multi-sum',
                'tokenizer': 'Salesforce/codet5-base'
            }
        }
        _model_param_meta = {
            'CodeBERT': {
                'authorship_attribution': {
                    'model_code_length': 512,
                    'model_num_label': 66,
                    # not train batch, just eval batch
                    'batch_size': 32,
                    'model_train_batch_size': 20,
                    'model_retrain_num_epoch': 6,
                    'model_retrain_hour_per_epoch': 0.05
                },
                'clone_detection': {
                    'model_code_length': 512,
                    'model_num_label': 2,
                    'batch_size': 32,
                    'model_train_batch_size': 10,
                    'model_retrain_num_epoch': 2,
                    'model_retrain_hour_per_epoch': 4
                },
                'code_classification': {
                    'model_code_length': 512,
                    'model_num_label': 250,
                    'batch_size': 32,
                    'model_train_batch_size': 16,
                    'model_retrain_num_epoch': 2,
                    'model_retrain_hour_per_epoch': 2
                },
                'vulnerability_detection': {
                    'model_code_length': 512,
                    'model_num_label': 1,
                    'batch_size': 32,
                    'model_train_batch_size': 16,
                    'model_retrain_num_epoch': 2,
                    'model_retrain_hour_per_epoch': 0.5
                }
            },
            'GraphCodeBERT': {
                'authorship_attribution': {
                    'model_code_length': 384,
                    'model_data_flow_length': 128,
                    'model_num_label': 66,
                    'batch_size': 32,
                    'model_train_batch_size': 16,
                    'model_retrain_num_epoch': 6,
                    'model_retrain_hour_per_epoch': 0.05
                },
                'clone_detection': {
                    'model_code_length': 384,
                    'model_data_flow_length': 128,
                    'model_num_label': 1,
                    'batch_size': 16,
                    'model_train_batch_size': 10,
                    'model_retrain_num_epoch': 2,
                    'model_retrain_hour_per_epoch': 6
                },
                'code_classification': {
                    'model_code_length': 384,
                    'model_data_flow_length': 128,
                    'model_num_label': 250,
                    'batch_size': 16,
                    'model_train_batch_size': 20,
                    'model_retrain_num_epoch': 2,
                    'model_retrain_hour_per_epoch': 2
                },
                'vulnerability_detection': {
                    'model_code_length': 384,
                    'model_data_flow_length': 128,
                    'model_num_label': 1,
                    'batch_size': 32,
                    'model_train_batch_size': 20,
                    'model_retrain_num_epoch': 2,
                    'model_retrain_hour_per_epoch': 0.8
                }
            },
            'CodeT5': {
                'authorship_attribution': {
                    'model_code_length': 384,
                    'model_num_label': 66,
                    'batch_size': 16,
                    'model_train_batch_size': 8,
                    'model_retrain_num_epoch': 6,
                    'model_retrain_hour_per_epoch': 0.08
                },
                'clone_detection': {
                    'model_code_length': 384,
                    'model_num_label': 2,
                    'batch_size': 12,
                    'model_train_batch_size': 4,
                    'model_retrain_num_epoch': 2,
                    'model_retrain_hour_per_epoch': 8
                },
                'code_classification': {
                    'model_code_length': 384,
                    'model_num_label': 250,
                    'batch_size': 12,
                    'model_train_batch_size': 8,
                    'model_retrain_num_epoch': 2,
                    'model_retrain_hour_per_epoch': 3
                },
                'vulnerability_detection': {
                    'model_code_length': 384,
                    'model_num_label': 1,
                    'batch_size': 16,
                    'model_train_batch_size': 8,
                    'model_retrain_num_epoch': 2,
                    'model_retrain_hour_per_epoch': 2
                }
            }
        }
        _model_meta_info = _model_meta[_model_name]
        _model_param_info = _model_param_meta[_model_name][_task_type]
        return {
            'model_meta': _model_meta_info,
            'model_param': _model_param_info
        }
    @staticmethod
    def get_dataset_meta(_mode, _task_type):
        if _mode in ['rank_position', 'generate_substitution', 'attack', 'fix_attack', 'transfer', 'eval_test', 'retraining_split', 'retraining', 'attack_after_retrain']:
            _dataset_meta = {
                'authorship_attribution': {
                    'size': 132,
                    'step': 500
                },
                'clone_detection': {
                    'size': 4000,
                    'step': 500
                },
                'code_classification': {
                    'size': 12500,
                    'step': 500
                },
                'vulnerability_detection': {
                    'size': 2732,
                    'step': 500,
                }
            }
            return _dataset_meta[_task_type]
        else:
            return None
    @staticmethod
    def prepare_attack_file(_args):
        if _args.task_type == 'authorship_attribution':
            _args.code_txt_path = f'tg/datasets/shared/authorship_attribution/valid.txt'
        elif _args.task_type == 'clone_detection':
            _args.code_txt_path = f'tg/datasets/shared/clone_detection/test_{_args.index[0]}_{_args.index[1]}.txt'
            # total_jsonl_path: tg/datasets/shared/clone_detection/data.jsonl
            _args.total_jsonl_path = f'tg/datasets/shared/clone_detection/data.jsonl'
        elif _args.task_type in ['code_classification', 'vulnerability_detection']:
            _args.code_jsonl_path = f'tg/datasets/shared/{_args.task_type}/test_{_args.index[0]}_{_args.index[1]}.jsonl'
    @staticmethod
    def prepare_attack_pos_file(_args):
        if _args.position_type == 'random':
            # example: tg/datasets/shared/authorship_attribution/position_random_0_500.jsonl
            _args.pos_jsonl_path = f'tg/datasets/shared/{_args.task_type}/position_{_args.position_type}_{_args.index[0]}_{_args.index[1]}.jsonl'
        elif _args.position_type in ['predefined_mask', 'predefined_replacement']:
            # rand model should always be same with attack model
            # example: tg/datasets/shared/authorship_attribution/position_CodeBERT_predefined_mask_0_500.jsonl
            _args.pos_jsonl_path = f'tg/datasets/shared/{_args.task_type}/position_{_args.model_name}_{_args.position_type}_{_args.index[0]}_{_args.index[1]}.jsonl'
        # random, predefined_mask, predefined_replacement
        # model related if predefined_mask, predefined_replacement
    @staticmethod
    def prepare_attack_subs_file(_args):
        # random is model free
        if _args.substitution_type in ['random']:
            # example: tg/datasets/shared/authorship_attribution/substitution_random_0_500.jsonl
            _args.subs_jsonl_path = f'tg/datasets/shared/{_args.task_type}/substitution_{_args.substitution_type}_{_args.index[0]}_{_args.index[1]}.jsonl'
        elif _args.substitution_type in ['token', 'code']:
            # example: tg/datasets/shared/authorship_attribution/substitution_CodeBERT_token_0_500.jsonl
            _args.subs_jsonl_path = f'tg/datasets/shared/{_args.task_type}/substitution_CodeBERT_{_args.substitution_type}_{_args.index[0]}_{_args.index[1]}.jsonl'
        # random, token, code
        # all model-free (since token and code are both based on codebert-base-mlm)
    @staticmethod
    def get_gpu_gb():
        if not torch.cuda.is_available():
            return 0
        gpu_memory_in_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_in_GB = gpu_memory_in_bytes / (1024 ** 3)  # Convert bytes to GB
        return gpu_memory_in_GB

class UtilsAttackerGA():
    @staticmethod
    def compute_chrome_fitness(_args, _chrome_dict, _model_info, _code_info, _attack_info, _logger):
        _mutated_code_str = get_example_batch(_attack_info['code_str'], _chrome_dict, _attack_info['lang'])
        _mutated_code_info = copy.deepcopy(_code_info)
        _mutated_code_info['code'] = _mutated_code_str
        _mutated_code_dataset = UtilsFeature.get_code_dataset(
            _args=_args,
            _code_info_list=[_mutated_code_info],
            _tokenizer=_model_info['tokenizer'],
            _logger=_logger
        )
        _mutated_logit_list, _mutated_label_pred_list = _model_info['model'].get_dataset_result(
            _dataset=_mutated_code_dataset,
            _batch_size=_args.model_eval_batch_size
        )
        # original prob in attack info subtract mutated prob
        _mutated_fitness_value =  _attack_info['prob'] - _mutated_logit_list[0][_code_info['label']]
        _mutated_label_pred = _mutated_label_pred_list[0]
        return {
            'fitness_value': _mutated_fitness_value,
            'label_pred': _mutated_label_pred
        }
    @staticmethod
    def random_select_parent(_population_list):
        _len_idx_list = range(len(_population_list))
        _idx_1 = random.choice(_len_idx_list)
        _idx_2 = random.choice(_len_idx_list)
        _chromesome_1 = _population_list[_idx_1]
        _chromesome_2 = _population_list[_idx_2]
        return (_chromesome_1, _idx_1), (_chromesome_2, _idx_2)
    @staticmethod
    def random_mutate(_chrome_dict, _subs_dict, _logger=None):
        if len(_chrome_dict) == 0:
            return _chrome_dict
        _target_chrome_idx = random.choice(range(len(_chrome_dict)))
        _target_name = list(_chrome_dict.keys())[_target_chrome_idx]
        _chrome_dict[_target_name] = random.choice(_subs_dict[_target_name])
        return _chrome_dict
    @staticmethod
    def random_crossover(_chrome_dict_1, _chrome_dict_2, _logger=None):
        # random idx not 0
        _random_idx = random.choice(range(len(_chrome_dict_1))) + 1
        _child_1 = {}
        _child_2 = {}
        for _name_idx, _name in enumerate(_chrome_dict_1.keys()):
            if _name_idx < _random_idx:
                _child_2[_name] = _chrome_dict_1[_name]
                _child_1[_name] = _chrome_dict_2[_name]
            else:
                _child_1[_name] = _chrome_dict_1[_name]
                _child_2[_name] = _chrome_dict_2[_name]
        return _child_1, _child_2
class UtilsAttackerMHM():
    __key_words__ = ["auto", "break", "case", "char", "const", "continue",
                 "default", "do", "double", "else", "enum", "extern",
                 "float", "for", "goto", "if", "inline", "int", "long",
                 "register", "restrict", "return", "short", "signed",
                 "sizeof", "static", "struct", "switch", "typedef",
                 "union", "unsigned", "void", "volatile", "while",
                 "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                 "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                 "_Thread_local", "__func__"]
    __ops__ = ["...", ">>=", "<<=", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=",
           ">>", "<<", "++", "--", "->", "&&", "||", "<=", ">=", "==", "!=", ";",
           "{", "<%", "}", "%>", ",", ":", "=", "(", ")", "[", "<:", "]", ":>",
           ".", "&", "!", "~", "-", "+", "*", "/", "%", "<", ">", "^", "|", "?"]
    __macros__ = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
                "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
                "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]     # <stdlib.h> macro
    __special_ids__ = ["main",  # main function
                    "stdio", "cstdio", "stdio.h",                                # <stdio.h> & <cstdio>
                    "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",     # <stdio.h> types & streams
                    "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush", # <stdio.h> functions
                    "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                    "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                    "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                    "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                    "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                    "rewind", "clearerr", "feof", "ferror", "perror", "getline"
                    "stdlib", "cstdlib", "stdlib.h",                             # <stdlib.h> & <cstdlib>
                    "size_t", "div_t", "ldiv_t", "lldiv_t",                      # <stdlib.h> types
                    "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",  # <stdlib.h> functions
                    "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                    "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                    "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                    "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                    "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                    "mbstowcs", "wcstombs",
                    "string", "cstring", "string.h",                                 # <string.h> & <cstring>
                    "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",     # <string.h> functions
                    "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                    "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                    "strpbrk" ,"strstr", "strtok", "strxfrm",
                    "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",      # <string.h> extension functions
                    "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                    "iostream", "istream", "ostream", "fstream", "sstream",      # <iostream> family
                    "iomanip", "iosfwd",
                    "ios", "wios", "streamoff", "streampos", "wstreampos",       # <iostream> types
                    "streamsize", "cout", "cerr", "clog", "cin",
                    "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",    # <iostream> manipulators
                    "noshowbase", "showpoint", "noshowpoint", "showpos",
                    "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                    "left", "right", "internal", "dec", "oct", "hex", "fixed",
                    "scientific", "hexfloat", "defaultfloat", "width", "fill",
                    "precision", "endl", "ends", "flush", "ws", "showpoint",
                    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",    # <math.h> functions
                    "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                    "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                    "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]
    _mhm_status_map = {
        'success': 's',
        'reject': 'r',
        'accept': 'a'
    }
    @staticmethod
    def is_valid_name(_text=""):
        '''
        Return if a token is a UID.
        '''
        
        _text = _text.strip()
        if _text == '':
            return False

        if " " in _text or "\n" in _text or "\r" in _text:
            return False
        elif _text in UtilsAttackerMHM.__key_words__:
            return False
        elif _text in UtilsAttackerMHM.__ops__:
            return False
        elif _text in UtilsAttackerMHM.__macros__:
            return False
        elif _text in UtilsAttackerMHM.__special_ids__:
            return False
        elif _text[0].lower() in "0123456789":
            return False
        elif "'" in _text or '"' in _text:
            return False
        elif _text[0].lower() in "abcdefghijklmnopqrstuvwxyz_":
            for _c in _text[1:-1]:
                if _c.lower() not in "0123456789abcdefghijklmnopqrstuvwxyz_":
                    return False
        else:
            return False
        return True
    @staticmethod
    def replace_uid(_args, _attack_info: dict, _code_info: dict, _model_info: dict, _name2index_dict: dict, _subs_dict: dict, _candi_n: int, _prob_threshold: float, _logger):
        # check subs dict empty or none
        if _subs_dict is None or len(_subs_dict) == 0:
           return {
               'status': _status_map['fail'],
               'new_code_str': _attack_info['code_str'],
               'mhm_status': UtilsAttackerMHM._mhm_status_map['reject'],
               'mhm_alpha': 0,
               'mhm_old_name': None,
               'mhm_new_name': None,
               'mhm_changed_pos_count': 0
           }
        # random select target name
        _target_name = random.sample(_subs_dict.keys(), 1)[0]
        _changed_pos_count = 0
        if _target_name in _name2index_dict.keys():
            _changed_pos_count = len(_name2index_dict[_target_name])
        _candi_token_list = [_target_name]
        _candi_code_list = [copy.deepcopy(_attack_info['code_str'])]
        _candi_label_list = [_code_info['label']]
        for _candi_name in random.sample(_subs_dict[_target_name], min(_candi_n, len(_subs_dict[_target_name]))):
            if _candi_name in _name2index_dict:
                # already in code
                continue
            if UtilsAttackerMHM.is_valid_name(_candi_name):
                _candi_token_list.append(_candi_name)
                _candi_code_list.append(copy.deepcopy(_attack_info['code_str']))
                _candi_label_list.append(_code_info['label'])
                _candi_code_list[-1] = get_example(
                    _candi_code_list[-1],
                    _target_name,
                    _candi_name,
                    _attack_info['lang']
                )
        # convert into dataset and then evaluate
        _candi_code_info_list = []
        for _idx, _candi_code_str in enumerate(_candi_code_list):
            _tmp_code_info = copy.deepcopy(_code_info)
            _tmp_code_info['code'] = _candi_code_str
            _candi_code_info_list.append(_tmp_code_info)
        _candi_code_dataset = UtilsFeature.get_code_dataset(
            _args=_args,
            _code_info_list=_candi_code_info_list,
            _tokenizer=_model_info['tokenizer'],
            _logger=_logger
        )
        _candi_logit_list, _candi_label_pred_list = _model_info['model'].get_dataset_result(
            _dataset=_candi_code_dataset,
            _batch_size=_args.model_eval_batch_size
        )
        # check whether it succeed
        for _candi_idx, _candi_logit in enumerate(_candi_logit_list):
            if _candi_label_pred_list[_candi_idx] != _code_info['label']:
                # # success, update code, replace_info
                # _attack_info['replace_info'].append({
                #     'old_name': _target_name,
                #     'new_name': _candi_token_list[_candi_idx],
                #     'changed_pos_count': len(_name2index_dict[_target_name])
                # })
                return {
                    'status': _status_map['success'],
                    'new_code_str': _candi_code_list[_candi_idx],
                    'mhm_status': UtilsAttackerMHM._mhm_status_map['success'],
                    'mhm_alpha': 1,
                    'mhm_old_name': _target_name,
                    'mhm_new_name': _candi_token_list[_candi_idx],
                    'mhm_changed_pos_count': _changed_pos_count
                }
        # find the idx with minimum prob on target label
        _target_idx = 0
        _min_prob = 1
        for _candi_idx, _candi_logit in enumerate(_candi_logit_list):
            if _candi_logit[_code_info['label']] < _min_prob:
                _min_prob = _candi_logit[_code_info['label']]
                _target_idx = _candi_idx
        # calc acceptance rate
        # alpha = (1-prob[candi_idx][_label]+1e-10) / (1-prob[0][_label]+1e-10)
        _alpha_v = (1 - _candi_logit_list[_target_idx][_code_info['label']] + 1e-10) / (1 - _candi_logit_list[0][_code_info['label']] + 1e-10)
        if random.uniform(0, 1) > _alpha_v or _alpha_v < _prob_threshold:
            return {
                'status': _status_map['fail'],
                'new_code_str': _attack_info['code_str'],
                'mhm_status': UtilsAttackerMHM._mhm_status_map['reject'],
                'mhm_alpha': _alpha_v,
                'mhm_changed_pos_count': _changed_pos_count
            }
        else:
            return {
                'status': _status_map['fail'],
                'new_code_str': _candi_code_list[-1],
                'mhm_status': UtilsAttackerMHM._mhm_status_map['accept'],
                'mhm_alpha': _alpha_v,
                'mhm_old_name': _target_name,
                'mhm_new_name': _candi_token_list[-1],
                'mhm_changed_pos_count': _changed_pos_count
            }
class UtilsAttacker():
    @staticmethod
    def filter_out_pos_subs_name(_args, _attack_info, _cur_name_list):
        _pos_list = _attack_info['pos']
        _subs_dict = _attack_info['subs']
        if _args.position_type in ['dynamic_mask', 'dynamic_replacement']:
            _pos_name_list = list(_subs_dict.keys())
        else:
            _pos_name_list = [x['name'] for x in _pos_list]
        _new_name_list = []
        for _name in _cur_name_list:
            if _name in _pos_name_list and _name in _subs_dict and len(_subs_dict[_name]) > 0:
                _new_name_list.append(_name)
        _valid_pos_list = [_pos for _pos in _pos_list if _pos['name'] in _new_name_list]
        _valid_subs_dict = {k: v for k, v in _subs_dict.items() if len(v) > 0 and k in _new_name_list}
        return {
            'name': _new_name_list,
            'pos': _valid_pos_list,
            'subs': _valid_subs_dict
        }
    @staticmethod
    def launch_attack_greedy(_args, _model_info, _code_info, _attack_info, _logger):
        # get cur token info
        _cur_tokenized_info = UtilsTokenizer.get_tokenized_info(
            _args=_args,
            _code_str=_attack_info['code_str'],
            _tokenizer=_model_info['tokenizer'],
            _logger=_logger
        )
        _cur_variable_name_list = list(_attack_info['subs'].keys())
        # filter by tokenized info word list
        _cur_variable_name_list = [_x for _x in _cur_variable_name_list if _x in _cur_tokenized_info['word_list']]
        # directly return if variable empty
        if len(_cur_variable_name_list) == 0:
            return {
                'status': _status_map['fail'],
                'replace_info': _attack_info['replace_info'],
                'new_code_str': _attack_info['code_str']
            }
        _name2index_dict = UtilsTokenizer.get_identifier_position_name2index(
            _token_list=_cur_tokenized_info['word_list'],
            _name_list=_cur_variable_name_list
        )
        # if position_type in dynamic_mask, dynamic_replacement, calculate target variable position score on the fly, otherwise use the predefined position
        if _args.position_type in ['dynamic_mask', 'dynamic_replacement', 'random',
                                   'predefined_mask', 'predefined_replacement']:
            # calc importance order in each loop
            _need_continue = True
            _res_info = None
            while _need_continue:
                _filtered_info = UtilsAttacker.filter_out_pos_subs_name(_args, _attack_info, _cur_variable_name_list)
                _valid_pos_list = _filtered_info['pos']
                _valid_subs_dict = _filtered_info['subs']
                _cur_variable_name_list = _filtered_info['name']
                # check direct return
                if (_args.position_type in ['predefined_mask',      'predefined_replacement', 'random'] and len(_valid_pos_list) == 0) \
                    or len(_valid_subs_dict.keys()) == 0:
                    _need_continue = False
                    _res_info = {
                        'status': _status_map['fail'],
                        'replace_info': _attack_info['replace_info'],
                        'new_code_str': _attack_info['code_str']
                    }
                    break
                # calc importance score
                if _args.position_type in ['dynamic_mask', 'dynamic_replacement']:
                    _tmp_code_info = copy.deepcopy(_code_info)
                    _tmp_code_info['code'] = _attack_info['code_str']
                    _tmp_name2index_dict = copy.deepcopy(_name2index_dict)
                    # filter name2index by key in cur name list
                    _tmp_name2index_dict = {k: v for k, v in _tmp_name2index_dict.items() if k in _cur_variable_name_list}
                    if _args.position_type == 'dynamic_mask':
                        _name_score_list = UtilsPositionScore.calc_identifier_importance_score_by_mask(
                            _args=_args,
                            _model_info=_model_info,
                            _code_info=_tmp_code_info,
                            _tokenized_info=_cur_tokenized_info,
                            _cur_name2index_dict=_tmp_name2index_dict,
                            _logger=_logger,
                        )
                    elif _args.position_type == 'dynamic_replacement':
                        _name_score_list = UtilsPositionScore.calc_identifier_importance_score_by_predefined_replacement(
                            _args=_args,
                            _model_info=_model_info,
                            _code_info=_tmp_code_info,
                            _tokenized_info=_cur_tokenized_info,
                            _cur_name2index_dict=_tmp_name2index_dict,
                            _logger=_logger,
                        )
                elif _args.position_type in ['predefined_mask', 'predefined_replacement', 'random']:
                    _name_score_list = _valid_pos_list
                _target_old_name = _name_score_list[0]['name']
                _target_sub_list = _valid_subs_dict[_target_old_name]
                _candi_code_info_list = []
                for _target_sub in _target_sub_list:
                    _tmp_new_code = get_example(
                        code=_attack_info['code_str'],
                        tgt_word=_target_old_name,
                        substitute=_target_sub,
                        lang=_attack_info['lang']
                    )
                    _tmp_code_info = copy.deepcopy(_code_info)
                    _tmp_code_info['code'] = _tmp_new_code
                    _candi_code_info_list.append(_tmp_code_info)
                # eval
                _candi_dataset = UtilsFeature.get_code_dataset(
                    _args=_args,
                    _code_info_list=_candi_code_info_list,
                    _tokenizer=_model_info['tokenizer'],
                    _logger=_logger
                )
                _candi_logit_list, _candi_label_pred_list = _model_info['model'].get_dataset_result(
                    _dataset=_candi_dataset,
                    _batch_size=_args.model_eval_batch_size
                )
                # direct return if some pred label != label
                _success_candi_index = [
                    _idx for _idx, _pred_label in enumerate(_candi_label_pred_list) if _pred_label != _code_info['label']
                ]
                if len(_success_candi_index) > 0:
                    _success_idx = _success_candi_index[0]
                    # prepare replace_info
                    _attack_info['replace_info'].append({
                        'old_name': _target_old_name,
                        'new_name': _target_sub_list[_success_idx],
                        'changed_pos_count': len(_name2index_dict[_target_old_name])
                    })
                    _need_continue = False
                    _res_info = {
                        'status': _status_map['success'],
                        'replace_info': _attack_info['replace_info'],
                        'new_code_str': _candi_code_info_list[_success_idx]['code'],
                    }
                else:
                    # find max prob change item as cur best, which means the minimum prob in list
                    _candi_prob_list = [_logit_list[_code_info['label']] for _logit_list in _candi_logit_list]
                    _min_prob_index = np.argmin(_candi_prob_list)
                    _attack_info['replace_info'].append({
                        'old_name': _target_old_name,
                        'new_name': _target_sub_list[_min_prob_index],
                        'changed_pos_count': len(_name2index_dict[_target_old_name])
                    })
                    # iterate, update pos, subs
                    _attack_info['pos'] = _valid_pos_list[1:]
                    _attack_info['subs'] = {k: v for k, v in _valid_subs_dict.items() if k != _target_old_name}
                    _attack_info['code_str'] = _candi_code_info_list[_min_prob_index]['code']
                    if _args.is_debug:
                        # log
                        _logger.info(f'candi old name: {_target_old_name}, new name: {_target_sub_list[_min_prob_index]}, old prob: {_attack_info["prob"]}, new prob: {_candi_prob_list[_min_prob_index]}')
            if _res_info is not None:
                return _res_info
            else:
                raise ValueError('attack fail')
    @staticmethod
    def launch_attack_beam(_args, _model_info, _code_info, _attack_info, _logger):
        # use beam search to find the best replacement
        _cur_tokenized_info = UtilsTokenizer.get_tokenized_info(
            _args=_args,
            _code_str=_attack_info['code_str'],
            _tokenizer=_model_info['tokenizer'],
            _logger=_logger
        )
        _cur_variable_name_list = list(_attack_info['subs'].keys())
        _cur_variable_name_list = [_x for _x in _cur_variable_name_list if _x in _cur_tokenized_info['word_list']]
        if len(_cur_variable_name_list) == 0:
            return {
                'status': _status_map['fail'],
                'replace_info': _attack_info['replace_info'],
                'new_code_str': _attack_info['code_str']
            }
        _name2index_dict = UtilsTokenizer.get_identifier_position_name2index(
            _token_list=_cur_tokenized_info['word_list'],
            _name_list=_cur_variable_name_list
        )
        if _args.position_type in ['predefined_mask', 'predefined_replacement', 'random', 'dynamic_mask', 'dynamic_replacement']:
            _need_continue = True
            _res_info_list = []
            _beam_info_list = [_attack_info]
            while _need_continue:
                _cur_beam_info_list = []
                for _old_beam_info in _beam_info_list:
                    if not _need_continue:
                        break
                    _filtered_info = UtilsAttacker.filter_out_pos_subs_name(_args, _old_beam_info, _cur_variable_name_list)
                    _valid_pos_list = _filtered_info['pos']
                    _valid_subs_dict = _filtered_info['subs']
                    _cur_variable_name_list = _filtered_info['name']
                    if (_args.position_type in ['predefined_mask',      'predefined_replacement', 'random'] and len(_valid_pos_list) == 0) \
                        or len(_valid_subs_dict.keys()) == 0:
                        _res_info_list.append({
                            'status': _status_map['fail'],
                            'replace_info': _old_beam_info['replace_info'],
                            'new_code_str': _old_beam_info['code_str']
                        })
                        continue
                    if _args.position_type in ['predefined_mask', 'predefined_replacement', 'random']:
                        _name_score_list = _valid_pos_list
                    elif _args.position_type in ['dynamic_mask', 'dynamic_replacement']:
                        _tmp_code_info = copy.deepcopy(_code_info)
                        _tmp_code_info['code'] = _old_beam_info['code_str']
                        _tmp_name2index_dict = copy.deepcopy(_name2index_dict)
                        _tmp_name2index_dict = {k: v for k, v in _tmp_name2index_dict.items() if k in _cur_variable_name_list}
                        if _args.position_type == 'dynamic_mask':
                            _name_score_list = UtilsPositionScore.calc_identifier_importance_score_by_mask(
                                _args=_args,
                                _model_info=_model_info,
                                _code_info=_tmp_code_info,
                                _tokenized_info=_cur_tokenized_info,
                                _cur_name2index_dict=_tmp_name2index_dict,
                                _logger=_logger,
                            )
                        elif _args.position_type == 'dynamic_replacement':
                            _name_score_list = UtilsPositionScore.calc_identifier_importance_score_by_predefined_replacement(
                                _args=_args,
                                _model_info=_model_info,
                                _code_info=_tmp_code_info,
                                _tokenized_info=_cur_tokenized_info,
                                _cur_name2index_dict=_tmp_name2index_dict,
                                _logger=_logger,
                            )
                    _target_old_name = _name_score_list[0]['name']
                    _target_sub_list = _valid_subs_dict[_target_old_name]
                    _candi_code_info_list = []
                    for _target_sub in _target_sub_list:
                        _tmp_new_code = get_example(
                            code=_old_beam_info['code_str'],
                            tgt_word=_target_old_name,
                            substitute=_target_sub,
                            lang=_old_beam_info['lang']
                        )
                        _tmp_code_info = copy.deepcopy(_code_info)
                        _tmp_code_info['code'] = _tmp_new_code
                        _candi_code_info_list.append(_tmp_code_info)
                    _candi_dataset = UtilsFeature.get_code_dataset(
                        _args=_args,
                        _code_info_list=_candi_code_info_list,
                        _tokenizer=_model_info['tokenizer'],
                        _logger=_logger
                    )
                    _canid_logit_list, _candi_label_pred_list = _model_info['model'].get_dataset_result(
                        _dataset=_candi_dataset,
                        _batch_size=_args.model_eval_batch_size
                    )
                    _success_candi_index = [
                        _idx for _idx, _pred_label in enumerate(_candi_label_pred_list) if _pred_label != _code_info['label']
                    ]
                    if len(_success_candi_index) > 0:
                        _success_idx = _success_candi_index[0]
                        _old_beam_info['replace_info'].append({
                            'old_name': _target_old_name,
                            'new_name': _target_sub_list[_success_idx],
                            'changed_pos_count': len(_name2index_dict[_target_old_name])
                        })
                        _res_info_list.append({
                            'status': _status_map['success'],
                            'replace_info': _old_beam_info['replace_info'],
                            'new_code_str': _candi_code_info_list[_success_idx]['code']
                        })
                        _need_continue = False
                        break
                    else:
                        _candi_prob_list = [_logit_list[_code_info['label']] for _logit_list in _canid_logit_list]
                        # sort prob index list, keep minimum beam_size item
                        _prob_index_order = np.argsort(_candi_prob_list)
                        _min_prob_index_list = _prob_index_order[:_args.search_beam_size]
                        for _item_index in _min_prob_index_list:
                            _tmp_new_beam_info = copy.deepcopy(_old_beam_info)
                            _tmp_new_beam_info['replace_info'].append({
                                'old_name': _target_old_name,
                                'new_name': _target_sub_list[_item_index],
                                'changed_pos_count': len(_name2index_dict[_target_old_name])
                            })
                            _cur_beam_info_list.append({
                                'pos': _valid_pos_list[1:],
                                'subs': {k: v for k, v in _valid_subs_dict.items() if k != _target_old_name},
                                'code_str': _candi_code_info_list[_item_index]['code'],
                                'lang': _tmp_new_beam_info['lang'],
                                'replace_info': _tmp_new_beam_info['replace_info'],
                                'beam_prob': _candi_prob_list[_item_index]
                            })
                if len(_cur_beam_info_list) == 0:
                    _need_continue = False
                if not _need_continue:
                    break
                #  keep top k beam in _cur_beam_info_list based on prob
                # sort by prob, ascending
                _cur_beam_info_list = sorted(_cur_beam_info_list, key=lambda x: x['beam_prob'])
                _beam_info_list = _cur_beam_info_list[:_args.search_beam_size]
                # _logger.info(f'beam info len: {len(_beam_info_list)}')
                _prob_str = ' '.join(map(lambda x: str(x['beam_prob']), _beam_info_list))
                # _logger.inbeam info prob list:fo(f'beam info prob list: {_prob_str}')
            if len(_res_info_list) == 0:
                raise ValueError('attack fail')
            # return last res info
            return _res_info_list[-1]
    @staticmethod
    def launch_attack_codeattack(_args, _model_info, _code_info, _attack_info, _logger):
        # greedy search with constraint limit
        # 1. choose variable to update
        # 2. check sim > threshold (sim before and after should be bounded) & perturbation < threshold constraint (theta limit changed token count or percentage)
        _cur_tokenized_info = UtilsTokenizer.get_tokenized_info(
            _args=_args,
            _code_str=_attack_info['code_str'],
            _tokenizer=_model_info['tokenizer'],
            _logger=_logger
        )
        _max_changed_count = int(len(_cur_tokenized_info['word_list']) * _args.codeattack_perturbation_threshold)
        _min_sim_threshold = _args.codeattack_sim_threshold
        # similar to greedy, but calculate sim > threshold and limit total changed count < threshold
        _cur_variable_name_list = list(_attack_info['subs'].keys())
        # filter by tokenized info word list
        _cur_variable_name_list = [_x for _x in _cur_variable_name_list if _x in _cur_tokenized_info['word_list']]
        # directly return if variable empty
        if len(_cur_variable_name_list) == 0:
            return {
                'status': _status_map['fail'],
                'replace_info': _attack_info['replace_info'],
                'new_code_str': _attack_info['code_str']
            }
        _name2index_dict = UtilsTokenizer.get_identifier_position_name2index(
            _token_list=_cur_tokenized_info['word_list'],
            _name_list=_cur_variable_name_list
        )
        if _args.position_type in ['dynamic_mask', 'dynamic_replacement']:
            raise NotImplementedError
        elif _args.position_type in ['predefined_mask', 'predefined_replacement', 'random']:
            _need_continue = True
            _res_info = None
            while _need_continue:
                # use predefined pos + subs
                _pos_list = _attack_info['pos']
                _subs_dict = _attack_info['subs']
                # join three variable name list
                _new_variable_name_list = []
                _pos_name_list = [x['name'] for x in _pos_list]
                for _name in _cur_variable_name_list:
                    # only exist in pos and dict, append
                    if _name in _pos_name_list and _name in _subs_dict and len(_subs_dict[_name]) > 0:
                        _new_variable_name_list.append(_name)
                _cur_variable_name_list = _new_variable_name_list
                # filter out _valid_pos_list, _valid_subs_dict
                _valid_pos_list = [_pos for _pos in _pos_list if _pos['name'] in _cur_variable_name_list]
                _valid_subs_dict = {k: v for k, v in _subs_dict.items() if len(v) > 0 and k in _cur_variable_name_list}
                # direct return if no pos left
                if len(_valid_pos_list) == 0 or len(_valid_subs_dict.keys()) == 0:
                    _need_continue = False
                    _res_info = {
                        'status': _status_map['fail'],
                        'replace_info': _attack_info['replace_info'],
                        'new_code_str': _attack_info['code_str']
                    }
                    break
                # check attack info replace info changed_pos_count count
                _replaced_sum = sum(map(lambda x: x['changed_pos_count'], _attack_info['replace_info']))
                if _replaced_sum >= _max_changed_count:
                    _need_continue = False
                    _res_info = {
                        'status': _status_map['fail'],
                        'replace_info': _attack_info['replace_info'],
                        'new_code_str': _attack_info['code_str']
                    }
                    break
                # loop in order to find name+subs under both constraints
                _temp_candi_info_list = []
                for _pos in _valid_pos_list:
                    # skip if target changed count exceed limit
                    _tmp_changed_count = len(_name2index_dict[_pos['name']])
                    _tmp_new_changed_count = _replaced_sum + _tmp_changed_count
                    if _tmp_new_changed_count > _max_changed_count:
                        continue
                    for _subs in _valid_subs_dict[_pos['name']]:
                        # check constraint, once meet, exit current iteration replacement
                        _temp_new_code = get_example(
                            code=_attack_info['code_str'],
                            tgt_word=_pos['name'],
                            substitute=_subs,
                            lang=_attack_info['lang']
                        )
                        _tmp_code_info = copy.deepcopy(_code_info)
                        _tmp_code_info['code'] = _temp_new_code
                        _tmp_dataset = UtilsFeature.get_code_dataset(
                            _args=_args,
                            _code_info_list=[_tmp_code_info],
                            _tokenizer=_model_info['tokenizer'],
                            _logger=_logger
                        )
                        # calc cosine sim between attack_info code_feature and temp_feature
                        _tmp_sim = cosine_similarity(
                            _attack_info['code_feature'][0].reshape(1, -1),
                            _tmp_dataset[0][0].reshape(1, -1)
                        ).item()
                        # skip if sim below threshold
                        if _tmp_sim < _min_sim_threshold:
                            continue
                        # append valid candi, rank by success or prob change later
                        _temp_candi_info_list.append({
                            'pos': _pos,
                            'subs': _subs,
                            'code_info': _tmp_code_info,
                            'feature': _tmp_dataset._feature_list[0],
                            'sim': _tmp_sim,
                        })
                # direct return if _temp_candi_info_list empty
                if len(_temp_candi_info_list) == 0:
                    _need_continue = False
                    _res_info = {
                        'status': _status_map['fail'],
                        'replace_info': _attack_info['replace_info'],
                        'new_code_str': _attack_info['code_str']
                    }
                    break
                # _temp_candi_dataset directly from feature list
                _temp_candi_dataset = InputDataset.get_dataset(_args, [
                    _item['feature'] for _item in _temp_candi_info_list
                ])
                # evaluate the whole dataset
                _candi_logit_list, _candi_label_pred_list = _model_info['model'].get_dataset_result(
                    _dataset=_temp_candi_dataset,
                    _batch_size=_args.model_eval_batch_size
                )
                _success_candi_index = [
                    _idx for _idx, _pred_label in enumerate(_candi_label_pred_list) if _pred_label != _code_info['label']
                ]
                if len(_success_candi_index) > 0:
                    _success_idx = _success_candi_index[0]
                    _attack_info['replace_info'].append({
                        'old_name': _temp_candi_info_list[_success_idx]['pos']['name'],
                        'new_name': _temp_candi_info_list[_success_idx]['subs'],
                        'changed_pos_count': len(_name2index_dict[_temp_candi_info_list[_success_idx]['pos']['name']])
                    })
                    _need_continue = False
                    _res_info = {
                        'status': _status_map['success'],
                        'replace_info': _attack_info['replace_info'],
                        'new_code_str': _temp_candi_info_list[_success_idx]['code_info']['code']
                    }
                    break
                else:
                    # find max prob change
                    _candi_prob_list = [_logit_list[_code_info['label']] for _logit_list in _candi_logit_list]
                    _min_prob_index = np.argmin(_candi_prob_list)
                    _attack_info['replace_info'].append({
                        'old_name': _temp_candi_info_list[_min_prob_index]['pos']['name'],
                        'new_name': _temp_candi_info_list[_min_prob_index]['subs'],
                        'changed_pos_count': len(_name2index_dict[_temp_candi_info_list[_min_prob_index]['pos']['name']])
                    })
                    _attack_info['pos'] = [x for x in _attack_info['pos'] if x['name'] != _temp_candi_info_list[_min_prob_index]['pos']['name']]
                    _attack_info['subs'] = {k: v for k, v in _attack_info['subs'].items() if k != _temp_candi_info_list[_min_prob_index]['pos']['name']}
                    _attack_info['code_str'] = _temp_candi_info_list[_min_prob_index]['code_info']['code']
                    if _args.is_debug:
                        _logger.info(f'candi old name: {_temp_candi_info_list[_min_prob_index]["pos"]["name"]}, new name: {_temp_candi_info_list[_min_prob_index]["subs"]}, old prob: {_attack_info["prob"]}, new prob: {_candi_prob_list[_min_prob_index]}')
            if _res_info is not None:
                return _res_info
            else:
                raise ValueError('attack fail')
    @staticmethod
    def launch_attack_random(_args, _model_info, _code_info, _attack_info, _logger):
        # totally random, use random pos + random + sub
        _cur_tokenized_info = UtilsTokenizer.get_tokenized_info(
            _args=_args,
            _code_str=_attack_info['code_str'],
            _tokenizer=_model_info['tokenizer'],
            _logger=_logger
        )
        _cur_variable_name_list = list(_attack_info['subs'].keys())
        _cur_variable_name_list = [_x for _x in _cur_variable_name_list if _x in _cur_tokenized_info['word_list']]
        if len(_cur_variable_name_list) == 0:
            return {
                'status': _status_map['fail'],
                'replace_info': _attack_info['replace_info'],
                'new_code_str': _attack_info['code_str']
            }
        _need_continue = True
        _res_info = None
        while _need_continue:
            _filtered_info = UtilsAttacker.filter_out_pos_subs_name(_args, _attack_info, _cur_variable_name_list)
            _valid_pos_list = _filtered_info['pos']
            _valid_subs_dict = _filtered_info['subs']
            _cur_variable_name_list = _filtered_info['name']
            if len(_valid_pos_list) == 0 or len(_valid_subs_dict.keys()) == 0:
                _need_continue = False
                _res_info = {
                    'status': _status_map['fail'],
                    'replace_info': _attack_info['replace_info'],
                    'new_code_str': _attack_info['code_str']
                }
                break
            _name2index_dict = UtilsTokenizer.get_identifier_position_name2index(
                _token_list=_cur_tokenized_info['word_list'],
                _name_list=_cur_variable_name_list
            )
            # random choose from pos 
            _pos_index = random.randint(0, len(_valid_pos_list) - 1)
            _target_pos = _valid_pos_list[_pos_index]
            _old_name = _target_pos['name']
            _target_subs = _valid_subs_dict[_target_pos['name']]
            _subs_index = random.randint(0, len(_target_subs) - 1)
            _target_name = _target_subs[_subs_index]
            _new_code_str = get_example(
                code=_attack_info['code_str'],
                tgt_word=_old_name,
                substitute=_target_name,
                lang=_attack_info['lang']
            )
            _new_code_info = copy.deepcopy(_code_info)
            _new_code_info['code'] = _new_code_str
            _new_dataset = UtilsFeature.get_code_dataset(
                _args=_args,
                _code_info_list=[_new_code_info],
                _tokenizer=_model_info['tokenizer'],
                _logger=_logger
            )
            _new_logit_list, _new_label_pred_list = _model_info['model'].get_dataset_result(
                _dataset=_new_dataset,
                _batch_size=_args.model_eval_batch_size
            )
            if _new_label_pred_list[0] != _code_info['label']:
                _attack_info['replace_info'].append({
                    'old_name': _old_name,
                    'new_name': _target_name,
                    'changed_pos_count': len(_name2index_dict[_old_name])
                })
                _need_continue = False
                _res_info = {
                    'status': _status_map['success'],
                    'replace_info': _attack_info['replace_info'],
                    'new_code_str': _new_code_str
                }
                break
            else:
                _attack_info['replace_info'].append({
                    'old_name': _old_name,
                    'new_name': _target_name,
                    'changed_pos_count': len(_name2index_dict[_old_name])
                })
                _attack_info['pos'] = [x for x in _attack_info['pos'] if x['name'] != _old_name]
                _attack_info['subs'] = {k: v for k, v in _attack_info['subs'].items() if k != _old_name}
                _attack_info['code_str'] = _new_code_str
                if _args.is_debug:
                    _logger.info(f'candi old name: {_old_name}, new name: {_target_name}, old prob: {_attack_info["prob"]}, new prob: {_new_logit_list[0][_code_info["label"]]}')
        if _res_info is not None:
            return _res_info
        else:
            raise ValueError('attack fail')
    @staticmethod
    def launch_attack_ga(_args, _model_info, _code_info, _attack_info, _logger, _cross_prob=0.7):
        _cur_tokenized_info = UtilsTokenizer.get_tokenized_info(
            _args=_args,
            _code_str=_attack_info['code_str'],
            _tokenizer=_model_info['tokenizer'],
            _logger=_logger
        )
        _cur_variable_name_list = list(_attack_info['subs'].keys())
        _cur_variable_name_list = [_x for _x in _cur_variable_name_list if _x in _cur_tokenized_info['word_list']]
        if len(_cur_variable_name_list) == 0:
            return {
                'status': _status_map['fail'],
                'replace_info': _attack_info['replace_info'],
                'new_code_str': _attack_info['code_str']
            }
        _filtered_info = UtilsAttacker.filter_out_pos_subs_name(_args, _attack_info, _cur_variable_name_list)
        _valid_subs_dict = _filtered_info['subs']
        _cur_variable_name_list = _filtered_info['name']
        _name2index_dict = UtilsTokenizer.get_identifier_position_name2index(
            _token_list=_cur_tokenized_info['word_list'],
            _name_list=_cur_variable_name_list
        )
        _cur_fitness_value_list = []
        _cur_chrome_dict = {_w: _w for _w in _cur_variable_name_list}
        _cur_population_list = [_cur_chrome_dict]
        for _name in _cur_variable_name_list:
            _candi_feature_list = []
            _candi_name_list = []
            _candi_code_list = []
            _max_prob_gap = 0
            _cur_subs_name = _name
            for _candi_name in _valid_subs_dict[_name]:
                _candi_name_list.append(_candi_name)
                _candi_code_str = get_example(
                    code=_attack_info['code_str'],
                    tgt_word=_name,
                    substitute=_candi_name,
                    lang=_attack_info['lang']
                )
                _candi_code_info = copy.deepcopy(_code_info)
                _candi_code_info['code'] = _candi_code_str
                _candi_feature = UtilsFeature.get_code_feature(
                    _args=_args,
                    _cur_code_str=_candi_code_str,
                    _code_info=_candi_code_info,
                    _tokenizer=_model_info['tokenizer'],
                    _logger=_logger
                )
                _candi_feature_list.append(_candi_feature)
                _candi_code_list.append(_candi_code_info)
            if len(_candi_name_list) == 0:
                continue        
            _candi_dataset = InputDataset.get_dataset(
                _args=_args,
                _feature_list=_candi_feature_list
            )
            _candi_logit_list, _candi_label_pred_list = _model_info['model'].get_dataset_result(
                _dataset=_candi_dataset,
                _batch_size=_args.model_eval_batch_size
            )
            _max_prob_gap_index = -1
            for _prob_index, _prob_v in enumerate(_candi_logit_list):
                _label_pred = _candi_label_pred_list[_prob_index]
                # check label if changed, direct return
                if _label_pred != _code_info['label']:
                    _attack_info['replace_info'].append({
                        'old_name': _name,
                        'new_name': _candi_name_list[_prob_index],
                        'changed_pos_count': len(_name2index_dict[_name])
                    })
                    return {
                        'status': _status_map['success'],
                        'replace_info': _attack_info['replace_info'],
                        'new_code_str': _candi_code_list[_prob_index]['code']
                    }
                # calc prob gap
                _prob_gap = _attack_info['prob'] - _prob_v[_code_info['label']]
                if _prob_gap > _max_prob_gap:
                    _max_prob_gap = _prob_gap
                    _max_prob_gap_index = _prob_index
            if _max_prob_gap_index != -1:
                # need change
                _cur_subs_name = _candi_name_list[_max_prob_gap_index]
            _new_chrome_dict = copy.deepcopy(_cur_chrome_dict)
            _new_chrome_dict[_name] = _cur_subs_name
            _cur_population_list.append(_new_chrome_dict)
            _fitness_info = UtilsAttackerGA.compute_chrome_fitness(
                _args=_args,
                _model_info=_model_info,
                _code_info=_code_info,
                _attack_info=_attack_info,
                _chrome_dict=_new_chrome_dict,
                _logger=_logger
            )
            _cur_fitness_value_list.append(_fitness_info['fitness_value'])
        # define iteration and mutate

        _max_iter_count = max(5 * len(_cur_population_list), 10)
        for _iter_idx in range(_max_iter_count):
            _mutant_list = []
            # iter in batch size
            for _batch_idx in range(_args.model_eval_batch_size):
                # check _valid_subs_dict empty
                if len(_valid_subs_dict) == 0:
                    break
                _random_v = random.random()
                # random select parent chromesome
                (_chromesome_1, _idx_1), (_chromesome_2, _idx_2) = UtilsAttackerGA.random_select_parent(_cur_population_list)
                # crossover
                if _random_v < _cross_prob:
                    if _idx_1 == _idx_2:
                        _child_1  = UtilsAttackerGA.random_mutate(_chromesome_1, _valid_subs_dict)
                        # skip crossover
                        continue
                    else:
                        _child_1, _child_2 = UtilsAttackerGA.random_crossover(_chromesome_1, _chromesome_2)
                        # avoid no change
                        if _child_1 == _chromesome_1 or _child_1 == _chromesome_2:
                            _child_1 = UtilsAttackerGA.random_mutate(_child_1, _valid_subs_dict)
                else:
                    # mutate
                    _child_1 = UtilsAttackerGA.random_mutate(_chromesome_1, _valid_subs_dict)
                _mutant_list.append(_child_1)
            if len(_mutant_list) == 0:
                break
            _batch_code_info_list = []
            for _mutatant_child in _mutant_list:
                _mutatant_code = get_example_batch(
                    _attack_info['code_str'],
                    _mutatant_child,
                    _attack_info['lang']
                )
                _mutatant_code_info = copy.deepcopy(_code_info)
                _mutatant_code_info['code'] = _mutatant_code
                _batch_code_info_list.append(_mutatant_code_info)
            _batch_dataset = UtilsFeature.get_code_dataset(
                _args=_args,
                _code_info_list=_batch_code_info_list,
                _tokenizer=_model_info['tokenizer'],
                _logger=_logger
            )
            _batch_logit_list, _batch_label_pred_list = _model_info['model'].get_dataset_result(
                _dataset=_batch_dataset,
                _batch_size=_args.model_eval_batch_size
            )
            _batch_fitness_value_list = []
            for _item_idx, _batch_logits in enumerate(_batch_logit_list):
                if _batch_label_pred_list[_item_idx] != _code_info['label']:
                    _success_code = _batch_code_info_list[_item_idx]['code']
                    # calc changed var, replaced count
                    for _old_name in _mutant_list[_item_idx].keys():
                        if _old_name != _mutant_list[_item_idx][_old_name]:
                            _attack_info['replace_info'].append({
                                'old_name': _old_name,
                                'new_name': _mutant_list[_item_idx][_old_name],
                                'changed_pos_count': len(_name2index_dict[_old_name])
                            })
                    return {
                        'status': _status_map['success'],
                        'replace_info': _attack_info['replace_info'],
                        'new_code_str': _success_code
                    }
                else:
                    # calc prob fitness
                    _fitness_value = _attack_info['prob'] - _batch_logits[_code_info['label']]
                    _batch_fitness_value_list.append(_fitness_value)
            for _item_idx, _fitness_v in enumerate(_batch_fitness_value_list):
                # replace based on fitness value
                _cur_min_fitness_v = min(_cur_fitness_value_list)
                if _fitness_v > _cur_min_fitness_v:
                    # replace
                    _min_idx = _cur_fitness_value_list.index(_cur_min_fitness_v)
                    _cur_population_list[_min_idx] = _mutant_list[_item_idx]
                    _cur_fitness_value_list[_min_idx] = _fitness_v
        # return fail if no success
        return {
            'status': _status_map['fail'],
            'replace_info': _attack_info['replace_info'],
            'new_code_str': _attack_info['code_str']
        }
    @staticmethod
    def launch_attack_pwvs(_args, _model_info, _code_info, _attack_info, _logger):
        ''' variable_names_.append(tgt_word)
            softmax_sum += np.exp(name_prob_dict[tgt_word])

        for tgt_word in variable_names_:
            [tgt_word] = np.exp(name_prob_dict[tgt_word]) / softmax_sum * candidates_delta[tgt_word]'''
        # calc word saliency sorted list to guide replace
        # search type greedy or beam
        # calc score for each original name based on its substitutes' prob change
        # keep max prob change for each original name, sort them and replace iteratively
        # greedy = predefined_mask + token replacement
        raise Exception('no need to run this attack')
        if _args.search_type not in ['greedy', 'beam']:
            raise ValueError('search type not supported')
        if _args.search_type == 'greedy':
            _cur_tokenized_info = UtilsTokenizer.get_tokenized_info(
                _args=_args,
                _code_str=_attack_info['code_str'],
                _tokenizer=_model_info['tokenizer'],
                _logger=_logger
            )
            _cur_var_name_list = list(_attack_info['subs'].keys())
            _cur_var_name_list = [_x for _x in _cur_var_name_list if _x in _cur_tokenized_info['word_list']]
            if len(_cur_var_name_list) == 0:
                return {
                    'status': _status_map['fail'],
                    'replace_info': _attack_info['replace_info'],
                    'new_code_str': _attack_info['code_str']
                }
            _name2index_dict = UtilsTokenizer.get_identifier_position_name2index(
                _token_list=_cur_tokenized_info['word_list'],
                _name_list=_cur_var_name_list
            )
            _filtered_info = UtilsAttacker.filter_out_pos_subs_name(_args, _attack_info, _cur_var_name_list)
            # list: name, score
            _valid_pos_list = _filtered_info['pos']
            # key: name -> value: name list
            _valid_subs_dict = _filtered_info['subs']
            print(_valid_pos_list)
            print(_valid_subs_dict)
            for _name in _cur_var_name_list:
                pass
            
            pass
        raise NotImplementedError
        pass
    @staticmethod
    def launch_attack_twostep(_args, _model_info, _code_info, _attack_info, _logger):
        # use greedy attack first, then apply ga/beam attack
        _final_attack_res = None
        _attack_idx = 0
        while _attack_idx < len(_args.attack_type_list) and (_final_attack_res == None or _final_attack_res['status'] != _status_map['success']):
            _final_attack_res = UtilsAttacker.launch_sub_single_attack(
                _args=_args,
                _attack_type=_args.attack_type_list[_attack_idx],
                _model_info=_model_info,
                _code_info=_code_info,
                _attack_info=_attack_info,
                _logger=_logger   
            )
            _attack_idx += 1
        return _final_attack_res
    @staticmethod
    def launch_sub_single_attack(_args, _attack_type, _model_info, _code_info, _attack_info, _logger):
        # only support 'greedy', 'beam', 'mhm', 'ga', 'codeattack'
        if _attack_type not in ['greedy', 'beam', 'mhm', 'ga', 'codeattack']:
            raise ValueError('attack type not supported')
        if _attack_type == 'greedy':
            return UtilsAttacker.launch_attack_greedy(
                _args=_args,
                _model_info=_model_info,
                _code_info=_code_info,
                _attack_info=_attack_info,
                _logger=_logger
            )
        elif _attack_type == 'beam':
            _tmp_args = copy.deepcopy(_args)
            _tmp_args.search_beam_size = UtilsMeta._default_search_beam_size
            return UtilsAttacker.launch_attack_beam(
                _args=_tmp_args,
                _model_info=_model_info,
                _code_info=_code_info,
                _attack_info=_attack_info,
                _logger=_logger
            )
        elif _attack_type == 'mhm':
            return UtilsAttacker.launch_attack_mhm(
                _args=_args,
                _model_info=_model_info,
                _code_info=_code_info,
                _attack_info=_attack_info,
                _logger=_logger
            )
        elif _attack_type == 'ga':
            return UtilsAttacker.launch_attack_ga(
                _args=_args,
                _model_info=_model_info,
                _code_info=_code_info,
                _attack_info=_attack_info,
                _logger=_logger
            )
        elif _attack_type == 'codeattack':
            return UtilsAttacker.launch_attack_codeattack(
                _args=_args,
                _model_info=_model_info,
                _code_info=_code_info,
                _attack_info=_attack_info,
                _logger=_logger
            )
        raise NotImplementedError
    @staticmethod
    def launch_attack_mhm(_args, _model_info, _code_info, _attack_info, _logger, _max_iter_n=100, _candi_n=30, _prob_threshold=0.95):
        '''_n_candi=30, _max_iter=100, _prob_threshold=0.95'''
        _cur_tokenized_info = UtilsTokenizer.get_tokenized_info(
            _args=_args,
            _code_str=_code_info['code'],
            _tokenizer=_model_info['tokenizer'],
            _logger=_logger
        )
        _cur_variable_name_list = list(_attack_info['subs'].keys())
        _cur_variable_name_list = [_x for _x in _cur_variable_name_list if _x in _cur_tokenized_info['word_list']]
        if len(_cur_variable_name_list) == 0:
            return {
                'status': _status_map['fail'],
                'replace_info': _attack_info['replace_info'],
                'new_code_str': _attack_info['code_str']
            }
        _name2index_dict = UtilsTokenizer.get_identifier_position_name2index(
            _token_list=_cur_tokenized_info['word_list'],
            _name_list=_cur_variable_name_list
        )
        _filtered_info = UtilsAttacker.filter_out_pos_subs_name(_args, _attack_info, _cur_variable_name_list)
        _valid_subs_dict = _filtered_info['subs']
        _used_subs_dict = {}
        for _iter_idx in range(1, 1 + _max_iter_n):
            _tmp_iter_res = UtilsAttackerMHM.replace_uid(
                _args=_args,
                _attack_info=_attack_info,
                _code_info=_code_info,
                _model_info=_model_info,
                _name2index_dict=_name2index_dict,
                _subs_dict=_valid_subs_dict,
                _candi_n=_candi_n,
                _prob_threshold=_prob_threshold,
                _logger=_logger
            )
            # check whether direct continue skip
            if _tmp_iter_res['mhm_old_name'] == None or _tmp_iter_res['mhm_new_name'] == None:
               continue
            if _tmp_iter_res['mhm_status'] in [UtilsAttackerMHM._mhm_status_map['success'], UtilsAttackerMHM._mhm_status_map['accept']]:
                if _iter_idx == 1:
                    _used_subs_dict[_tmp_iter_res['mhm_old_name']] = [_tmp_iter_res['mhm_new_name']]
                _flag_v = 0
                for _old_name in _used_subs_dict.keys():
                    if _tmp_iter_res['mhm_old_name'] == _used_subs_dict[_old_name][-1]:
                        _flag_v = 1
                        _used_subs_dict[_old_name].append(_tmp_iter_res['mhm_new_name'])
                        break
                if _flag_v == 0:
                    _used_subs_dict[_tmp_iter_res['mhm_old_name']] = [_tmp_iter_res['mhm_new_name']]
                _attack_info['code_str'] = _tmp_iter_res['new_code_str']
                # update name2index, name2subs
                _name2index_dict[_tmp_iter_res['mhm_new_name']] = _name2index_dict.pop(_tmp_iter_res['mhm_old_name'])
                _valid_subs_dict[_tmp_iter_res['mhm_new_name']] = _valid_subs_dict.pop(_tmp_iter_res['mhm_old_name'])
                # iterate in word list
                for _word_idx in range(len(_cur_tokenized_info['word_list'])):
                    # replace
                    if _cur_tokenized_info['word_list'][_word_idx] == _tmp_iter_res['mhm_old_name']:
                        _cur_tokenized_info['word_list'][_word_idx] = _tmp_iter_res['mhm_new_name']
                    # if already success, just return
                    if _tmp_iter_res['mhm_status'] == UtilsAttackerMHM._mhm_status_map['success']:
                        # update replace info, try find existing by old_name&new_name, append if not exist
                        _target_replace_info = [_info for _info in _attack_info['replace_info'] if _info['old_name'] == _tmp_iter_res['mhm_old_name'] and _info['new_name'] == _tmp_iter_res['mhm_new_name']]
                        if len(_target_replace_info) == 0:
                            _attack_info['replace_info'].append({
                                'old_name': _tmp_iter_res['mhm_old_name'],
                                'new_name': _tmp_iter_res['mhm_new_name'],
                                'changed_pos_count': _tmp_iter_res['mhm_changed_pos_count']
                            })
                        else:
                            _target_replace_info[0]['changed_pos_count'] += _tmp_iter_res['mhm_changed_pos_count']
                        return {
                            'status': _status_map['success'],
                            'replace_info': _attack_info['replace_info'],
                            'new_code_str': _tmp_iter_res['new_code_str']
                        }
                    pass
        # extract code infor from _used_subs_dict
        for _old_name in _used_subs_dict.keys():
            # try to match from existing replace_info
            _target_replace_info = [_info for _info in _attack_info['replace_info'] if _info['old_name'] == _old_name and _info['new_name'] == _used_subs_dict[_old_name][-1]]
            if len(_target_replace_info) == 0:
                _attack_info['replace_info'].append({
                    'old_name': _old_name,
                    'new_name': _used_subs_dict[_old_name][-1],
                    'changed_pos_count': _tmp_iter_res['mhm_changed_pos_count']
                })
            else:
               _target_replace_info[0]['changed_pos_count'] += _tmp_iter_res['mhm_changed_pos_count']
        # return failed with info finally
        return {
            'status': _status_map['fail'],
            'replace_info': _attack_info['replace_info'],
            'new_code_str': _tmp_iter_res['new_code_str']
        }
    @staticmethod
    def launch_attack(_args, _model_info, _code_info, _pos_line, _subs_line, _logger):
        # check ground truth label and predict label, directly return ground truth label != predict label 
        _truth_dataset = UtilsFeature.get_code_dataset(
            _args=_args,
            _code_info_list=[_code_info],
            _tokenizer=_model_info['tokenizer'],
            _logger=_logger
        )
        # eval first
        _truth_logit_list, _truth_label_pred_list = _model_info['model'].get_dataset_result(
            _dataset = _truth_dataset,
            _batch_size=_args.model_eval_batch_size
        )
        # already plain list _truth_logit_list
        # _truth_logit_list = _truth_logit_list.tolist()
        # log truth label, pred label
        if _truth_label_pred_list[0] != _code_info['label']:
            return {
                'status': _status_map['truth label mismatch'],
            }
        _pos_list = []
        if _pos_line is not None:
            _pos_list = json.loads(_pos_line)
        _subs_dict = {}
        if _subs_line is not None:
            _subs_dict = json.loads(_subs_line)
        # attack info has code_str, while code info as code key
        _cur_attack_info = {
            'code_str': _code_info['code'],
            'code_feature': _truth_dataset[0],
            'prob': max(_truth_logit_list[0]),
            'pos': _pos_list,
            'subs': _subs_dict,
            'lang': _task_lang_map[_args.task_type],
            'replace_info': []
        }
        # prepare attack info: code_str, prob, pos, subs
        # return attack res with format: status, replace info list (each item is [old_name, new_name, changed_count], new_code_info: new_code_str (after all replacement) + new label after replacement
        _attack_res = {}
        _start_time = time.time()
        # reset query
        _model_info['model'].reset_query_count()
        if _args.attack_type == 'greedy':
            # _attacker = ga
            _attack_res = UtilsAttacker.launch_attack_greedy(
                _args=_args,
                _model_info=_model_info,
                _code_info=_code_info,
                _attack_info=_cur_attack_info,
                _logger=_logger
            )
        elif _args.attack_type == 'beam':
            _attack_res = UtilsAttacker.launch_attack_beam(
                _args=_args,
                _model_info=_model_info,
                _code_info=_code_info,
                _attack_info=_cur_attack_info,
                _logger=_logger
            )
        elif _args.attack_type == 'codeattack':
            _attack_res = UtilsAttacker.launch_attack_codeattack(
                _args=_args,
                _model_info=_model_info,
                _code_info=_code_info,
                _attack_info=_cur_attack_info,
                _logger=_logger
            )
        elif _args.attack_type == 'ga':
            _attack_res = UtilsAttacker.launch_attack_ga(
                _args=_args,
                _model_info=_model_info,
                _code_info=_code_info,
                _attack_info=_cur_attack_info,
                _logger=_logger
            )
        elif _args.attack_type == 'twostep':
            _attack_res = UtilsAttacker.launch_attack_twostep(
                _args=_args,
                _model_info=_model_info,
                _code_info=_code_info,
                _attack_info=_cur_attack_info,
                _logger=_logger
            )
        # elif _args.attack_type == 'pwvs':
        #     _attack_res = UtilsAttacker.launch_attack_pwvs(
        #         _args=_args,
        #         _model_info=_model_info,
        #         _code_info=_code_info,
        #         _attack_info=_cur_attack_info,
        #         _logger=_logger
        #     )
        elif _args.attack_type == 'random':
            _attack_res = UtilsAttacker.launch_attack_random(
                _args=_args,
                _model_info=_model_info,
                _code_info=_code_info,
                _attack_info=_cur_attack_info,
                _logger=_logger
            )
        elif _args.attack_type == 'mhm':
            _attack_res = UtilsAttacker.launch_attack_mhm(
                _args=_args,
                _model_info=_model_info,
                _code_info=_code_info,
                _attack_info=_cur_attack_info,
                _logger=_logger
            )
        _end_time = time.time()
        _attack_res['time'] = _end_time - _start_time
        _attack_res['query_count'] = _model_info['model']._query_count
        # record model query
        return _attack_res
    pass