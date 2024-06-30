import sys, os, json, pickle, logging, argparse, torch, numpy as np, logging, uuid, transformers, copy, shutil
from tqdm import tqdm
from datetime import datetime
from itertools import zip_longest
from torch.utils.data import RandomSampler


from src.model_release.utils_debug import set_seed, UtilsTokenizer, UtilsPositionScore, NpEncoder, UtilsMeta, UtilsFeature, InputDataset, UtilsSubstitutionGeneration, UtilsAttacker, count_file_line, _status_map
from src.model_release.model_debug import VRModel

_global_logger = None
def initlize_logger():
    global _global_logger
    _logger = logging.getLogger()
    if not os.path.exists('logs'):
        os.makedirs('logs')
    _cur_time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    _uuid_str = str(uuid.uuid4())
    _log_file = f'logs/{_cur_time_str}-{_uuid_str}.log'
    print(f'Log file: {_log_file}')
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', filename=_log_file, filemode='w')
    _global_logger = _logger
    pass

def run_initial_attack(_args):
    # python -u src/model/start_debug.py --mode=attack --task_type=authorship_attribution --position_type=predefined_mask --substitution_type=token --search_type=greedy --attack_type=greedy --model_name=CodeBERT --model_name_or_path=microsoft/codebert-base --model_tokenizer_name=roberta-base --model_num_label=66 --model_code_length=512 --model_data_flow_length=128 --model_eval_batch_size=32 --index 0 500 --is_debug=True
    _attak_res_info = UtilsMeta.validate_attack_args(_args)
    # skip if existing_len > target_len
    if _attak_res_info['existing_len'] >= _attak_res_info['target_len']:
        _global_logger.info(f'skip attack: existing_len >= target_len')
        return
    _global_logger.info(f'_attak_res_info: {_attak_res_info}')
    UtilsMeta.prepare_attack_file(_args)
    UtilsMeta.prepare_attack_pos_file(_args)
    UtilsMeta.prepare_attack_subs_file(_args)
    _code_info_list = InputDataset.load_code_list(_args, _global_logger)
    # load target model also
    _class_info = VRModel.get_model_class_list(_args)
    # config, model, tokenizer
    _config = _class_info[0].from_pretrained(_args.model_name_or_path, cache_dir='tg/cache')
    _config.num_labels = _args.model_num_label
    _tokenizer = _class_info[2].from_pretrained(_args.model_tokenizer_name, do_lower_case=False, cache_dir='tg/cache')
    _model = _class_info[1].from_pretrained(_args.model_name_or_path, config=_config, from_tf=False, cache_dir='tg/cache')
    _vr_model = VRModel(_config=_config,
                        _args=_args,
                        _encoder=_model,
                        _tokenizer=_tokenizer,
                        _logger=_global_logger)
    _ptm_path = VRModel.get_model_state_path(_args)
    # load
    _vr_model.load_state_dict(torch.load(_ptm_path, map_location=_args.device))
    _vr_model.to(_args.device)
    _global_logger.info(f'model initlized with {_ptm_path}')
    _model_info = {
        'model': _vr_model,
        'tokenizer': _tokenizer
    }
    # load each file iteratively, tricky, 3 info in same loop, 1 code + 2 read: pos_jsonl_path, subs_jsonl_path
    # if pos_jsonl_path not exist, means dynamic pos, detect whether pos_jsonl_path key in _args
    pos_jsonl_reader = []
    if 'pos_jsonl_path' in vars(_args):
        pos_jsonl_reader = open(_args.pos_jsonl_path)
    # if subs_jsonl_path not exist, raise error
    for _line_idx, _line_info in enumerate(zip_longest(_code_info_list,pos_jsonl_reader, open(_args.subs_jsonl_path))):
        # skip if _line_idx < _attak_res_info['existing_len']
        if _line_idx < _attak_res_info['existing_len']:
            _global_logger.info(f'Skip attack: {_line_idx} of {len(_code_info_list)}')
            continue    
        _code_info, _pos_line, _subs_line = _line_info
        # construct attack here
        _attack_res = UtilsAttacker.launch_attack(
            _args = _args,
            _model_info = _model_info,
            _code_info = _code_info,
            _pos_line = _pos_line,
            _subs_line = _subs_line,
            _logger=_global_logger
        )
        # validate here
        if _attack_res['status'] == _status_map['success']:
            _new_code = _attack_res['new_code_str']
            _new_code_info = copy.deepcopy(_code_info)
            _new_code_info['code'] = _new_code
            _new_code_dataset = UtilsFeature.get_code_dataset(
                _args=_args,
                _code_info_list=[_new_code_info],
                _tokenizer=_tokenizer,
                _logger=_global_logger
            )
            _, _label_pred = _vr_model.get_dataset_result(
                _dataset=_new_code_dataset,
                _batch_size=_args.model_eval_batch_size
            )
            assert len(_label_pred) == 1
            assert _label_pred[0] != _code_info['label']
            if _label_pred[0] == _code_info['label']:
                raise NotImplementedError
        _global_logger.info(f'attack res {_line_idx}: {_attack_res["status"]}')
        # if _args.is_debug:
        #     print(f'attack res: {json.dumps(_attack_res)}')
        #     raise NotImplementedError
        # write append into res file, json format
        open(_attak_res_info['path'], 'a').write(json.dumps(_attack_res, cls=NpEncoder) + '\n')
def run_attack_after_retrain(_args):
    # python -u src/model/start_debug.py --mode=attack_after_retrain --task_type=authorship_attribution --position_type=predefined_mask --substitution_type=token --search_type=greedy --attack_type=greedy --model_name=CodeBERT --model_name_or_path=microsoft/codebert-base --model_tokenizer_name=roberta-base --model_num_label=66 --model_code_length=512 --model_data_flow_length=128 --model_eval_batch_size=32 --index 0 500 --is_debug=True
    # atk after retrain, differ original atk with new model bin file and part of atk target data
    # currently the second atk method should be the same as the first one
    _reatk_res_info = UtilsMeta.validate_reattack_args(_args)
    _global_logger.info(f'_reatk_res_info: {_reatk_res_info}')
    UtilsMeta.prepare_attack_file(_args)
    UtilsMeta.prepare_attack_pos_file(_args)
    UtilsMeta.prepare_attack_subs_file(_args)
    _code_info_list = InputDataset.load_code_list(_args, _global_logger)
    # direct return if res file exists and len >= code_info_list
    if _reatk_res_info['existing_len'] >= len(_code_info_list):
        _global_logger.info(f'res file exists and len >= code_info_list')
        return
    # load model
    _class_info = VRModel.get_model_class_list(_args)
    _config = _class_info[0].from_pretrained(_args.model_name_or_path, cache_dir='tg/cache')
    _config.num_labels = _args.model_num_label
    _tokenizer = _class_info[2].from_pretrained(_args.model_tokenizer_name, do_lower_case=False, cache_dir='tg/cache')
    _model = _class_info[1].from_pretrained(_args.model_name_or_path, config=_config, from_tf=False, cache_dir='tg/cache')
    _reatk_vr_model = VRModel(
        _config=_config,
        _args=_args,
        _encoder=_model,
        _tokenizer=_tokenizer,
        _logger=_global_logger
    )
    _reatk_ptm_path = VRModel.get_reatk_model_state_path(_args, _reatk_res_info)
    _reatk_vr_model.load_state_dict(torch.load(_reatk_ptm_path, map_location=_args.device))
    _reatk_vr_model.to(_args.device)
    _global_logger.info(f'load reatk model: {_reatk_ptm_path}')
    _model_info = {
        'model': _reatk_vr_model,
        'tokenizer': _tokenizer
    }
    # load test index from args.reattack_index_path
    _to_atk_index_list = json.loads(open(_args.reattack_index_path, 'r').read())
    _pos_jsonl_reader = []
    if 'pos_jsonl_path' in vars(_args):
        _pos_jsonl_reader = open(_args.pos_jsonl_path)
    for _line_idx, _line_info in enumerate(
        zip_longest(_code_info_list, _pos_jsonl_reader, open(_args.subs_jsonl_path))):
        if _line_idx < _reatk_res_info['existing_len']:
            _global_logger.info(f'Skip existing reattack: {_line_idx} of {len(_code_info_list)}')
            continue
        # if idx+args.index 0 not in _to_atk_index_list, skip
        if _line_idx + _args.index[0] not in _to_atk_index_list:
            # write empty dict
            open(_reatk_res_info['path'], 'a').write(json.dumps({}) + '\n')
            _global_logger.info(f'Skip non-test reattack: {_line_idx} of {len(_code_info_list)}')
            continue
        _code_info, _pos_line, _subs_line = _line_info
        _reatk_res = UtilsAttacker.launch_attack(
            _args = _args,
            _model_info = _model_info,
            _code_info = _code_info,
            _pos_line = _pos_line,
            _subs_line = _subs_line,
            _logger=_global_logger
        )
        if _reatk_res['status'] == _status_map['success']:
            # validate again here
            _new_code = _reatk_res['new_code_str']
            _new_code_info = copy.deepcopy(_code_info)
            _new_code_info['code'] = _new_code
            _new_code_dataset = UtilsFeature.get_code_dataset(
                _args=_args,
                _code_info_list=[_new_code_info],
                _tokenizer=_tokenizer,
                _logger=_global_logger
            )
            _, _label_pred = _reatk_vr_model.get_dataset_result(
                _dataset=_new_code_dataset,
                _batch_size=_args.model_eval_batch_size
            )
            assert len(_label_pred) == 1
            assert _label_pred[0] != _code_info['label']
            if _label_pred[0] == _code_info['label']:
                raise NotImplementedError
        _global_logger.info(f'reattack res {_line_idx}: {_reatk_res["status"]}')
        if _args.is_debug:
            raise NotImplementedError
        open(_reatk_res_info['path'], 'a').write(json.dumps(_reatk_res, cls=NpEncoder) + '\n')

def run_transfer_atk(_args, _after_retrain=False, _exclude_self=True):
    # python -u src/model/start_debug.py --mode=transfer
    # evaluate across different model after their own adv retrain
    # python -u src/model/start_debug.py --mode=transfer_after_retrain
    # just after initial attack, evalute the initial attack across different model
    # attack examples: success adversarial examples from different attack methods
    # attack target: original trained model
    _model_list = UtilsMeta.get_model_list()
    _task_list = UtilsMeta.get_task_list()
    # res under tg/datasets/atk_transfer
    _res_dir = f'tg/datasets/atk_transfer'
    if _after_retrain:
        _res_dir = f'tg/datasets/atk_transfer_after_retrain'
    if not os.path.exists(_res_dir):
        os.makedirs(_res_dir, exist_ok=True)
    # multiply model_eval_batch_size by gpu size / 16
    _gpu_gb = UtilsMeta.get_gpu_gb()
    _gpu_times = _gpu_gb / 16
    # save _res_info into res_dir as atk_transfer_res.json
    _res_info_path = f'{_res_dir}/atk_transfer_res.json'
    if _after_retrain:
        _res_info_path = f'{_res_dir}/atk_transfer_after_retrain_res.json'
    if os.path.exists(_res_info_path):
        _global_logger.info(f'load res info: {_res_info_path}')
        _res_info = json.loads(open(_res_info_path, 'r').read())
    else:
        _res_info = {}
    for _model in _model_list:
        for _task in _task_list:
            _atk_args_list = UtilsMeta.get_attack_type_args_list(_model, _task, _mode='attack')
            _global_logger.info(f'atk method res len: {len(_atk_args_list)}')
            for _atk_args_group in _atk_args_list:
                _base_atk_args = _atk_args_group[0]
                _atk_res_list = [UtilsMeta.validate_attack_args(_atk_args) for _atk_args in _atk_args_group]
                _res_key = f'{_model}_{_task}_{_atk_res_list[0]["prefix"]}'
                if _res_key not in _res_info:
                    _base_atk_args_dict = vars(_base_atk_args)
                    _res_info[_res_key] = {
                        'atk_args': _base_atk_args_dict,
                    }
                # # if model CodeBERT and task vulnerability_detection, pop all res
                # if _args.is_debug and _model == 'CodeBERT' and _task == 'vulnerability_detection':
                #     _global_logger.info(f'pop all res: {_res_key}')
                #     del _res_info[_res_key]
                #     open(_res_info_path, 'w').write(json.dumps(_res_info, cls=NpEncoder))
                #     continue
                # assert all atk res existing len = target len
                if not all([_atk_res['existing_len'] == _atk_res['target_len'] for _atk_res in _atk_res_list]):
                    print(f'len not match: {_base_atk_args}')
                    _global_logger.info(f'len not match: {_base_atk_args}')
                    continue
                # if not _exclude_self, to transfer all
                if not _exclude_self:
                   _to_transfer_model_list = _model_list
                else:
                   _to_transfer_model_list = [x for x in _model_list if x != _model]
                # just skip if all to transfer key exist
                if all([x in _res_info[_res_key] for x in _to_transfer_model_list]):
                    _global_logger.info(f'skip transfer atk: {_res_key} - all exist')
                    continue
                # prepare model path, skip if not exist
                _to_transfer_args_dict = {}
                for _to_transfer_model in _to_transfer_model_list:
                    # skip if _to_transfer_model in _res_info and _to_transfer_model in _res_info[_res_key]
                    if _to_transfer_model in _res_info[_res_key]:
                        _global_logger.info(f'skip transfer atk: {_res_key} - {_to_transfer_model}')
                        continue
                    _to_transfer_args = copy.deepcopy(_base_atk_args)
                    _to_atk_model_meta = UtilsMeta.get_model_meta(_to_transfer_model, _task)
                    _to_transfer_args.model_name = _to_transfer_model
                    # set model_name_or_path
                    _to_transfer_args.model_name_or_path = _to_atk_model_meta['model_meta']['model']
                    # set model_tokenizer_name
                    _to_transfer_args.model_tokenizer_name = _to_atk_model_meta['model_meta']['tokenizer']
                    # set model_num_label
                    _to_transfer_args.model_num_label = _to_atk_model_meta['model_param']['model_num_label']
                    # set model_code_length
                    _to_transfer_args.model_code_length = _to_atk_model_meta['model_param']['model_code_length']
                    # set model_eval_batch_size
                    _to_transfer_args.model_eval_batch_size = int(_to_atk_model_meta['model_param']['batch_size'] * _gpu_times)
                    # set model_data_flow_length if model_data_flow_length in param
                    if 'model_data_flow_length' in _to_atk_model_meta['model_param']:
                        _to_transfer_args.model_data_flow_length = _to_atk_model_meta['model_param']['model_data_flow_length']
                    # device from _args
                    _to_transfer_args.device = _args.device
                    if _after_retrain:
                        # determine model path
                        _to_transfer_atk_res_info = UtilsMeta.validate_attack_args(_to_transfer_args)
                        _to_transfer_ptm_path = VRModel.get_reatk_model_state_path(_to_transfer_args, _to_transfer_atk_res_info)
                    else:
                        _to_transfer_ptm_path = VRModel.get_model_state_path(_to_transfer_args)
                    # only add if path exist
                    if os.path.exists(_to_transfer_ptm_path):
                        _to_transfer_args_dict[_to_transfer_model] = _to_transfer_args
                # skip if _to_transfer_args_dict key len = 0  
                if len(_to_transfer_args_dict.keys()) == 0:
                    _global_logger.info(f'skip transfer atk: {_res_key} - all exist')
                    continue
                # _to_transfer_model_list = [x for x in _model_list if x != _model]
                _atk_adv_code_info_list = InputDataset.load_transfer_code_list(_atk_args_group, _atk_res_list, _global_logger)
                _atk_adv_code_label_truth_list = [x['label'] for x in _atk_adv_code_info_list]
                for _to_transfer_model in _to_transfer_model_list:
                    _res_atk_key = _to_transfer_model
                    # skip if _res_key in _res_info and _res_atk_key in _res_info[_res_key]
                    if _res_key in _res_info and _res_atk_key in _res_info[_res_key]:
                        _global_logger.info(f'skip transfer atk: {_res_key} - {_res_atk_key}')
                        continue
                    # save _label_pred_list under _res_dir as pickle format *.pkl
                    # save path as {_res_dir}/atk_transfer_{prefix}_{model}_label.pkl
                    _label_pred_path = f'{_res_dir}/atk_transfer_{_res_key}_{_res_atk_key}_label.pkl'
                    if _after_retrain:
                        _label_pred_path = f'{_res_dir}/atk_transfer_after_retrain_{_res_key}_{_res_atk_key}_label.pkl'
                    # just load if pkl exist
                    if os.path.exists(_label_pred_path) and not (_args.is_debug and _model == 'CodeBERT' and _task == 'vulnerability_detection'):
                        _global_logger.info(f'load label pred: {_label_pred_path}')
                        _label_pred_list = pickle.load(open(_label_pred_path, 'rb'))
                    else:
                        # construct from _to_transfer_args_dict
                        _to_atk_args = _to_transfer_args_dict[_to_transfer_model]
                        # load to atk model
                        _to_atk_model_class_info = VRModel.get_model_class_list(_to_atk_args)
                        _to_atk_config = _to_atk_model_class_info[0].from_pretrained(_to_atk_args.model_name_or_path, cache_dir='tg/cache')
                        _to_atk_config.num_labels = _to_atk_args.model_num_label
                        _to_atk_tokenizer = _to_atk_model_class_info[2].from_pretrained(_to_atk_args.model_tokenizer_name, do_lower_case=False, cache_dir='tg/cache')
                        _to_atk_model = _to_atk_model_class_info[1].from_pretrained(_to_atk_args.model_name_or_path, config=_to_atk_config, from_tf=False, cache_dir='tg/cache')
                        _to_atk_vr_model = VRModel(
                            _config=_to_atk_config,
                            _args=_to_atk_args,
                            _encoder=_to_atk_model,
                            _tokenizer=_to_atk_tokenizer,
                            _logger=_global_logger
                        )
                        if _after_retrain:
                            # get after retrain model path
                            _to_atk_res_info = UtilsMeta.validate_attack_args(_to_atk_args)
                            _to_atk_ptm_path = VRModel.get_reatk_model_state_path(_to_atk_args, _to_atk_res_info)
                            # skip if model not exist
                            if not os.path.exists(_to_atk_ptm_path):
                                _global_logger.info(f'skip atk after retrain: {_to_atk_ptm_path} not exist')
                                continue
                        else:
                            _to_atk_ptm_path = VRModel.get_model_state_path(_to_atk_args)
                        _to_atk_vr_model.load_state_dict(torch.load(_to_atk_ptm_path, map_location=_to_atk_args.device))
                        _to_atk_vr_model.to(_to_atk_args.device)
                        _global_logger.info(f'load to atk model: {_to_atk_ptm_path}')
                        # just evaluate
                        _to_atk_model.eval()
                        _to_atk_code_dataset = UtilsFeature.get_code_dataset(
                            _args=_to_atk_args,
                            _code_info_list=_atk_adv_code_info_list,
                            _tokenizer=_to_atk_tokenizer,
                            _logger=_global_logger
                        )
                        _, _label_pred_list = _to_atk_vr_model.get_dataset_result(
                            _dataset=_to_atk_code_dataset,
                            _batch_size=_to_atk_args.model_eval_batch_size
                        )
                        open(_label_pred_path, 'wb').write(pickle.dumps(_label_pred_list))
                    # get perf
                    _atk_transfer_perf = InputDataset.get_perf(
                        _atk_adv_code_label_truth_list,
                        _label_pred_list
                    )
                    _res_info[_res_key][_res_atk_key] = _atk_transfer_perf
                    _global_logger.info(f'atk transfer perf: {_res_key} - {_res_atk_key} - {_atk_transfer_perf}')
                    open(_res_info_path, 'w+').write(json.dumps(_res_info, cls=NpEncoder))
    pass
def run_transfer_reatk(_args, _after_retrain=False, _exclude_self=True):
    # load from reatk res to eval
    # python -u src/model/start_debug.py --mode=reatk_transfer
    # python -u src/model/start_debug.py --mode=reatk_transfer_after_retrain
    # just after initial attack, evalute the initial attack across different model
    # attack examples: success adversarial examples from different attack methods
    # attack target: original trained model
    _model_list = UtilsMeta.get_model_list()
    _task_list = UtilsMeta.get_task_list()
    _res_dir = f'tg/datasets/reatk_transfer'
    if _after_retrain:
        _res_dir = f'tg/datasets/reatk_transfer_after_retrain'
    if not os.path.exists(_res_dir):
        os.makedirs(_res_dir, exist_ok=True)
    # multiply model_eval_batch_size by gpu size / 16
    _gpu_gb = UtilsMeta.get_gpu_gb()
    _gpu_times = _gpu_gb / 16
    _res_info_path = f'{_res_dir}/reatk_transfer_res.json'
    if _after_retrain:
        _res_info_path = f'{_res_dir}/reatk_transfer_after_retrain_res.json'
    if os.path.exists(_res_info_path):
        _global_logger.info(f'load res info: {_res_info_path}')
        _res_info = json.loads(open(_res_info_path, 'r').read())
    else:
        _res_info = {}
    for _model in _model_list:
        for _task in _task_list:
            _atk_args_list = UtilsMeta.get_attack_type_args_list(_model, _task, _mode='attack')
            _global_logger.info(f'atk method res len: {len(_atk_args_list)}')
            for _atk_args_group in _atk_args_list:
                _base_atk_args = _atk_args_group[0]
                _atk_res_list = [UtilsMeta.validate_reattack_args(_atk_args) for _atk_args in _atk_args_group]
                _res_key = f'{_model}_{_task}_{_atk_res_list[0]["prefix"]}'
                if _res_key not in _res_info:
                    _base_atk_args_dict = vars(_base_atk_args)
                    _res_info[_res_key] = {
                        'atk_args': _base_atk_args_dict,
                    }
                # assert all reatk res existing len = target len
                if not all([_atk_res['existing_len'] == _atk_res['target_len'] for _atk_res in _atk_res_list]):
                    print(f'len not match: {_base_atk_args}')
                    _global_logger.info(f'len not match: {_base_atk_args}')
                    continue
                # if not _exclude_self, to transfer all
                if not _exclude_self:
                   _to_transfer_model_list = _model_list
                else:
                   _to_transfer_model_list = [x for x in _model_list if x != _model]
                # just skip if all to transfer key exist
                if all([x in _res_info[_res_key] for x in _to_transfer_model_list]):
                    _global_logger.info(f'skip transfer reatk: {_res_key} - all exist')
                    continue
                # prepare model path, skip if not exist
                _to_transfer_args_dict = {}
                for _to_transfer_model in _to_transfer_model_list:
                    # skip if _to_transfer_model in _res_info and _to_transfer_model in _res_info[_res_key]
                    if _to_transfer_model in _res_info[_res_key]:
                        _global_logger.info(f'skip transfer reatk: {_res_key} - {_to_transfer_model}')
                        continue
                    _to_transfer_args = copy.deepcopy(_base_atk_args)
                    _to_atk_model_meta = UtilsMeta.get_model_meta(_to_transfer_model, _task)
                    _to_transfer_args.model_name = _to_transfer_model
                    # set model_name_or_path
                    _to_transfer_args.model_name_or_path = _to_atk_model_meta['model_meta']['model']
                    # set model_tokenizer_name
                    _to_transfer_args.model_tokenizer_name = _to_atk_model_meta['model_meta']['tokenizer']
                    # set model_num_label
                    _to_transfer_args.model_num_label = _to_atk_model_meta['model_param']['model_num_label']
                    # set model_code_length
                    _to_transfer_args.model_code_length = _to_atk_model_meta['model_param']['model_code_length']
                    # set model_eval_batch_size
                    _to_transfer_args.model_eval_batch_size = int(_to_atk_model_meta['model_param']['batch_size'] * _gpu_times)
                    # set model_data_flow_length if model_data_flow_length in param
                    if 'model_data_flow_length' in _to_atk_model_meta['model_param']:
                        _to_transfer_args.model_data_flow_length = _to_atk_model_meta['model_param']['model_data_flow_length']
                    # device from _args
                    _to_transfer_args.device = _args.device
                    if _after_retrain:
                        # determine model path
                        _to_transfer_atk_res_info = UtilsMeta.validate_reattack_args(_to_transfer_args)
                        _to_transfer_ptm_path = VRModel.get_reatk_model_state_path(_to_transfer_args, _to_transfer_atk_res_info)
                    else:
                        _to_transfer_ptm_path = VRModel.get_model_state_path(_to_transfer_args)
                    # only add if path exist
                    if os.path.exists(_to_transfer_ptm_path):
                        _to_transfer_args_dict[_to_transfer_model] = _to_transfer_args
                # skip if _to_transfer_args_dict key len = 0  
                if len(_to_transfer_args_dict.keys()) == 0:
                    _global_logger.info(f'skip transfer reatk: {_res_key} - all exist')
                    continue
                # _to_transfer_model_list = [x for x in _model_list if x != _model]
                _atk_adv_code_info_list = InputDataset.load_transfer_code_list(_atk_args_group, _atk_res_list, _global_logger)
                _atk_adv_code_label_truth_list = [x['label'] for x in _atk_adv_code_info_list]
                for _to_transfer_model in _to_transfer_model_list:
                    _res_atk_key = _to_transfer_model
                    # skip if _res_key in _res_info and _res_atk_key in _res_info[_res_key]
                    if _res_key in _res_info and _res_atk_key in _res_info[_res_key]:
                        _global_logger.info(f'skip transfer reatk: {_res_key} - {_res_atk_key}')
                        continue
                    # save _label_pred_list under _res_dir as pickle format *.pkl
                    # save path as {_res_dir}/atk_transfer_{prefix}_{model}_label.pkl
                    _label_pred_path = f'{_res_dir}/reatk_transfer_{_res_key}_{_res_atk_key}_label.pkl'
                    if _after_retrain:
                        _label_pred_path = f'{_res_dir}/reatk_transfer_after_retrain_{_res_key}_{_res_atk_key}_label.pkl'
                    # just load if pkl exist
                    if os.path.exists(_label_pred_path) and not (_args.is_debug and _model == 'CodeBERT' and _task == 'vulnerability_detection'):
                        _global_logger.info(f'load label pred: {_label_pred_path}')
                        _label_pred_list = pickle.load(open(_label_pred_path, 'rb'))
                    else:
                        # construct from _to_transfer_args_dict
                        _to_atk_args = _to_transfer_args_dict[_to_transfer_model]
                        # load to atk model
                        _to_atk_model_class_info = VRModel.get_model_class_list(_to_atk_args)
                        _to_atk_config = _to_atk_model_class_info[0].from_pretrained(_to_atk_args.model_name_or_path, cache_dir='tg/cache')
                        _to_atk_config.num_labels = _to_atk_args.model_num_label
                        _to_atk_tokenizer = _to_atk_model_class_info[2].from_pretrained(_to_atk_args.model_tokenizer_name, do_lower_case=False, cache_dir='tg/cache')
                        _to_atk_model = _to_atk_model_class_info[1].from_pretrained(_to_atk_args.model_name_or_path, config=_to_atk_config, from_tf=False, cache_dir='tg/cache')
                        _to_atk_vr_model = VRModel(
                            _config=_to_atk_config,
                            _args=_to_atk_args,
                            _encoder=_to_atk_model,
                            _tokenizer=_to_atk_tokenizer,
                            _logger=_global_logger
                        )
                        if _after_retrain:
                            # get after retrain model path
                            _to_atk_res_info = UtilsMeta.validate_attack_args(_to_atk_args)
                            _to_atk_ptm_path = VRModel.get_reatk_model_state_path(_to_atk_args, _to_atk_res_info)
                            # skip if model not exist
                            if not os.path.exists(_to_atk_ptm_path):
                                _global_logger.info(f'skip reatk after retrain: {_to_atk_ptm_path} not exist')
                                continue
                        else:
                            _to_atk_ptm_path = VRModel.get_model_state_path(_to_atk_args)
                        _to_atk_vr_model.load_state_dict(torch.load(_to_atk_ptm_path, map_location=_to_atk_args.device))
                        _to_atk_vr_model.to(_to_atk_args.device)
                        _global_logger.info(f'load to atk model: {_to_atk_ptm_path}')
                        # just evaluate
                        _to_atk_model.eval()
                        _to_atk_code_dataset = UtilsFeature.get_code_dataset(
                            _args=_to_atk_args,
                            _code_info_list=_atk_adv_code_info_list,
                            _tokenizer=_to_atk_tokenizer,
                            _logger=_global_logger
                        )
                        _, _label_pred_list = _to_atk_vr_model.get_dataset_result(
                            _dataset=_to_atk_code_dataset,
                            _batch_size=_to_atk_args.model_eval_batch_size
                        )
                        open(_label_pred_path, 'wb').write(pickle.dumps(_label_pred_list))
                    # get perf
                    _atk_transfer_perf = InputDataset.get_perf(
                        _atk_adv_code_label_truth_list,
                        _label_pred_list
                    )
                    _res_info[_res_key][_res_atk_key] = _atk_transfer_perf
                    _global_logger.info(f'atk transfer perf: {_res_key} - {_res_atk_key} - {_atk_transfer_perf}')
                    open(_res_info_path, 'w+').write(json.dumps(_res_info, cls=NpEncoder))
    pass

def run_rank_position(_args):
    # python -u src/model/start_debug.py --mode=rank_position --position_type=predefined_mask --model_name=CodeBERT --model_name_or_path=microsoft/codebert-base --model_tokenizer_name=microsoft/codebert-base --model_code_length=512 --model_num_label=66 --task_type=authorship_attribution --index 0 132
    # python -u src/model/start_debug.py --mode=rank_position --position_type=predefined_replacement
    # python -u src/model/start_debug.py --mode=rank_position --position_type=random
    # skip dynamic value calc
    # python -u src/model/start_debug.py --mode=rank_position --position_type=dynamic_mask
    # python -u src/model/start_debug.py --mode=rank_position --position_type=dynamic_replacement
    if _args.position_type not in ['predefined_mask', 'predefined_replacement', 'random']:
        _global_logger.info(f'position_type: {_args.position_type} not need to run rank_position')
        return
    UtilsMeta.prepare_attack_file(_args)
    _code_info_list = InputDataset.load_code_list(_args, _global_logger)
    # log len
    _global_logger.info(f'code_info_list len: {len(_code_info_list)}')
    _rank_file_info = UtilsPositionScore.get_pos_file_info(_args, _global_logger)
    # direct return if res file exists and len >= code_info_list
    if _rank_file_info['existing_len'] >= len(_code_info_list):
        _global_logger.info(f'res file exists and len >= code_info_list')
        return
    _class_info = VRModel.get_model_class_list(_args)
    _config = _class_info[0].from_pretrained(_args.model_name_or_path, cache_dir='tg/cache')
    _config.num_labels = _args.model_num_label
    _tokenizer = _class_info[2].from_pretrained(_args.model_tokenizer_name, do_lower_case=False, cache_dir='tg/cache')
    _vr_model = None
    if _args.position_type != 'random':
        _model = _class_info[1].from_pretrained(_args.model_name_or_path, config=_config, from_tf=False, cache_dir='tg/cache')
        _vr_model = VRModel(_config=_config, 
                            _args=_args,
                            _encoder=_model, 
                            _tokenizer=_tokenizer, 
                            _logger=_global_logger)
        if _args.rank_with_ptm:
            _ptm_path = VRModel.get_model_state_path(_args)
            _global_logger.info(f'load ptm path: {_ptm_path}')
            # load state
            _vr_model.load_state_dict(torch.load(_ptm_path, map_location=_args.device))
        _vr_model.to(_args.device)
    _model_info = {
        'model': _vr_model,
        'tokenizer': _tokenizer
    }
    _code_loader = tqdm(_code_info_list)
    for _code_idx, _code_info in enumerate(_code_loader):
        # skip if _code_idx < _res_file_info['existing_len']
        if _code_idx < _rank_file_info['existing_len']:
            _global_logger.info(f'Skip rank position: {_code_idx} of {len(_code_info_list)}')
            continue
        # rank position
        _code_str = _code_info['code']
        _tokenized_info = UtilsTokenizer.get_tokenized_info(_args, _code_str, _model_info['tokenizer'], _global_logger)
        _code_pos_info = None
        # skip if _tokenized_info valid_identifier_list empty
        if len(_tokenized_info['valid_identifier_list']) == 0:
            _code_pos_info = []
        else:
            # log idx
            _global_logger.info(f'Rank position: {_code_idx} of {len(_code_info_list)}')
            if _args.position_type == 'predefined_mask':
                # mask each position, rank by score
                _code_pos_info = UtilsPositionScore.calc_identifier_importance_score_by_mask(
                    _args=_args,
                    _model_info=_model_info,
                    _code_info=_code_info,
                    _tokenized_info=_tokenized_info,
                    _logger=_global_logger
                )
            elif _args.position_type == 'predefined_replacement':
                _code_pos_info = UtilsPositionScore.calc_identifier_importance_score_by_predefined_replacement(
                    _args=_args,
                    _model_info=_model_info,
                    _code_info=_code_info,
                    _tokenized_info=_tokenized_info,
                    _logger=_global_logger
                )
            elif _args.position_type == 'random':
                _code_pos_info = UtilsPositionScore.calc_identifier_importance_score_by_random(
                    _args=_args,
                    _model_info=_model_info,
                    _code_info=_code_info,
                    _tokenized_info=_tokenized_info,
                    _logger=_global_logger
                )
        # set description
        _code_loader.set_description(f'Rank position: {_code_idx} of {len(_code_info_list)}')
        # write append into res file, json format
        open(_rank_file_info['path'], 'a').write(json.dumps(_code_pos_info, cls=NpEncoder) + '\n')
        # raise NotImplementedError
    _global_logger.info(f'Rank position done')
    # tokenize code, mask/replace each position, rank by score
    pass
def run_generate_substitution(_args):
    # python -u src/model/start_debug.py --mode=generate_substitution --substitution_type=random
    # python -u src/model/start_debug.py --mode=generate_substitution --substitution_type=token --model_name=CodeBERT --model_name_or_path=microsoft/codebert-base --model_tokenizer_name=microsoft/codebert-base --model_code_length=512 --model_num_label=66 --model_eval_batch_size=16 --task_type=authorship_attribution --index 0 500
    if _args.substitution_type not in ['token', 'code', 'random']:
        _global_logger.info(f'substitution_type: {_args.substitution_type} not need to run generate_substitution')
        return
    UtilsMeta.prepare_attack_file(_args)
    _code_info_list = InputDataset.load_code_list(_args, _global_logger)
    _global_logger.info(f'code_info_list len: {len(_code_info_list)}')
    _subs_file_info = UtilsSubstitutionGeneration.get_subs_file_info(_args, _global_logger)
    # skip if already finished
    if _subs_file_info['existing_len'] >= len(_code_info_list):
        _global_logger.info(f'res file exists and len >= code_info_list')
        return
    # only load model if not random
    _class_info = UtilsSubstitutionGeneration.get_masked_model_class_list(_args)
    _masked_tokenier = _class_info[1].from_pretrained(_args.model_tokenizer_name, do_lower_case=False, cache_dir='tg/cache')
    _masked_model = None
    if _args.substitution_type != 'random':
        _masked_model = _class_info[0].from_pretrained(_args.model_name_or_path, from_tf=False, cache_dir='tg/cache')
        _masked_model.to(_args.device)
    _model_info = {
        'model': _masked_model,
        'tokenizer': _masked_tokenier
    }
    # iterate code_info_list, mask & generate & rank, or random generate
    _code_loader = tqdm(_code_info_list)
    for _code_idx, _code_info in enumerate(_code_loader):
        # skip if less than existing_len
        if _code_idx < _subs_file_info['existing_len']:
            _global_logger.info(f'Skip generate_substitution: {_code_idx} of {len(_code_info_list)}')
            continue
        # generate substitution
        _code_str = _code_info['code']
        _tokenized_info = UtilsTokenizer.get_tokenized_info(_args, _code_str, _masked_tokenier, _global_logger)
        # skip gen if _tokenized_info valid_identifier_list empty
        if len(_tokenized_info['valid_identifier_list']) == 0:
            _global_logger.info(f'Skip generate_substitution: {_code_idx} of {len(_code_info_list)}')
            _subs_info = {}
        else:
            # log idx
            _global_logger.info(f'Generate substitution: {_code_idx} of {len(_code_info_list)}')
            # target subs dict, key=original name, value=new name list rank by score
            if _args.substitution_type == 'token':
                _subs_info = UtilsSubstitutionGeneration.generate_mask_subs_dict_by_token_sim(
                    _args=_args,
                    _model_info=_model_info,
                    _code_info=_code_info,
                    _tokenized_info=_tokenized_info,
                    _logger=_global_logger
                )
            elif _args.substitution_type == 'code':
                _subs_info = UtilsSubstitutionGeneration.generate_mask_subs_dict_by_code_sim(
                    _args=_args,
                    _model_info=_model_info,
                    _code_info=_code_info,
                    _tokenized_info=_tokenized_info,
                    _logger=_global_logger
                )
            elif _args.substitution_type == 'random':
                _subs_info = UtilsSubstitutionGeneration.generate_mask_subs_dict_by_random(
                    _args=_args,
                    _model_info=_model_info,
                    _code_info=_code_info,
                    _tokenized_info=_tokenized_info,
                    _logger=_global_logger
                )
        _code_loader.set_description(f'Generate substitution: {_code_idx} of {len(_code_info_list)}')
        if _args.is_debug:
            # log
            _global_logger.info(f'subs: {json.dumps(_subs_info)}')
            raise NotImplementedError
        # write append into res file, json format
        open(_subs_file_info['path'], 'a').write(json.dumps(_subs_info, cls=NpEncoder) + '\n')
    _global_logger.info(f'Generate substitution done')
def run_eval_test(_args):
    '''eval model performance on data
    python -u src/model/start_debug.py --mode=eval_test --model_eval_batch_size=16
    '''
    _task_list = UtilsMeta.get_task_list()
    _model_list = UtilsMeta.get_model_list()
    _eval_res_file = 'tmp_fix_atk/eval_test_res.json'
    _eval_res = {}
    if os.path.exists(_eval_res_file):
        _eval_res = json.loads(open(_eval_res_file).read())
    for _task in _task_list:
        for _model in _model_list:
            # skip if exists in _eval_res
            if _model in _eval_res and _task in _eval_res[_model]:
                _global_logger.info(f'Skip eval_test: {_task} {_model}')
                continue
            # initlize model, load model state
            _model_meta = UtilsMeta.get_model_meta(_model_name=_model, _task_type=_task)
            _batch_size = _model_meta['model_param']['batch_size']
            _gpu_db = UtilsMeta.get_gpu_gb()
            _batch_size = int(_batch_size * (_gpu_db / 16))
            # log batch size
            _global_logger.info(f'eval_test: {_task} {_model} batch_size: {_batch_size}')
            _tmp_args = argparse.Namespace(
                mode=_args.mode,
                model_eval_batch_size=_batch_size,
                device=_args.device,
                task_type=_task,
                model_name=_model,
                model_name_or_path=_model_meta['model_meta']['model'],
                model_tokenizer_name=_model_meta['model_meta']['tokenizer'],
                model_code_length=_model_meta['model_param']['model_code_length'],
                model_num_label=_model_meta['model_param']['model_num_label'],
                code_with_cache=False
            )
            if 'model_data_flow_length' in _model_meta['model_param']:
                _tmp_args.model_data_flow_length = _model_meta['model_param']['model_data_flow_length']
            _model_class_info = VRModel.get_model_class_list(_tmp_args)
            _config = _model_class_info[0].from_pretrained(_tmp_args.model_name_or_path, cache_dir='tg/cache')
            _config.num_labels = _tmp_args.model_num_label
            _tokenizer = _model_class_info[2].from_pretrained(_tmp_args.model_tokenizer_name, do_lower_case=True, cache_dir='tg/cache')
            _model_obj = _model_class_info[1].from_pretrained(_tmp_args.model_name_or_path, config=_config, from_tf=False, cache_dir='tg/cache')
            _vr_model = VRModel(_config=_config,
                                _args=_tmp_args,
                                _encoder=_model_obj,
                                _tokenizer=_tokenizer,
                                _logger=_global_logger)
            _ptm_path = VRModel.get_model_state_path(_tmp_args)
            _vr_model.load_state_dict(torch.load(_ptm_path, map_location=_args.device))
            _vr_model.to(_tmp_args.device)
            _dataset_meta = UtilsMeta.get_dataset_meta('eval_test', _task)
            _size = _dataset_meta['size']
            _step = _dataset_meta['step']
            _step_count = _size // _step
            if _size % _step != 0:
                _step_count += 1
            # iterate step
            _total_label_truth = []
            _total_label_pred = []
            for _step_idx in range(_step_count):
                _step_start = _step_idx * _step
                _start_end = _step_start + _step
                _tmp_args.index = [_step_start, _start_end]
                _global_logger.info(f'eval_test: {_task} {_model} {_step_start} {_start_end}')
                UtilsMeta.prepare_attack_file(_tmp_args)
                _code_info_list = InputDataset.load_code_list(_tmp_args, _global_logger)
                # covert code into feature
                _code_feature_matrix = [UtilsFeature.get_code_feature(
                    _tmp_args, _code_info['code'], _code_info, _tokenizer, _global_logger) 
                                        for _code_info in _code_info_list]
                _dataset_info = InputDataset.get_dataset(_tmp_args, _code_feature_matrix)
                _label_truth = [_info['label'] for _info in _code_info_list]
                _prob_pred, _label_pred = _vr_model.get_dataset_result(
                    _dataset=_dataset_info,
                    _batch_size=_batch_size,
                )
                # compare _label_pred
                _pref_info = InputDataset.get_perf(_label_truth, _label_pred)
                # log
                _global_logger.info(f'eval_test: {_task} {_model} {_step_start} {_start_end} {_pref_info}')
                _total_label_truth.extend(_label_truth)
                _total_label_pred.extend(_label_pred)
            # total perf
            _total_pref_info = InputDataset.get_perf(_total_label_truth, _total_label_pred)
            _global_logger.info(f'eval_test: {_task} {_model} total {_total_pref_info}')
            if _model not in _eval_res:
                _eval_res[_model] = {}
            _eval_res[_model][_task] = _total_pref_info
            # write into file
            open(_eval_res_file, 'w+').write(json.dumps(_eval_res, 
                                                        cls=NpEncoder,
                                                        indent=4
                                                        ) + '\n')
    pass

def run_retraining_split(_args):
# python -u src/model/start_debug.py --mode=retraining_split --task_type=authorship_attribution
# python -u src/model/start_debug.py --mode=retraining_split --task_type=clone_detection
# python -u src/model/start_debug.py --mode=retraining_split --task_type=code_classification
# python -u src/model/start_debug.py --mode=retraining_split --task_type=vulnerability_detection
    UtilsMeta.prepare_attack_file(_args)
    UtilsMeta.validate_retraining_args(_args)
    # split data index, random
    InputDataset.split_retraining_dataset(_args, _global_logger)
def run_retraining(_args):
    # python -u src/model/start_debug.py --mode=retraining --task_type=authorship_attribution  --substitution_type=code  --attack_type=ga --model_name=CodeBERT --model_name_or_path=microsoft/codebert-base --model_tokenizer_name=microsoft/codebert-base --model_num_label=66 --model_code_length=512 --model_eval_batch_size=128
    # split successful attack code (just the whole attack target data) into 2 part, 1 for adv tuning with original data, 1 for further attack after runing (avoid data leakage), the whole attack target data may contain some unsuccessful sample, them are also needed in re-training
    # load original model, launch re-training on original data + 1/2 adversarial attack samples
    # calc model param delta, save
    # evaluate model's performance before and after adv tuning based on the rest of 1/2 adversarial data
    # same dataset should share same split, tg/datasets/shared/{task}
    _args_dict = vars(copy.deepcopy(_args))
    # pop device
    _args_dict.pop('device')
    _to_compare_key_list = ['task_type', 'model_name', 'position_type', 'substitution_type', 'search_type', 'attack_type', 'attack_type_list']
    if _args.is_debug:
        _cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        # try to load correct and wrong args list and compare find
        _correct_args_dict = {}
        _wrong_args_dict = {}
        _cur_key = json.dumps({k: _args_dict[k] for k in _to_compare_key_list})
        # list all under tmp_fix_atk, start witl correct_args
        _correct_file_list = os.listdir('tmp_fix_atk')
        _correct_file_list = [f for f in _correct_file_list if f.startswith('correct_args')]
        for _file in _correct_file_list:
            for _line in open(f'tmp_fix_atk/{_file}').readlines():
                _tmp_dict = json.loads(_line)
                _tmp_key = json.dumps({k: _tmp_dict[k] for k in _to_compare_key_list})
                _correct_args_dict[_tmp_key] = _tmp_dict
        # print correct dict key len
        print(f'correct_args_dict key len: {len(_correct_args_dict)}')
        if _cur_key in _correct_args_dict:
            _global_logger.info(f'correct args found: {_correct_args_dict[_cur_key]}')
            print('correct args found')
            return
        _wrong_file_list = os.listdir('tmp_fix_atk')
        _wrong_file_list = [f for f in _wrong_file_list if f.startswith('wrong_args')]
        for _file in _wrong_file_list:
            for _line in open(f'tmp_fix_atk/{_file}').readlines():
                _tmp_dict = json.loads(_line)
                _tmp_key = json.dumps({k: _tmp_dict[k] for k in _to_compare_key_list})
                _wrong_args_dict[_tmp_key] = _tmp_dict
        print(f'wrong_args_dict key len: {len(_wrong_args_dict)}')
        if _cur_key in _wrong_args_dict:
            _global_logger.info(f'wrong args found: {_wrong_args_dict[_cur_key]}')
            print('wrong args found')
            return
    UtilsMeta.prepare_attack_file(_args)
    UtilsMeta.validate_retraining_args(_args)
    InputDataset.split_retraining_dataset(_args, _global_logger)
    # load_retrain_code_list
    _retrain_info = InputDataset.load_retrain_code_list(_args, _global_logger)
    _original_code_info_list = _retrain_info['original_code_info_list']
    
    _augmented_code_info_list = _retrain_info['augmented_code_info_list']
    _test_code_info_list = _retrain_info['test_code_info_list']
    _test_adv_code_info_list = _retrain_info['test_adv_code_info_list']
    _class_info = VRModel.get_model_class_list(_args)
    _config = _class_info[0].from_pretrained(_args.model_name_or_path, cache_dir='tg/cache')
    _config.num_labels = _args.model_num_label
    _tokenizer = _class_info[2].from_pretrained(_args.model_tokenizer_name, do_lower_case=False, cache_dir='tg/cache')
    _model = _class_info[1].from_pretrained(_args.model_name_or_path, config=_config, from_tf=False, cache_dir='tg/cache')
    _vr_model = VRModel(
        _config=_config,
        _args=_args,
        _encoder=_model,
        _tokenizer=_tokenizer,
        _logger=_global_logger
    )
    _retraining_model_info = VRModel.get_retraining_model_info(_args)
    _is_from_original = False    
    _train_info = {
        'augmented_perf': [],
        'test_perf': [],
        'test_adv_perf': [],
        'train_loss': []
    }
    # log _retraining_model_info
    _global_logger.info(f'_retraining_model_info: {_retraining_model_info}')
    if _retraining_model_info['retrain_model_exist']:
        if os.path.exists(_retraining_model_info['retrain_info_path']):
            # check _retraining_model_info['retrain_info_path']
            # skip if pert len > num epoch not debug
            _pert_info = json.loads(open(_retraining_model_info['retrain_info_path']).read())
            if len(_pert_info['test_adv_perf']) > _args.train_num_epoch:
                _global_logger.info(f'skip retraining: {len(_pert_info["test_adv_perf"])} > {_args.train_num_epoch}')
                if _args.is_debug:
                    # write into correct_args
                    open(f'tmp_fix_atk/correct_args_{_cuda_visible_devices}.jsonl', 'a').write(json.dumps(_args_dict) + '\n')
                    _global_logger.info(f'write into correct_args: {_args_dict}')
                    print('write into correct_args')
                return
            else:
                # merge train info perf key
                for _k in _train_info.keys():
                    _train_info[_k] = _pert_info[_k]
                _global_logger.info(f'continue retraining: {len(_pert_info["test_adv_perf"])} <= {_args.train_num_epoch}')
        # load model path
        _vr_model.load_state_dict(torch.load(_retraining_model_info['retrain_model_path'], map_location=_args.device))
        _global_logger.info(f'model initlized with previous {_retraining_model_info["retrain_model_path"]}')
    else:
        _is_from_original = True
        _vr_model.load_state_dict(torch.load(_retraining_model_info['model_path'], map_location=_args.device))
        _global_logger.info(f'model initlized with {_retraining_model_info["model_path"]}')
    _vr_model.to(_args.device)
    # cache _code_dataset to speed up, tg/datasets/shared/{task_type}/{model_name}_dataset_original_cache.pkl
    _original_code_dataset_cache_path = f'tg/datasets/shared/{_args.task_type}/{_args.model_name}_dataset_original_cache.pkl'
    if os.path.exists(_original_code_dataset_cache_path):
        _original_code_dataset = pickle.load(open(_original_code_dataset_cache_path, 'rb'))
        _global_logger.info(f'load original code dataset len: {len(_original_code_dataset)} from cache: {_original_code_dataset_cache_path}')
    else:
        _original_code_dataset = UtilsFeature.get_code_dataset(
            _args=_args,
            _code_info_list=_original_code_info_list,
            _tokenizer=_tokenizer,
            _logger=_global_logger
        )
        # only dump if debug
        pickle.dump(_original_code_dataset, open(_original_code_dataset_cache_path, 'wb'))
        _global_logger.info(f'save original code dataset len: {len(_original_code_dataset)} into cache: {_original_code_dataset_cache_path}')
        if _args.is_debug:
            raise NotImplementedError
    _augmented_dataset = UtilsFeature.get_code_dataset(
        _args=_args,
        _code_info_list=_augmented_code_info_list,
        _tokenizer=_tokenizer,
        _logger=_global_logger
    )
    # _code_dataset = _original_code_dataset concat augmented
    # _code_dataset = _original_code_dataset._feature_list + _augmented_dataset._feature_list
    _code_dataset = InputDataset.get_dataset(_args, _original_code_dataset._feature_list + _augmented_dataset._feature_list)
    _global_logger.info(f'code dataset len: {len(_code_dataset)}')
    _test_code_dataset = UtilsFeature.get_code_dataset(
        _args=_args,
        _code_info_list=_test_code_info_list,
        _tokenizer=_tokenizer,
        _logger=_global_logger
    )
    _test_adv_code_dataset = UtilsFeature.get_code_dataset(
        _args=_args,
        _code_info_list=_test_adv_code_info_list,
        _tokenizer=_tokenizer,
        _logger=_global_logger
    )
    _augmented_label_truth_list = [int(_info['label']) for _info in _augmented_code_info_list]
    _test_label_truth_list = [int(_info['label']) for _info in _test_code_info_list]
    _test_adv_label_truth_list = [int(_info['label']) for _info in _test_adv_code_info_list]
    _train_sampler = RandomSampler(_code_dataset)
    _train_dataloader = torch.utils.data.DataLoader(
        _code_dataset,
        sampler=_train_sampler,
        batch_size=_args.model_train_batch_size
    )
    _args.max_step = len(_train_dataloader) * _args.train_num_epoch
    _args.save_step = _args.train_eval_step_size * len(_train_dataloader)
    _args.warmup_step = len(_train_dataloader)
    _args.logging_step = len(_train_dataloader)
    _no_decay = ['bias', 'LayerNorm.weight']
    _optimizer_group_params = [
        {
            'params': [p for n, p in _vr_model.named_parameters() if not any(nd in n for nd in _no_decay)],
            'weight_decay': _args.train_weight_decay
        },
        {
            'params': [p for n, p in _vr_model.named_parameters() if any(nd in n for nd in _no_decay)],
            'weight_decay': 0.0
        }
    ]
    _optimizer = torch.optim.AdamW(
        _optimizer_group_params,
        lr=_args.train_lr,
        eps=_args.train_adam_epsilon
    )
    _scheduler = transformers.get_linear_schedule_with_warmup(
        _optimizer,
        num_warmup_steps=_args.warmup_step,
        num_training_steps=_args.max_step
    )
    _global_logger.info(f'optimizer and scheduler initlized')
    _global_logger.info('***** Running retraining *****')
    _global_step = 0
    _train_loss, _iter_train_loss, _logging_loss, _avg_loss, _train_idx, _train_num = 0.0, 0.0, 0.0, 0.0, 0, 0
    _vr_model.zero_grad()
    _args_dict = vars(copy.deepcopy(_args))
    _args_dict.pop('device')
    _train_info['args'] = _args_dict
    # only eval if _train_info['augmented_perf'] empty
    if len(_train_info['augmented_perf']) == 0:
        _vr_model.eval()
        _, _test_adv_label_pred_list = _vr_model.get_dataset_result(
            _dataset=_test_adv_code_dataset,
            _batch_size=_args.model_eval_batch_size
        )
        _test_adv_perf = InputDataset.get_perf(_test_adv_label_truth_list, _test_adv_label_pred_list)
        _global_logger.info(f'test_adv_perf_info: {_test_adv_perf}')
        _train_info['test_adv_perf'].append(_test_adv_perf)
        # assert _test_adv_perf f1 0, else raise error
        if _is_from_original and _test_adv_perf['macro_f1'] > 0:
            if _args.is_debug:
                if _cur_key not in _wrong_args_dict:
                    # write args dict into tmp_fix_atk/wrong_args.jsonl
                    open(f'tmp_fix_atk/wrong_args_{_cuda_visible_devices}.jsonl', 'a').write(json.dumps(_args_dict) + '\n')
                    print('write into wrong_args')
                    return
            raise NotImplementedError
        # return if debug
        if _args.is_debug and _cur_key not in _correct_args_dict:
            # write into tmp_fix_atk/correct_args.jsonl
            open(f'tmp_fix_atk/correct_args_{_cuda_visible_devices}.jsonl', 'a').write(json.dumps(_args_dict) + '\n')
            print('write into correct_args')
            return
        _, _augmented_label_pred_list = _vr_model.get_dataset_result(
            _dataset=_augmented_dataset,
            _batch_size=_args.model_eval_batch_size
        )
        _augmented_perf = InputDataset.get_perf(_augmented_label_truth_list, _augmented_label_pred_list)
        _train_info['augmented_perf'].append(_augmented_perf)
        _global_logger.info(f'augmented_perf_info: {_augmented_perf}')
        _, _test_label_pred_list = _vr_model.get_dataset_result(
            _dataset=_test_code_dataset,
            _batch_size=_args.model_eval_batch_size
        )
        _test_perf = InputDataset.get_perf(_test_label_truth_list, _test_label_pred_list)
        _global_logger.info(f'test_perf_info: {_test_perf}')
        _train_info['test_perf'].append(_test_perf)
    # return if debug
    if _args.is_debug:
        if _cur_key not in _correct_args_dict:
            # write into tmp_fix_atk/correct_args.jsonl
            open(f'tmp_fix_atk/correct_args_{_cuda_visible_devices}.jsonl', 'a').write(json.dumps(_args_dict) + '\n')
            print('write into correct_args')
        return
    # start training
    # log args again
    _global_logger.info(f'training args: {_args}')
    _vr_model.train()
    for _idx in range(_args.train_num_epoch):
        _batch_loader = tqdm(_train_dataloader, 
                             total=len(_train_dataloader), 
                             desc=f'Epoch {_idx} of {_args.train_num_epoch}')
        _train_num = 0
        _train_loss = 0.0
        _global_logger.info(f'Epoch {_idx} of {_args.train_num_epoch}')
        for _step_idx, _batch_info in enumerate(_batch_loader):
            # iter if tensor to device else not
            # _input_param = [x.to("cuda")  for x in _batch_info]
            _input_param = [x.to(_args.device) if isinstance(x, torch.Tensor) else x for x in _batch_info]
            _vr_model.train()
            _loss, _logits = _vr_model(*_input_param)
            if _args.train_grad_accu_step > 1:
                _loss = _loss / _args.train_grad_accu_step
            _loss.backward()
            # clip
            torch.nn.utils.clip_grad_norm_(_vr_model.parameters(), _args.train_max_grad_norm)
            _train_num += 1
            _train_loss += _loss.item()
            _iter_train_loss += _loss.item()
            if _avg_loss == 0:
                _avg_loss = _loss.item()
            _avg_loss = _train_loss / _train_num
            _batch_loader.set_description(f'Epoch {_idx}-{_args.train_num_epoch} of {len(_train_dataloader)} loss: {_avg_loss:.4f}')
            if (_step_idx + 1) % _args.train_grad_accu_step == 0:
                _optimizer.step()
                _scheduler.step()
                _vr_model.zero_grad()
                _global_step += 1
                _avg_loss = np.exp((_iter_train_loss - _logging_loss) / (_global_step - _train_idx))
                if _global_step % _args.logging_step == 0:
                    _logging_loss = _iter_train_loss
                    _train_idx = _global_step
                    _global_logger.info(f'Epoch {_idx}-{_args.train_num_epoch} of {len(_train_dataloader)} loss: {_avg_loss:.4f}')
                    _train_info['train_loss'].append(_avg_loss)
                if _global_step % _args.save_step == 0:
                    # save model diff
                    # eval test again
                    _vr_model.eval()
                    _, _augmented_label_pred_list = _vr_model.get_dataset_result(
                        _dataset=_augmented_dataset,
                        _batch_size=_args.model_eval_batch_size
                    )
                    _augmented_perf = InputDataset.get_perf(_augmented_label_truth_list, _augmented_label_pred_list)
                    _global_logger.info(f'augmented_perf_info: {_augmented_perf}')
                    _train_info['augmented_perf'].append(_augmented_perf)
                    _, _test_label_pred_list = _vr_model.get_dataset_result(
                        _dataset=_test_code_dataset,
                        _batch_size=_args.model_eval_batch_size
                    )
                    _test_perf = InputDataset.get_perf(_test_label_truth_list, _test_label_pred_list)
                    _global_logger.info(f'test_perf_info: {_test_perf}')
                    _train_info['test_perf'].append(_test_perf)
                    _, _test_adv_label_pred_list = _vr_model.get_dataset_result(
                        _dataset=_test_adv_code_dataset,
                        _batch_size=_args.model_eval_batch_size
                    )
                    _test_adv_perf = InputDataset.get_perf(_test_adv_label_truth_list, _test_adv_label_pred_list)
                    _global_logger.info(f'test_adv_perf_info: {_test_adv_perf}')
                    _train_info['test_adv_perf'].append(_test_adv_perf)
                    _need_dump_model = False
                    if len(_train_info['augmented_perf']) <= 1 or not os.path.exists(_retraining_model_info['retrain_model_path']):
                        _need_dump_model = True
                    else:
                        _history_max_f1 = max([_info['macro_f1'] for _info in _train_info['augmented_perf']])
                        if _test_perf['macro_f1'] > _history_max_f1:
                            _need_dump_model = True
                    # skip save if debug
                    if not _args.is_debug:
                        # dump train info
                        with open(_retraining_model_info['retrain_info_path'], 'w+') as _f:
                            _f.write(json.dumps(_train_info))
                        # compare among history perf, if not better skip dump
                        # dump current model
                        if _need_dump_model:
                            torch.save(_vr_model.state_dict(), _retraining_model_info['retrain_model_path'])
                            _global_logger.info(f'model saved: {_retraining_model_info["retrain_model_path"]}')
                pass
            pass   
    pass

def main_handler():
    # print sys argv
    print(sys.argv)
    _parser = UtilsMeta.get_args_parser()
    _args = _parser.parse_args()
    if _args.use_gpu:
        if not torch.cuda.is_available():
            raise ValueError('CUDA is not available')
        _args.device = torch.device('cuda')
    else:
        # python -u src/model/start_debug_bak.py --mode=attack --task_type=clone_detection  --position_type=predefined_replacement  --substitution_type=random  --attack_type_list greedy ga --attack_type=twostep --model_name=CodeT5 --model_name_or_path=Salesforce/codet5-base-multi-sum --model_tokenizer_name=Salesforce/codet5-base --model_num_label=2 --model_code_length=384 --model_eval_batch_size=4  --index 3000 3500 --use_gpu=False
        _args.device = torch.device('cpu')
    set_seed(_args.seed)
    global _global_logger
    _global_logger.info(f'args: {_args}')
    print(f'args: {_args}')
    if _args.mode == 'attack':
        run_initial_attack(_args)
    elif _args.mode == 'rank_position':
        run_rank_position(_args)
    elif _args.mode == 'generate_substitution':
        run_generate_substitution(_args)
    elif _args.mode == 'eval_test':
        run_eval_test(_args)
    elif _args.mode == 'retraining_split':
        run_retraining_split(_args)
    elif _args.mode == 'retraining':
        run_retraining(_args)
    elif _args.mode == 'attack_after_retrain':
        run_attack_after_retrain(_args)
    elif _args.mode == 'transfer':
        run_transfer_atk(_args, _exclude_self=False)
    elif _args.mode == 'transfer_after_retrain':
        run_transfer_atk(_args, _after_retrain=True)
        # run_transfer_atk_after_retrain(_args)
    elif _args.mode == 'reatk_transfer':
        run_transfer_reatk(_args)
    elif _args.mode == 'reatk_transfer_after_retrain':
        run_transfer_reatk(_args, _after_retrain=True)
    pass


if __name__ == '__main__':
    initlize_logger()
    main_handler()
    print('Done')
    torch.cuda.empty_cache()
    exit(0)