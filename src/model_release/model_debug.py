import torch, copy, numpy as np, os
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification, T5Config, T5ForConditionalGeneration

class RobertaClassificationHead(torch.nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, _config, _args):
        super().__init__()
        self._args = _args
        if self._args.task_type == 'clone_detection':
            self.dense = torch.nn.Linear(_config.hidden_size*2, _config.hidden_size)
            self.dropout = torch.nn.Dropout(_config.hidden_dropout_prob)
            self.out_proj = torch.nn.Linear(_config.hidden_size, 2)
        else:
            self.dense = torch.nn.Linear(_config.hidden_size, _config.hidden_size)
            self.dropout = torch.nn.Dropout(_config.hidden_dropout_prob)
            self.out_proj = torch.nn.Linear(_config.hidden_size, _config.num_labels)
    def forward(self, feature_list, **kwargs):
        x = feature_list[:, 0, :]
        if self._args.task_type == 'clone_detection':
            x = x.reshape(-1, x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
class CodeT5RobertaClassificationHead(torch.nn.Module):
    def __init__(self, _config, _args):
        super().__init__()
        self._args = _args
        if self._args.task_type == 'clone_detection':
            self.dense = torch.nn.Linear(_config.hidden_size*2, _config.hidden_size)
            self.out_proj = torch.nn.Linear(_config.hidden_size, _config.num_labels)
        else:
            self.dense = torch.nn.Linear(_config.hidden_size, _config.hidden_size)
            self.out_proj = torch.nn.Linear(_config.hidden_size, _config.num_labels)
    def forward(self, feature_list, **kwargs):
        _x = feature_list
        if self._args.task_type == 'clone_detection':
            _x = _x.reshape(-1, _x.size(-1)*2)
        _x = self.dense(_x)
        _x = torch.tanh(_x)
        _x = self.out_proj(_x)
        return _x
class VRModel(torch.nn.Module):
    @staticmethod
    def get_model_state_path(_args):
        if _args.mode in ['rank_position', 'attack', 'fix_attack', 'transfer', 'eval_test']:
            # load from example: tg/models/CodeT5/authorship_attribution/model/model.bin
            _model_state_path = f'tg/models/{_args.model_name}/{_args.task_type}/model/model.bin'
            return _model_state_path
    def get_reatk_model_state_path(_args, _atk_res_info):
        # example tg/datasets/attack_res/CodeBERT_authorship_attribution/greedy_predefined_mask_token_retraining.bin
        _model_prefix = _atk_res_info['prefix']
        _model_path = f'tg/datasets/attack_res/{_args.model_name}_{_args.task_type}/{_model_prefix}_retraining.bin'
        return _model_path
    @staticmethod
    def get_retraining_model_info(_args):
        # try new model path first, if not exist use original one
        # target model tg/datasets/attack_res/{CodeBERT}_{authorship_attribution}/
        # model path has no index
        if _args.attack_type in ['mhm', 'codeattack', 'ga']:
            _retrain_model_path = f'tg/datasets/attack_res/{_args.model_name}_{_args.task_type}/{_args.attack_type}_{_args.substitution_type}_retraining.bin'
        elif _args.attack_type in ['random']:
            _retrain_model_path = f'tg/datasets/attack_res/{_args.model_name}_{_args.task_type}/random_retraining.bin'
        elif _args.attack_type in ['greedy', 'beam']:
            _retrain_model_path = f'tg/datasets/attack_res/{_args.model_name}_{_args.task_type}/{_args.attack_type}_{_args.position_type}_{_args.substitution_type}_retraining.bin'
        elif _args.attack_type in ['twostep']:
            _attack_type_label = f'{_args.attack_type}_{"_".join(_args.attack_type_list)}'
            if all([_x in ['mhm', 'ga', 'codeattack'] for _x in _args.attack_type_list]):
                _retrain_model_path = f'tg/datasets/attack_res/{_args.model_name}_{_args.task_type}/{_attack_type_label}_{_args.substitution_type}_retraining.bin'
            else:
                _retrain_model_path = f'tg/datasets/attack_res/{_args.model_name}_{_args.task_type}/{_attack_type_label}_{_args.position_type}_{_args.substitution_type}_retraining.bin'
        _model_path = f'tg/models/{_args.model_name}/{_args.task_type}/model/model.bin'
        _train_file_path = _retrain_model_path.replace('.bin', '_info.json')
        return {
            'model_path': _model_path,
            'retrain_model_path': _retrain_model_path,
            'retrain_model_exist': os.path.exists(_retrain_model_path),
            'retrain_info_path': _train_file_path,
        }
    def reset_query_count(self):
        self._query_count = 0
    @staticmethod
    def get_model_class_list(_args):
        '''return config, model, tokenizer'''
        if _args.model_name in ['GraphCodeBERT', 'CodeBERT'] and \
            _args.task_type in ['clone_detection', 'authorship_attribution', 'code_classification', 'vulnerability_detection']:
            if _args.model_name == 'CodeBERT' and _args.task_type in ['authorship_attribution', 'clone_detection']:
                return (RobertaConfig, RobertaModel, RobertaTokenizer)
            else:
                return (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
        elif _args.model_name == 'CodeT5':
            return (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
    def __init__(self, _config, _args, _encoder, _tokenizer, _logger=None):
        super().__init__()
        self._config = _config
        self._args = _args
        self.encoder = _encoder
        self._tokenizer = _tokenizer
        self._logger = _logger
        # initialize classifier
        if (_args.model_name == 'GraphCodeBERT' and _args.task_type in ['authorship_attribution', 'clone_detection', 'code_classification']) or \
            (_args.model_name == 'CodeBERT' and _args.task_type in ['clone_detection', 'authorship_attribution']):
            self.classifier = RobertaClassificationHead(_config, _args)
        elif (_args.model_name == 'CodeT5' and _args.task_type in ['authorship_attribution', 'clone_detection', 'code_classification', 'vulnerability_detection']):
            self.classifier = CodeT5RobertaClassificationHead(_config, _args)
        self._query_count = 0
    def forward(self, *params):
        '''
        params may have different len
        '''
        if self._args.model_name == 'GraphCodeBERT':
            if self._args.task_type == 'clone_detection':
                _input_id_list_0, _position_idx_0, _attn_mask_0, _input_id_list_1, _position_idx_1, _attn_mask_1 = params[:6]
                _label_list = None
                if len(params) > 6:
                    _label_list = params[6]
                _bs, _l = _input_id_list_0.size()
                _input_id_list = torch.cat([
                    _input_id_list_0.unsqueeze(1),
                    _input_id_list_1.unsqueeze(1)
                ], dim=1).view(_bs*2, _l)
                _position_idx = torch.cat([
                    _position_idx_0.unsqueeze(1),
                    _position_idx_1.unsqueeze(1)
                ], dim=1).view(_bs*2, _l)
                _attn_mask = torch.cat([
                    _attn_mask_0.unsqueeze(1),
                    _attn_mask_1.unsqueeze(1)
                ], dim=1).view(_bs*2, _l, _l)
                # embedding
                _node_mask = _position_idx.eq(0)
                _token_mask = _position_idx.ge(2)
                _inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(_input_id_list)
                _node2token_mask = _node_mask[:,:,None] & _token_mask[:,None,:] & _attn_mask
                _node2token_mask = _node2token_mask / (_node2token_mask.sum(-1) + 1e-10)[:,:,None]
                _avg_embeddings = torch.einsum("abc,acd->abd", _node2token_mask, _inputs_embeddings)
                _inputs_embeddings = _inputs_embeddings * (~_node_mask)[:,:,None] + _avg_embeddings * _node_mask[:,:,None]
                _output_list = self.encoder.roberta(
                    inputs_embeds=_inputs_embeddings,
                    attention_mask=_attn_mask,
                    position_ids=_position_idx,
                    token_type_ids=_position_idx.eq(-1).long()
                )[0]
                _logit_list = self.classifier(_output_list)
                _prob_list = torch.nn.functional.softmax(_logit_list, dim=1)
                if _label_list is not None:
                    _loss_func = torch.nn.CrossEntropyLoss()
                    _loss_list = _loss_func(_logit_list, _label_list)
                    return _loss_list, _prob_list
                else:
                    return _prob_list
            elif self._args.task_type in ['authorship_attribution', 'code_classification', 'vulnerability_detection']:
                # _input_id_list, _attn_mask, _position_idx = params[:3]
                # new param order
                _input_id_list, _position_idx, _attn_mask = params[:3]
                _label_list = None
                if len(params) > 3:
                    _label_list = params[3]
                _node_mask = _position_idx.eq(0)
                _token_mask = _position_idx.ge(2)
                _inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(_input_id_list)
                # log node_mask, token_mask, attn_mask shape
                # self._logger.info(f'node_mask shape: {_node_mask.size()}, token_mask shape: {_token_mask.size()}, attn_mask shape: {_attn_mask.size()}')
                _node2token_mask = _node_mask[:,:,None] & _token_mask[:,None,:] & _attn_mask
                _node2token_mask = _node2token_mask / (_node2token_mask.sum(-1) + 1e-10)[:,:,None]
                _avg_embeddings = torch.einsum("abc,acd->abd", _node2token_mask, _inputs_embeddings)
                _inputs_embeddings = _inputs_embeddings * (~_node_mask)[:,:,None] + _avg_embeddings * _node_mask[:,:,None]
                if self._args.task_type in ['authorship_attribution', 'code_classification']:
                    _output_list = self.encoder.roberta(
                        inputs_embeds=_inputs_embeddings,
                        attention_mask=_attn_mask,
                        position_ids=_position_idx,
                        token_type_ids=_position_idx.eq(-1).long()
                    )[0]
                    _logit_list = self.classifier(_output_list)
                    _prob_list = torch.nn.functional.softmax(_logit_list, dim=1)
                    if _label_list is not None:
                        _loss_func = torch.nn.CrossEntropyLoss()
                        _loss_list = _loss_func(_logit_list, _label_list)
                        return _loss_list, _prob_list
                    else:
                        return _prob_list
                elif self._args.task_type == 'vulnerability_detection':
                    _output_list = self.encoder(
                        inputs_embeds=_inputs_embeddings,
                        attention_mask=_attn_mask,
                        position_ids=_position_idx,
                        token_type_ids=_position_idx.eq(-1).long()
                    )[0]
                    _logit_list = _output_list
                    _prob_list = torch.nn.functional.sigmoid(_logit_list)
                    if _label_list is not None:
                        _label_list = _label_list.float()
                        _loss_list = torch.log(_prob_list[:,0] + 1e-10) * _label_list + torch.log((1 - _prob_list)[:, 0] + 1e-10) * (1 - _label_list)
                        _loss_list = -_loss_list.mean()
                        return _loss_list, _prob_list
                    else:
                        return _prob_list
        elif self._args.model_name == 'CodeT5':
            _input_id_list = params[0]
            _label_list = None
            if len(params) > 1:
                _label_list = params[1]
            _input_id_list = _input_id_list.view(-1, self._args.model_code_length)
            _attention_mask = _input_id_list.ne(self._tokenizer.pad_token_id)
            _output_list = self.encoder(
                input_ids=_input_id_list,
                attention_mask=_attention_mask,
                labels=_input_id_list,
                decoder_attention_mask=_attention_mask,
                output_hidden_states=True
            )
            _hidden_states = _output_list['decoder_hidden_states'][-1]
            _eos_mask = _input_id_list.eq(self._config.eos_token_id)
            if len(torch.unique(_eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            _output_list = _hidden_states[_eos_mask, :].view(
                _hidden_states.size(0), -1, _hidden_states.size(-1))[:, -1, :]
            _logit_list = self.classifier(_output_list)
            if self._args.task_type in ['vulnerability_detection']:
                _prob_list = torch.nn.functional.sigmoid(_logit_list)
                if _label_list is not None:
                    _label_list = _label_list.float()
                    _loss_list = torch.log(_prob_list[:,0] + 1e-10) * _label_list + \
                        torch.log((1 - _prob_list)[:, 0] + 1e-10) * (1 - _label_list)
                    _loss_list = -_loss_list.mean()
                    return _loss_list, _prob_list
                else:
                    return _prob_list
            else:
                _prob_list = torch.nn.functional.softmax(_logit_list, dim=1)
                if _label_list is not None:
                    _loss_func = torch.nn.CrossEntropyLoss()
                    _loss_list = _loss_func(_logit_list, _label_list)
                    return _loss_list, _prob_list
                else:
                    return _prob_list
        elif self._args.model_name == 'CodeBERT':
            _input_id_list = params[0]
            _label_list = None
            if len(params) > 1:
                _label_list = params[1]
            if self._args.task_type in ['authorship_attribution', 'clone_detection']:
                _id_list = _input_id_list.view(-1, self._args.model_code_length)
                _output_list = self.encoder(
                    input_ids = _id_list,
                    attention_mask=_id_list.ne(1),
                )[0]
                _logit_list = self.classifier(_output_list)
                _prob_list = torch.nn.functional.softmax(_logit_list, dim=1)
                if _label_list != None:
                    _loss_func = torch.nn.CrossEntropyLoss()
                    # log logit and label size
                    # self._logger.info(f'logit size: {_logit_list.size()}, label size: {_label_list.size()}, {_label_list}')
                    _loss_list = _loss_func(_logit_list, _label_list)
                    return _loss_list, _prob_list
                else:
                    return _prob_list
            elif self._args.task_type in ['code_classification', 'vulnerability_detection']:
                if _input_id_list.dim() == 1:
                    _input_id_list = _input_id_list.unsqueeze(0)
                _logit_list = self.encoder(
                    input_ids = _input_id_list,
                    attention_mask=_input_id_list.ne(1),
                )[0]
                if self._args.task_type == 'code_classification':
                    _prob_list = torch.softmax(_logit_list, -1)
                    if _label_list != None:
                        _loss_func = torch.nn.CrossEntropyLoss()
                        _loss_list = _loss_func(_logit_list, _label_list)
                        return _loss_list, _prob_list
                    else:
                        return _prob_list
                elif self._args.task_type == 'vulnerability_detection':
                    _prob_list = torch.nn.functional.sigmoid(_logit_list)
                    if _label_list != None:
                        _label_list = _label_list.float()
                        _loss_list = torch.log(_prob_list[:,0] + 1e-10) * _label_list + torch.log((1 - _prob_list)[:, 0] + 1e-10) * (1 - _label_list)
                        _loss_list = -_loss_list.mean()
                        return _loss_list, _prob_list
                    else:
                        return _prob_list
    def get_dataset_result(self, _dataset, _batch_size, _threshold=0.5):
        # if dataset len 0, return 2 empty list
        if len(_dataset) == 0:
            return [], []
        # update query count
        self._query_count += len(_dataset)
        _sampler = torch.utils.data.SequentialSampler(_dataset)
        _dataloader = torch.utils.data.DataLoader(_dataset, sampler=_sampler, batch_size=_batch_size, num_workers=4, pin_memory=False)
        _tqdm_loader = tqdm(_dataloader, desc='Model Get Result')
        # eval mode
        self.eval()
        _prob_list = []
        _logit_list = []
        _label_pred_list = []
        if self._args.model_name == 'GraphCodeBERT':
            if self._args.task_type == 'clone_detection':
                for _batch_idx, _batch_info in enumerate(_tqdm_loader):
                    # parse param from batch info after cuda
                    (_input_id_list_0, _position_idx_0, _attn_mask_0, _input_id_list_1, _position_idx_1, _attn_mask_1, _label) = [_x.to(self._args.device) for _x in _batch_info[:7]]
                    with torch.no_grad():
                        _logit = self.forward(
                            _input_id_list_0, _position_idx_0, _attn_mask_0,
                            _input_id_list_1, _position_idx_1, _attn_mask_1
                        )
                        _logit_list.append(_logit.cpu().numpy())
                    _tqdm_loader.set_description(f'{_batch_idx} of {len(_dataloader)}')
                _logit_list = np.concatenate(_logit_list, axis=0)
                _prob_list = _logit_list
                _label_pred_list = [0 if _first_v > _threshold else 1 for _first_v in _logit_list[:,0]]
            else:
                for _batch_idx, _batch_info in enumerate(_tqdm_loader):
                    _input_id_list, _position_idx, _attn_mask, _label = [_x.to(self._args.device) for _x in _batch_info[:4]]
                    with torch.no_grad():
                        _, _logit = self.forward(_input_id_list, _position_idx, _attn_mask, _label)
                        _logit_list.append(_logit.cpu().numpy())
                    _tqdm_loader.set_description(f'{_batch_idx} of {len(_dataloader)}')
                _logit_list = np.concatenate(_logit_list, axis=0)
                if self._args.task_type in ['authorship_attribution', 'code_classification']:
                    _prob_list = _logit_list
                    for _logit in _logit_list:
                        _label_pred_list.append(np.argmax(_logit))
                elif self._args.task_type == 'vulnerability_detection':
                    _prob_list = [[1 - _prob[0], _prob[0]] for _prob in _logit_list]
                    _label_pred_list = [1 if _label_v else 0 for _label_v in _logit_list[:, 0] > _threshold]
        elif self._args.model_name == 'CodeT5':
            for _batch_idx, _batch_info in enumerate(_tqdm_loader):
                _input_id_list, _label = [_x.to(self._args.device) for _x in _batch_info[:2]]
                with torch.no_grad():
                    _, _logit = self.forward(_input_id_list, _label)
                    _logit_list.append(_logit.cpu().numpy())
                _tqdm_loader.set_description(f'{_batch_idx} of {len(_dataloader)}')
            _logit_list = np.concatenate(_logit_list, axis=0)
            if self._args.task_type == 'clone_detection':
                _prob_list = _logit_list
                _label_pred_list = [0 if _first_v > _threshold else 1 for _first_v in _logit_list[:,0]]
            elif self._args.task_type == 'vulnerability_detection':
                _prob_list = [[1 - _prob[0], _prob[0]] for _prob in _logit_list]
                _label_pred_list = [1 if _label_v else 0 for _label_v in _logit_list[:, 0] > _threshold]
            else:
                _prob_list = _logit_list
                for _logit in _logit_list:
                    _label_pred_list.append(np.argmax(_logit))
        elif self._args.model_name == 'CodeBERT':
            if self._args.task_type in ['authorship_attribution', 'code_classification']:
                for _batch_idx, _batch_info in enumerate(_tqdm_loader):
                    _tqdm_loader.set_description(f'{_batch_idx} of {len(_dataloader)}')
                    _input_id_list, _label_list = [_x.to(self._args.device) for _x in _batch_info[:2]]
                    with torch.no_grad():
                        _, _logit  = self.forward(_input_id_list, _label_list)
                        _logit_list.append(_logit.cpu().numpy())
                _logit_list = np.concatenate(_logit_list, axis=0)
                _prob_list = _logit_list
                for _logit in _logit_list:
                    _label_pred_list.append(np.argmax(_logit))
            elif self._args.task_type == 'clone_detection':
                for _batch_idx, _batch_info in enumerate(_tqdm_loader):
                    _tqdm_loader.set_description(f'{_batch_idx} of {len(_dataloader)}')
                    _input_id_list, _label_list = [_x.to(self._args.device) for _x in _batch_info[:2]]
                    with torch.no_grad():
                        _, _logit  = self.forward(_input_id_list, _label_list)
                        _logit_list.append(_logit.cpu().numpy())
                _logit_list = np.concatenate(_logit_list, axis=0)
                _prob_list = _logit_list
                _label_pred_list = [0 if _first_v > _threshold else 1 for _first_v in _logit_list[:,0]]
            elif self._args.task_type == 'vulnerability_detection':
                for _batch_idx, _batch_info in enumerate(_tqdm_loader):
                    _tqdm_loader.set_description(f'{_batch_idx} of {len(_dataloader)}')
                    _input_id_list, _label_list = [_x.to(self._args.device) for _x in _batch_info[:2]]
                    with torch.no_grad():
                        _, _logit  = self.forward(_input_id_list, _label_list)
                        _logit_list.append(_logit.cpu().numpy())
                _logit_list = np.concatenate(_logit_list, axis=0)
                _prob_list = [[1 - _prob[0], _prob[0]] for _prob in _logit_list]
                _label_pred_list = [1 if _first_v > _threshold else 0 for _first_v in _logit_list[:,0]]
        if _prob_list is None or _label_pred_list is None:
            raise NotImplementedError
        # covnert _prob_list to list if not list type 
        if type(_prob_list) != list:
            _prob_list = _prob_list.tolist()
        return _prob_list, _label_pred_list

