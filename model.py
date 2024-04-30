# /usr/bin/env python
# coding=utf-8
"""model"""
from collections import Counter

import torch
import torch.nn as nn
from transformers import BertModel

from modules import PERC


class MultiNonLinearClassifier(nn.Module):
    """
    实体提取：两个三分类任务
    输入：（1）句子的特征向量h（BERT预训练模型编码器的输出），（2）潜在的关系Prel

    关系判断：多标签二分类问题
    输入：句子的特征向量h（BERT预训练模型编码器的输出）
    输出：
    """

    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        features_output = self.hidden2tag(features_tmp)
        return features_output


class SequenceLabelForSO(nn.Module):
    """
    2.实体提取的两个输入sum结合方式，（1）句子的特征向量h（BERT预训练模型编码器的输出），（2）潜在的关系Prel
    """

    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(SequenceLabelForSO, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag_sub = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.hidden2tag_obj = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        """
        Args:
            input_features: (bs, seq_len, h)
        """
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        sub_output = self.hidden2tag_sub(features_tmp)
        obj_output = self.hidden2tag_obj(features_tmp)
        return sub_output, obj_output


class BertForRE(nn.Module):
    """
    算法原理参考：https://zhuanlan.zhihu.com/p/440495418
    """

    def __init__(self, params):
        super(BertForRE, self).__init__()
        self.max_seq_len = params.max_seq_length
        self.seq_tag_size = params.seq_tag_size  # NER序列标注 标签数 BIO=3
        self.rel_num = params.rel_num
        self.bert_embedding = params.hidden_size
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # ResNet
        self.perc = PERC(input_dims=self.bert_embedding, output_dims=self.bert_embedding)

        # BiGRU
        # self.gru_embedding = 384
        # self.gru = nn.GRU(self.bert_embedding, self.gru_embedding, batch_first=True, bidirectional=True)
        # self._dropout = nn.Dropout(p=params.drop_prob)

        # ATT
        # self.num_heads = 16
        # self.attention_type = 'scaled_dot'
        # self.att = MultiHeadAttention(model_dim=self.bert_embedding,
        #                               num_heads=self.num_heads,
        #                               dropout_rate=params.drop_prob,
        #                               attention_type=self.attention_type)
        self.lnorm = nn.LayerNorm(self.bert_embedding)
        self.sigmoid = nn.Sigmoid()
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # pretrain model
        self.bert = BertModel.from_pretrained(params.bert_model_dir)
        # sequence tagging 2.实体提取
        self.sequence_tagging_sub = MultiNonLinearClassifier(self.bert_embedding * 2, self.seq_tag_size,
                                                             params.drop_prob)  # hidden_size * 2：输入是2部分 hi + uj？
        self.sequence_tagging_obj = MultiNonLinearClassifier(self.bert_embedding * 2, self.seq_tag_size,
                                                             params.drop_prob)
        # 实体提取的两个输入sum结合方式
        self.sequence_tagging_sum = SequenceLabelForSO(self.bert_embedding, self.seq_tag_size, params.drop_prob)
        # global correspondence 3.主宾语对齐
        self.global_corres = MultiNonLinearClassifier(self.bert_embedding * 2, 1, params.drop_prob)
        # relation judgement  1.关系判断
        self.rel_judgement = MultiNonLinearClassifier(self.bert_embedding, params.rel_num, params.drop_prob)
        self.rel_embedding = nn.Embedding(params.rel_num, self.bert_embedding)

    @staticmethod
    def masked_avgpool(sent, mask):
        mask_ = mask.masked_fill(mask == 0, -1e9).float()  # 将0替换一个很小的负值
        score = torch.softmax(mask_, -1)
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)
        # (batch_size, 1, seq_len) * (batch_size, seq_len, h) = (batch_size, 1, h)  squeeze(1)=(batch_size, h)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            seq_tags=None,
            potential_rels=None,
            corres_tags=None,
            rel_tags=None,
            ex_params=None
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            rel_tags: (bs, rel_num)
            potential_rels: (bs,), only in train stage.
            seq_tags: (bs, 2, seq_len)
            corres_tags: (bs, seq_len, seq_len)
            ex_params: experiment parameters
        """
        # get params for experiments  每个关系的主客体对齐阈值拉姆达2，关系阈值拉姆达1
        corres_threshold, rel_threshold = ex_params.get('corres_threshold', 0.5), ex_params.get('rel_threshold', 0.1)
        # ablation study 消融研究
        ensure_corres, ensure_rel = ex_params['ensure_corres'], ex_params['ensure_rel']

        # pre-train model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        bs, seq_len, h = sequence_output.size()
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # PERCNet
        out = sequence_output.permute(2, 0, 1)  # [bt,seq,dim] -> [dim, bt, seq]
        out = out.unsqueeze(0)
        out = self.perc(out)
        out = out.permute(1, 2, 0)

        # BiGRU
        # out, _ = self.gru(sequence_output)  # (B, T, 2 * D/2)
        # out = out.contiguous().view(-1, self.gru_embedding * 2)
        # out = self._dropout(out)
        # out = out.contiguous().view(bs, seq_len, -1)
        # print(sequence_output)
        # out, _ = self.gru(sequence_output, None)  # (B, T, 2 * D/2)
        # print(out)

        # ATT
        # out = sequence_output
        # attn_mask = attention_padding_mask(attention_mask, attention_mask)  # (B, T, T)
        # out, _ = self.att(out, out, out, attn_mask=attn_mask)
        # # Residual connection and Layer Normalization
        sequence_output = self.lnorm(sequence_output + out)
        sequence_output = self.sigmoid(sequence_output)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # 1.关系判断：平均池化 + 全连接层 + 激活函数
        if ensure_rel:
            # (bs, h)  平均池化
            h_k_avg = self.masked_avgpool(sequence_output, attention_mask)
            # (bs, rel_num)
            rel_pred = self.rel_judgement(h_k_avg)

        # 3.主宾语对齐
        # before fuse relation representation
        if ensure_corres:
            # 主体、客体的对应关系
            # for every position $i$ in sequence, should concate $j$ to predict.
            sub_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (bs, s, s, h)
            obj_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (bs, s, s, h)
            # batch x seq_len x seq_len x 2*hidden
            corres_pred = torch.cat([sub_extend, obj_extend], 3)  # 把两个实体的token直接拼接
            # (bs, seq_len, seq_len)
            corres_pred = self.global_corres(corres_pred).squeeze(-1)
            mask_tmp1 = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            mask_tmp2 = attention_mask.unsqueeze(1)  # (batch_size, 1, seq_len)
            corres_mask = mask_tmp1 * mask_tmp2  # (batch_size, seq_len, seq_len)

        # 验证阶段：relation predict and data construction in inference stage
        xi, pred_rels = None, None
        if ensure_rel and seq_tags is None:
            # (bs, rel_num)
            rel_pred_onehot = torch.where(torch.sigmoid(rel_pred) > rel_threshold,
                                          torch.ones(rel_pred.size(), device=rel_pred.device),
                                          torch.zeros(rel_pred.size(), device=rel_pred.device))

            # if potential relation is null
            for idx, sample in enumerate(rel_pred_onehot):
                if 1 not in sample:
                    # (rel_num,)  若根据阈值判断不存在任何关系时,取rel_pred的最大值关系
                    max_index = torch.argmax(rel_pred[idx])
                    sample[max_index] = 1
                    rel_pred_onehot[idx] = sample

            # 2*(sum(x_i),)
            bs_idxs, pred_rels = torch.nonzero(rel_pred_onehot, as_tuple=True)  # 存在关系的bs及其关系类别pred_rels
            # get x_i
            xi_dict = Counter(bs_idxs.tolist())  # 统计每一个句子预测关系种数
            xi = [xi_dict[idx] for idx in range(bs)]  # xi=每一个句子的预测关系种数
            # xi = [(xi_dict[idx],) for idx in range(bs)]

            pos_seq_output = []  # list长度=bs * num()
            pos_potential_rel = []
            pos_attention_mask = []
            for bs_idx, rel_idx in zip(bs_idxs, pred_rels):
                # (seq_len, h)
                pos_seq_output.append(sequence_output[bs_idx])  # 预测关系种类数=n, 每个句子重复n次
                pos_attention_mask.append(attention_mask[bs_idx])  # 预测关系种类数=n, 每个句子重复n次
                pos_potential_rel.append(rel_idx)
            # (sum(x_i), seq_len, h)
            sequence_output = torch.stack(pos_seq_output, dim=0)
            # (sum(x_i), seq_len)
            attention_mask = torch.stack(pos_attention_mask, dim=0)
            # (sum(x_i),)
            potential_rels = torch.stack(pos_potential_rel, dim=0)
        # ablation of relation judgement
        elif not ensure_rel and seq_tags is None:
            # construct test data
            sequence_output = sequence_output.repeat((1, self.rel_num, 1)).view(bs * self.rel_num, seq_len, h)
            attention_mask = attention_mask.repeat((1, self.rel_num)).view(bs * self.rel_num, seq_len)
            potential_rels = torch.arange(0, self.rel_num, device=input_ids.device).repeat(bs)

        # 2.实体抽取
        # (bs/sum(x_i), h)
        rel_emb = self.rel_embedding(potential_rels)

        # relation embedding vector fusion
        rel_emb = rel_emb.unsqueeze(1).expand(-1, seq_len, h)  # （bs, seq_len, h）  每个数据复制seq_len次
        # 两个输入的结合方式
        if ex_params['emb_fusion'] == 'concat':
            # (bs/sum(x_i), seq_len, 2*h)
            decode_input = torch.cat([sequence_output, rel_emb], dim=-1)  # 输入：（1）句子的特征向量h，（2）潜在的关系Prel (训练阶段用的是真实tag)
            # (bs/sum(x_i), seq_len, tag_size)  两次三分类
            output_sub = self.sequence_tagging_sub(decode_input)
            output_obj = self.sequence_tagging_obj(decode_input)
        elif ex_params['emb_fusion'] == 'sum':
            # (bs/sum(x_i), seq_len, h)
            decode_input = sequence_output + rel_emb
            # (bs/sum(x_i), seq_len, tag_size)
            output_sub, output_obj = self.sequence_tagging_sum(decode_input)

        # train
        if seq_tags is not None:
            # calculate loss
            attention_mask = attention_mask.view(-1)  # 将张量变成一维的结构  (batch_size * seq_len)
            # Loss(seq): sequence label loss
            loss_func = nn.CrossEntropyLoss(reduction='none')
            loss_seq_sub = (loss_func(output_sub.view(-1, self.seq_tag_size),
                                      seq_tags[:, 0, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
            loss_seq_obj = (loss_func(output_obj.view(-1, self.seq_tag_size),
                                      seq_tags[:, 1, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
            loss_seq = (loss_seq_sub + loss_seq_obj) / 2
            # Loss(global)
            # init
            loss_matrix, loss_rel = torch.tensor(0), torch.tensor(0)
            if ensure_corres:
                corres_pred = corres_pred.view(bs, -1)
                corres_mask = corres_mask.view(bs, -1)
                corres_tags = corres_tags.view(bs, -1)
                loss_func = nn.BCEWithLogitsLoss(reduction='none')
                loss_matrix = (loss_func(corres_pred,
                                         corres_tags.float()) * corres_mask).sum() / corres_mask.sum()

            if ensure_rel:
                # Loss(rel)
                loss_func = nn.BCEWithLogitsLoss(reduction='mean')
                loss_rel = loss_func(rel_pred, rel_tags.float())

            loss = loss_seq + loss_matrix + loss_rel
            return loss, loss_seq, loss_matrix, loss_rel
        # inference
        else:
            # (sum(x_i), seq_len)
            pred_seq_sub = torch.argmax(torch.softmax(output_sub, dim=-1), dim=-1)
            pred_seq_obj = torch.argmax(torch.softmax(output_obj, dim=-1), dim=-1)
            # (sum(x_i), 2, seq_len)
            pred_seqs = torch.cat([pred_seq_sub.unsqueeze(1), pred_seq_obj.unsqueeze(1)], dim=1)
            if ensure_corres:
                corres_pred = torch.sigmoid(corres_pred) * corres_mask
                # (bs, seq_len, seq_len)
                pred_corres_onehot = torch.where(corres_pred > corres_threshold,
                                                 torch.ones(corres_pred.size(), device=corres_pred.device),
                                                 torch.zeros(corres_pred.size(), device=corres_pred.device))
                xi = torch.tensor(xi).cuda()
                return pred_seqs, pred_corres_onehot, xi, pred_rels
                # return pred_seqs, pred_corres_onehot, pred_rels
            return pred_seqs, xi, pred_rels


if __name__ == '__main__':
    from transformers import BertConfig
    import utils
    import os

    params = utils.Params()
    # Prepare model
    bert_config = BertConfig.from_json_file(os.path.join(params.bert_model_dir, 'bert_config.json'))
    model = BertForRE.from_pretrained(config=bert_config,
                                      pretrained_model_name_or_path=params.bert_model_dir,
                                      params=params)
    model.to(params.device)

    for n, _ in model.named_parameters():
        print(n)
