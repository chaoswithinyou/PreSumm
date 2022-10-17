import copy

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, DistilBertConfig, DistilBertModel, AutoModel
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer

# ### START MODIFYING ###
# import sys
# # Add MobileBert_PyTorch
# sys.path.insert(1, '../../MobileBert_PyTorch')
# from model.modeling_mobilebert import MobileBertConfig, MobileBertModel
# ### END MODIFYING ###

def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False, other_bert=None):
        super(Bert, self).__init__()
        self.other_bert = other_bert
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)

        ### Start Modifying ###
        elif other_bert == 'distilbert':
            self.model = DistilBertModel.from_pretrained('distilbert-base-uncased', cache_dir=temp_dir)
        elif other_bert == 'phobert':
            self.model = AutoModel.from_pretrained('vinai/phobert-base',cache_dir=temp_dir)
            # Update config to finetune token type embeddings
            self.model.config.type_vocab_size = 2 

            # Create a new Embeddings layer, with 2 possible segments IDs instead of 1
            self.model.embeddings.token_type_embeddings = nn.Embedding(2, self.model.config.hidden_size)
                            
            # Initialize it
            self.model.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
        ### End Modifying ###

        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        ### Start Modifying ###
        # No token_type_ids for DistilBertModel
        if self.other_bert == 'distilbert':
            if(self.finetune):
                top_vec = self.model(input_ids=x, attention_mask=mask)[0]
            else:
                self.eval()
                with torch.no_grad():
                    top_vec = self.model(input_ids=x, attention_mask=mask)[0]
        ### End Modifying ###

        else:
            if(self.finetune):
                top_vec, _ = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)
            else:
                self.eval()
                with torch.no_grad():
                    top_vec, _ = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert1 = Bert(args.large, args.temp_dir, args.finetune_bert, args.other_bert) # Modified: Add `args.other_bert`
        self.bert2 = Bert(args.large, args.temp_dir, args.finetune_bert, args.other_bert) # Modified: Add `args.other_bert`
        self.bert3 = Bert(args.large, args.temp_dir, args.finetune_bert, args.other_bert) # Modified: Add `args.other_bert`
        self.bert4 = Bert(args.large, args.temp_dir, args.finetune_bert, args.other_bert) # Modified: Add `args.other_bert`
        self.ext_layer = ExtTransformerEncoder(self.bert1.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert1.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
            self.bert1.model = BertModel(bert_config)
            self.bert2.model = BertModel(bert_config)
            self.bert3.model = BertModel(bert_config)
            self.bert4.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert1.model.config.hidden_size)

        # if(args.max_pos>256):
        #     my_pos_embeddings = nn.Embedding(args.max_pos+2, self.bert.model.config.hidden_size)
        #     my_pos_embeddings.weight.data[:258] = self.bert.model.embeddings.position_embeddings.weight.data
        #     my_pos_embeddings.weight.data[258:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos+2-258,1)
        #     self.bert.model.embeddings.position_embeddings = my_pos_embeddings


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        #scr 1,lendoc
        lendoc = src[0].shape[0]
        count = 0
        isbert2 = 0
        isbert3 = 0
        isbert4 = 0
        endbert = lendoc
        numsen = 0
        for i in range(lendoc):
            count += 1
            if src[0,i] == 0:
                istart = i
            if src[0,i] == 2:
                iend = i
                numsen += 1
            if count == 256:
                count = i-istart+1
                if isbert2 == 0:
                    ibert2 = istart
                    isbert2 = 1
                elif isbert3 == 0:
                    ibert3 = istart
                    isbert3 = 1
                elif isbert4 == 0:
                    ibert4 = istart
                    isbert4 = 1
                else:
                    endbert = iend+1
                    break

        if isbert4 == 1:            
            top_vec1 = self.bert1(src[:,:ibert2], segs[:,:ibert2], mask_src[:,:ibert2])
            top_vec2 = self.bert2(src[:,ibert2:ibert3], segs[:,ibert2:ibert3], mask_src[:,ibert2:ibert3])
            top_vec3 = self.bert3(src[:,ibert3:ibert4], segs[:,ibert3:ibert4], mask_src[:,ibert3:ibert4])
            top_vec4 = self.bert4(src[:,ibert4:endbert], segs[:,ibert4:endbert], mask_src[:,ibert4:endbert])
            top_vec = torch.cat((top_vec1,top_vec2,top_vec3,top_vec4),1)
        elif isbert3 == 1:
            top_vec1 = self.bert1(src[:,:ibert2], segs[:,:ibert2], mask_src[:,:ibert2])
            top_vec2 = self.bert2(src[:,ibert2:ibert3], segs[:,ibert2:ibert3], mask_src[:,ibert2:ibert3])
            top_vec3 = self.bert3(src[:,ibert3:], segs[:,ibert3:], mask_src[:,ibert3:])
            top_vec = torch.cat((top_vec1,top_vec2,top_vec3),1)
        elif isbert2 == 1:
            top_vec1 = self.bert1(src[:,:ibert2], segs[:,:ibert2], mask_src[:,:ibert2])
            top_vec2 = self.bert2(src[:,ibert2:], segs[:,ibert2:], mask_src[:,ibert2:])
            top_vec = torch.cat((top_vec1,top_vec2),1)
        else:
            top_vec1 = self.bert1(src, segs, mask_src)
            top_vec = top_vec1
        #top vec 1, lendoc, 768
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss[:,:numsen]]
        sents_vec = sents_vec * mask_cls[:, :numsen, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls[:,:numsen]).squeeze(-1)
        return sent_scores, mask_cls[:,:numsen], numsen

class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if(args.max_pos>256):
            my_pos_embeddings = nn.Embedding(args.max_pos+2, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:258] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[258:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos+2-258,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
