import torch
import torch.distributed as dist
import torch.nn.functional as F

from LAVIS.lavis.models.base_model import all_gather_with_grad, concat_all_gather

def forward_stage1(model, samples, local_rank):
    image = samples["image"].to(f"cuda:{local_rank}")
    text = samples["text_input"]

    image_embeds = model.ln_vision(model.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device
    )

    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)

    query_output = model.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        use_cache=True,
        return_dict=True,
    )

    image_feats = F.normalize(
        model.vision_proj(query_output.last_hidden_state), dim=-1
    )

    text_tokens = model.tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=model.max_txt_len,
        return_tensors="pt",
    ).to(image.device)
    text_output = model.Qformer.bert(
        text_tokens.input_ids,
        attention_mask=text_tokens.attention_mask,
        return_dict=True,
    )
    text_feat = F.normalize(
        model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
    )

    ###============== Image-text Contrastive ===================###
    image_feats_all = concat_all_gather(
        image_feats
    )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
    text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

    sim_q2t = torch.matmul(
        image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
    ).squeeze()
    # [batch_size, batch_size*num_gpu, num_query_tokens]

    # image-text similarity: aggregate across all query tokens
    sim_i2t, _ = sim_q2t.max(-1)
    sim_i2t = sim_i2t / model.temp

    # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
    sim_t2q = torch.matmul(
        text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
    ).squeeze()

    # text-image similarity: aggregate across all query tokens
    sim_t2i, _ = sim_t2q.max(-1)
    sim_t2i = sim_t2i / model.temp  # [batch_size, batch_size*num_gpu]

    rank = dist.get_rank()
    bs = image.size(0)
    targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
        image.device
    )

    if "image_id" in samples.keys(): #coco retrieval finetuning
        image_ids = samples["image_id"].view(-1,1)
        image_ids_all = concat_all_gather(image_ids)
        pos_idx = torch.eq(image_ids, image_ids_all.t()).float()       
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
        sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)

        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()     
        loss_itc = (loss_t2i+loss_i2t)/2  
    else:                     
        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

    ###============== Image-text Matching ===================###
    text_input_ids_world = concat_all_gather(text_tokens.input_ids)
    text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
    image_embeds_world = all_gather_with_grad(image_embeds)
    with torch.no_grad():
        if "image_id" in samples.keys():
            mask = torch.eq(image_ids, image_ids_all.t())
            sim_t2i.masked_fill_(mask, -10000)
            sim_i2t.masked_fill_(mask, -10000)
        else:    
            sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
            sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)            
            
        weights_t2i = F.softmax(sim_t2i, dim=1)
        weights_i2t = F.softmax(sim_i2t, dim=1)

    # select a negative image for each text
    image_embeds_neg = []
    for b in range(bs):
        neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        image_embeds_neg.append(image_embeds_world[neg_idx])
    image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

    # select a negative text for each image
    text_ids_neg = []
    text_atts_neg = []
    for b in range(bs):
        neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        text_ids_neg.append(text_input_ids_world[neg_idx])
        text_atts_neg.append(text_attention_mask_world[neg_idx])

    text_ids_neg = torch.stack(text_ids_neg, dim=0)
    text_atts_neg = torch.stack(text_atts_neg, dim=0)

    text_ids_all = torch.cat(
        [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
    )  # pos, pos, neg
    text_atts_all = torch.cat(
        [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
        dim=0,
    )

    query_tokens_itm = model.query_tokens.expand(text_ids_all.shape[0], -1, -1)
    query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
        image.device
    )
    attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

    image_embeds_all = torch.cat(
        [image_embeds, image_embeds_neg, image_embeds], dim=0
    )  # pos, neg, pos
    image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
        image.device
    )

    output_itm = model.Qformer.bert(
        text_ids_all,
        query_embeds=query_tokens_itm,
        attention_mask=attention_mask_all,
        encoder_hidden_states=image_embeds_all,
        encoder_attention_mask=image_atts_all,
        return_dict=True,
    )

    vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
    vl_output = model.itm_head(vl_embeddings)
    logits = vl_output.mean(dim=1)

    itm_labels = torch.cat(
        [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
        dim=0,
    ).to(image.device)
    loss_itm = F.cross_entropy(logits, itm_labels)

    ##================= Image Captioning ========================##
    decoder_input_ids = text_tokens.input_ids.clone()
    decoder_input_ids[:, 0] = model.tokenizer.bos_token_id
    labels = decoder_input_ids.masked_fill(
        decoder_input_ids == model.tokenizer.pad_token_id, -100
    )

    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
        image.device
    )
    attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
    lm_output = model.Qformer(
        decoder_input_ids,
        attention_mask=attention_mask,
        past_key_values=query_output.past_key_values,
        return_dict=True,
        labels=labels,
    )

    loss_lm = lm_output.loss

    return {
        "loss": loss_itc + loss_itm + loss_lm,
        "loss_itc": loss_itc,
        "loss_itm": loss_itm,
        "loss_lm": loss_lm,
    }

def forward_stage2(model, samples, local_rank):
    image = samples["image"].to(f"cuda:{local_rank}")
    with model.maybe_autocast():
        image_embeds = model.ln_vision(model.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device
    )

    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = model.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )

    inputs_opt = model.opt_proj(query_output.last_hidden_state)
    atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

    model.opt_tokenizer.padding_side = "right"

    text = [t + "\n" for t in samples["text_input"]]

    opt_tokens = model.opt_tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=model.max_txt_len,
    ).to(image.device)

    targets = opt_tokens.input_ids.masked_fill(
        opt_tokens.input_ids == model.opt_tokenizer.pad_token_id, -100
    )
    if model.prompt:
        targets[:, : model.prompt_length] = -100  # do not apply loss to the prompt

    empty_targets = (
        torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
    )
    targets = torch.cat([empty_targets, targets], dim=1)

    inputs_embeds = model.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
    inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
    attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

    with model.maybe_autocast():
        outputs = model.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
    loss = outputs.loss

    return {"loss": loss}