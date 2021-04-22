from bertviz import head_view


def show_head_view(model, tokenizer, sentence_1, sentence_2, layer=None, heads=None):
    """Visualize attention head of BERTology models

    """
    inputs = tokenizer.encode_plus(
        sentence_1, sentence_2, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention = model(input_ids=input_ids,
                      token_type_ids=token_type_ids).attentions
    sentence_b_start = token_type_ids[0].tolist().index(1)
    input_id_list = input_ids[0].tolist() 
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    head_view(attention, tokens, sentence_b_start, layer=layer, heads=heads)
