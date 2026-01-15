from circuits.analysis.process_circuits import (
    ADAGConfig,
    get_all_pairs_cl_ja_effects_with_attributions,
    prepare_cis,
)
from util.subject import Subject, llama31_8B_instruct_config

subject = Subject(llama31_8B_instruct_config)

prompts = [
    "What is the capital of the state containing the city of Dallas?",
]
seed_responses = ["Answer:"] * len(prompts)
k = 5

cis, attention_masks, focus_tokens, keep_pos, starts = prepare_cis(
    subject,
    subject.tokenizer,
    questions=prompts,
    seed_responses=seed_responses,
    k=k,
    verbose=True,
)
node_scores, embed_scores = get_all_pairs_cl_ja_effects_with_attributions(
    subject,
    cis,
    config=ADAGConfig(
        device="cuda:0",
        verbose=True,
        parent_threshold=None,
        edge_threshold=0.01,
        node_attribution_threshold=None,
        topk=None,
        batch_aggregation="any",
        topk_neurons=500,
        use_relp_grad=True,
        use_shapley_grad=False,
        disable_half_rule=False,
        disable_stop_grad=False,
        ablation_mode="zero",
        use_stop_grad_on_mlps=True,
        return_nodes_only=False,
        focus_last_residual=False,
        skip_attr_contrib=True,
        center_logits=False,
        ig_steps=None,
        ig_mode="eap-ig-inp",
        return_only_important_neurons=True,  # main difference from CLSO
    ),
    attention_masks=attention_masks,
    focus_logits=focus_tokens,
    src_tokens=keep_pos,
    tgt_tokens=[max(keep_pos) for _ in range(k)],
)

scores_orig = node_scores.flatten()
scores = scores_orig.abs()
embeds_sum = embed_scores.flatten().abs().sum()
# for neurons in [10, 100, 1000, 10000, 100000, 1000000]:
#     # what percentage of the total score is covered by the top k neurons?
#     topk = scores.topk(k=neurons)
#     print(f"Top {neurons} neurons cover {topk.values.sum() / scores.sum():.2%}% of the total score")
#     print(f"    Original vals sum: {scores_orig[topk.indices].sum():.2f}")
#     print(f"    Total logit sum: {embeds_sum:.2f}")
#     # print(f"Top {neurons} neurons cover {topk.values.sum() / embeds_sum:.2%}% of the total embeds")

for cutoff in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
    mask = scores > (cutoff * embeds_sum)
    print(
        f"Cutoff {cutoff} covers {mask.nonzero().numel()} ({mask.nonzero().numel() / scores.numel():.2%}) of the neurons"
    )
