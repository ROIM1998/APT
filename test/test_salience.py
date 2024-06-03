import torch

salience = torch.load('output/bert-base-uncased_lora_minus_rte_cubic_gradual_running_fisher_alloc_running_fisher_momentum_mapping_static_teacher_dynamic_cofi_student_distill_tophalf_limited_resizing_nonormalize_correctweight_clippedmoving_correctuncertain_bothsquare_freeteacher/mac0.4/epoch120/bz32/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r8/pruning_start-1/distill_epoch96/first_salience.pt', map_location='cpu')
vanilla_score = salience['mask_salience']['intermediate_mask'] * salience['mask_uncertainty']['intermediate_mask']
sorted_score, sorted_idx = vanilla_score.sort(descending=False)
neuron_tuning_score = torch.cat([salience['grafting_mask_salience']['modules'][i]['intermediate']['output_mask']['s'] * salience['grafting_mask_salience']['modules'][i]['intermediate']['output_mask']['u'] for i in range(12)])
sorted_tuning_score, sorted_tuning_idx = neuron_tuning_score.sort(descending=False)

combined_score = vanilla_score * neuron_tuning_score
sorted_combined_score, sorted_combined_idx = combined_score.sort(descending=False)

torch.cat([salience['grafting_mask_salience']['modules'][i]['intermediate']['output_mask']['s'] * salience['grafting_mask_salience']['modules'][i]['intermediate']['output_mask']['u'] for i in range(12)]).mean()

for i in range(12):
    print

for i in range(12):
    print((salience['grafting_mask_salience']['modules'][i]['value']['bottleneck_mask']['s'] * salience['grafting_mask_salience']['modules'][i]['value']['bottleneck_mask']['u']).mean())