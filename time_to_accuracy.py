import os
import json
import sys
import traceback

def main():
    if len(sys.argv) != 5:
        print("Usage: python time_to_accuracy.py <result_dir> <max_acc> <epoch_cutoff> <base_time>")
        return
    result_dir = sys.argv[1]
    max_acc = float(sys.argv[2])
    epoch_cutoff = float(sys.argv[3])
    base_time = float(sys.argv[4])
    results = json.load(open(os.path.join(result_dir, 'trainer_state.json'), 'r'))
    log_history = results['log_history']
    epoch_to_traintime = {v['epoch']: v['training_time'] for v in log_history if 'training_time' in v}
    epoch_to_accuracy = {v['epoch']: v['eval_accuracy'] for v in log_history if 'eval_accuracy' in v}
    epoch_to_traintime = {k: v for k, v in epoch_to_traintime.items() if k in epoch_to_accuracy}
    epoch_to_accuracy = {k: v for k, v in epoch_to_accuracy.items() if k in epoch_to_traintime}
    sorted_epoch_to_traintime = sorted(epoch_to_traintime.items(), key=lambda x: x[0])
    
        
    if os.path.exists(os.path.join(result_dir, 'post_distillation_model', 'eval_results.json')):
        post_distill_eval = json.load(open(os.path.join(result_dir, 'post_distillation_model', 'eval_results.json'), 'r'))
        post_distill_epoch = post_distill_eval['epoch']
        # Find the nearest epoch to post_distill_epoch that has training time
        for epoch, train_time in sorted_epoch_to_traintime:
            if epoch < epoch_cutoff:
                continue
            if epoch > post_distill_epoch:
                post_distill_epoch = epoch
                break
        best_distill_acc = json.load(open(os.path.join(result_dir, 'post_distillation_model', 'best_eval_results.json'), 'r'))['eval_accuracy']
        best_distill_epoch = None
        for epoch, train_time in sorted_epoch_to_traintime:
            if epoch < epoch_cutoff:
                continue
            if epoch_to_accuracy[epoch] >= best_distill_acc:
                best_distill_epoch = epoch
                break
        print("Best distillation epoch: {}".format(best_distill_epoch))
    else:
        post_distill_epoch = float('inf')
        best_distill_epoch = None
    
    print("Post distillation epoch: {}".format(post_distill_epoch))
    # Find the first epoch that reaches different accuracy levels
    ratios = [0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999, 1.0]
    acc_levels = [round(max_acc * ratio, 3) for ratio in ratios]
    results = {}
    for level, ratio in zip(acc_levels, ratios):
        for epoch, train_time in sorted_epoch_to_traintime:
            if epoch < epoch_cutoff:
                continue
            if round(epoch_to_accuracy[epoch], 3) >= level:
                try:
                    if epoch < post_distill_epoch:
                        results[ratio] = epoch_to_traintime[epoch] + base_time
                    else:
                        if epoch not in epoch_to_traintime:
                            continue
                        results[ratio] = epoch_to_traintime[best_distill_epoch] + epoch_to_traintime[epoch] - epoch_to_traintime[post_distill_epoch] + base_time
                    print("Epoch {} reaches {}% max accuracy in {} seconds".format(epoch, ratio * 100, results[ratio]))
                except Exception as e:
                    print(e, epoch, best_distill_epoch, post_distill_epoch)
                break
    json.dump(results, open(os.path.join(result_dir, 'time_to_accuracy.json'), 'w'), indent=4)
    
    tta_history = []
    best_acc = 0
    print(len(sorted_epoch_to_traintime))
    for epoch, train_time in sorted_epoch_to_traintime:
        if epoch < epoch_cutoff:
            continue
        if epoch_to_accuracy[epoch] > best_acc:
            try:
                if epoch < post_distill_epoch:
                    tta_history.append((epoch_to_traintime[epoch] + base_time, epoch_to_accuracy[epoch]))
                else:
                    if epoch not in epoch_to_traintime:
                        continue
                    tta_history.append((epoch_to_traintime[best_distill_epoch] + epoch_to_traintime[epoch] - epoch_to_traintime[post_distill_epoch] + base_time, epoch_to_accuracy[epoch]))
                ratio = epoch_to_accuracy[epoch] / max_acc
                print("Epoch {} reaches {}% max accuracy in {} seconds".format(epoch, ratio * 100, tta_history[-1][0]))
                best_acc = epoch_to_accuracy[epoch]
            except Exception as e:
                print(e, epoch, best_distill_epoch, post_distill_epoch)
                traceback.print_exc()
    print("Final best accuracy: {}".format(best_acc))
    json.dump(tta_history, open(os.path.join(result_dir, 'tta_history.json'), 'w'), indent=4)

if __name__ == '__main__':
    main()