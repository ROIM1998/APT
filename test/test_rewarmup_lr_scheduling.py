import seaborn as sns
from matplotlib import pyplot as plt

num_epochs = 20
epoch_size = 3068
reset_epochs = [2, 5, 8, 11, 14]
reset_steps = [epoch * epoch_size for epoch in reset_epochs]
num_training_steps = num_epochs * epoch_size
num_warmup_steps = 0.06 * num_training_steps


if __name__ == '__main__':
    steppoints = []
    if not reset_steps[0] == 0:
        reset_steps = [0] + reset_steps
    warmup_starts = set(reset_steps)
    for step in reset_steps:
        steppoints.append(step)
        steppoints.append(step + num_warmup_steps)
    steppoints.append(num_training_steps)

    # Determine which range an integer belongs to using binary search
    def find_range(n):
        for idx, step in enumerate(steppoints):
            if step <= n < steppoints[idx + 1]:
                if step in warmup_starts:
                    return step, steppoints[idx + 1], True # is warmup
                else:
                    return step, steppoints[idx + 1], False # is not warmup

    def lr_lambda(current_step: int):
        range_start, range_end, is_warmup = find_range(current_step)
        if is_warmup:
            return float(current_step - range_start) / float(max(1, range_end - range_start))
        else:
            return max(
                0.0, float(range_end - current_step) / float(max(1, range_end - range_start))
            )
    
    steps = list(range(num_training_steps))
    lrs = [lr_lambda(step) for step in steps]
    sns.lineplot(x=steps, y=lrs)
    plt.savefig('lr_test.png')