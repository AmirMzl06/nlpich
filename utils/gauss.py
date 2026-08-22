import numpy as np
import scipy.stats as stats

def smooth_trials(data, bin_size, kernel_type, kernel_SD, sqrt=0):
    binned_spikes = data.T.tolist()
    smoothed = []
    if sqrt == 1:
        for (i, each) in enumerate(binned_spikes):
            binned_spikes[i] = np.sqrt(each)
    kernel_hl = 3 * int(kernel_SD / bin_size)
    normalDistribution = stats.norm(0, kernel_SD)
    x = np.arange(-kernel_hl*bin_size, (kernel_hl+1)*bin_size, bin_size)
    kernel = normalDistribution.pdf(x)
    if kernel_type == 'half_gaussian':
        for i in range(0, int(kernel_hl)):
            kernel[i] = 0
    n_sample = np.size(binned_spikes[0])
    nm = np.convolve(kernel, np.ones((n_sample))).T[int(kernel_hl):n_sample + int(kernel_hl)]
    for each in binned_spikes:
        temp1 = np.convolve(kernel, each)
        temp2 = temp1[int(kernel_hl):n_sample + int(kernel_hl)] / nm
        smoothed.append(temp2)
    return np.asarray(smoothed).T