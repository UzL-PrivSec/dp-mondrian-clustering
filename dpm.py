import numpy as np
from fast_histogram import histogram1d
from diffprivlib.tools import percentile
from diffprivlib.mechanisms import Exponential, Laplace, GaussianAnalytic


class DPM:
    def __init__(
        self,
        data,
        bounds,
        epsilon,
        delta,
        *,
        epsilon_scale_bandswidth=0.04,
        epsilon_scale_count=0.18,
        epsilon_scale_exp=0.18,
        epsilon_scale_average=0.6,
        t=0.3,
        q=1 / 12,
        **unused_kwargs
    ):

        self.data = data
        self.epsilon_bandswidth = epsilon * epsilon_scale_bandswidth
        self.epsilon_count = epsilon * epsilon_scale_count
        self.epsilon_exp = epsilon * epsilon_scale_exp
        self.epsilon_average = epsilon * epsilon_scale_average
        self.bounds = bounds
        self.t = t
        self.q = q

        self.w_c = 1.0
        self.w_e = 5.0
        self.num_split_levels = 7
        self.delta_count = delta * 0.2 / (self.num_split_levels + 1)
        self.delta_average = delta * 0.8
        self.max_norm = (
            self.bounds[1] * np.sqrt(data.shape[1])
            if np.abs(self.bounds[1]) > np.abs(self.bounds[0])
            else -self.bounds[0] * np.sqrt(data.shape[1])
        )

        self.epsilon_count_per_split = (
            self.epsilon_count
            * np.sqrt(2 ** np.arange(0, self.num_split_levels + 1))
            / np.sum(np.sqrt(2 ** np.arange(0, self.num_split_levels + 1)))
        )

        self.epsilon_exp_per_split = (
            self.epsilon_exp
            * np.sqrt(2 ** np.arange(0, self.num_split_levels))
            / np.sum(np.sqrt(2 ** np.arange(0, self.num_split_levels)))
        )

        self.num_intervals = None
        self.interval_size = None
        self.split_limit = None
        self.clusters = []
        self.cluster_noisy_count = []

    def perform_clustering(self):

        self.clusters = []
        self.cluster_noisy_count = []

        n = self.data.shape[0]

        noisy_count = self._laplace_mechanism(0, n)

        self.split_limit = noisy_count / 2**self.num_split_levels
        self.interval_size = self._interval_size_estimation(noisy_count)
        self.num_intervals = int((self.bounds[1] - self.bounds[0]) / self.interval_size)

        self._build_clustering(np.arange(n), "", 0, noisy_count)

        centres = self._averaging()

        return centres, self.clusters

    def _interval_size_estimation(self, noisy_count):

        def emp_sigma_lookup(n, q):

            max_sigma = 30
            sigma_step = 0.5
            num_iters = 30

            sigmas = np.arange(sigma_step, max_sigma, sigma_step)
            codebook = np.zeros((len(sigmas), 2))

            for idx_i, s in enumerate(sigmas):
                samples = np.random.normal(loc=0, scale=s, size=(n, num_iters))
                samples = np.sort(samples, axis=0)
                distances = samples[1:] - samples[:-1]
                distance_quantiles = np.percentile(distances, q, axis=0)
                distance_quantile_iter = np.average(distance_quantiles)
                codebook[idx_i] = [distance_quantile_iter, s]

            return codebook

        q = 65

        rounded_noisy_count = int(noisy_count)
        codebook = emp_sigma_lookup(rounded_noisy_count, q)

        x = np.sort(self.data, axis=0)

        distances_mean = np.sum(x[1:] - x[:-1], axis=1) / self.data.shape[1]

        try:
            approx_percentile = percentile(
                distances_mean,
                q,
                epsilon=0.5 * self.epsilon_bandswidth,
                bounds=(0, self.bounds[1] - self.bounds[0]),
            )  # * 0.5 because sens is 2 instead of 1
        except:
            return 0  # Bug in percentile for uci_letters

        found_index_quantile = np.argmin(np.abs(codebook[:, 0] - approx_percentile))
        pred_sigma_q = codebook[found_index_quantile][-1] * 0.5

        return pred_sigma_q

    def _build_clustering(self, current, subset_id, split_level, noisy_count):

        if split_level >= self.num_split_levels:
            self.clusters.append(current)
            self.cluster_noisy_count.append(noisy_count)
            return

        pos, dim = self._split(current, split_level, noisy_count)

        vec_normal = np.zeros(self.data.shape[1])
        vec_normal[dim] = 1

        mask = (np.matmul(self.data[current], vec_normal) - pos) <= 0

        subset_0 = current[np.flatnonzero(1 - mask == 1)]
        subset_1 = current[np.flatnonzero(mask == 1)]

        noisy_count_0 = self._laplace_mechanism(split_level + 1, subset_0.shape[0])
        noisy_count_1 = self._laplace_mechanism(split_level + 1, subset_1.shape[0])

        if noisy_count_0 <= self.split_limit or noisy_count_1 <= self.split_limit:
            self.clusters.append(current)
            self.cluster_noisy_count.append(noisy_count)
            return

        self._build_clustering(
            subset_0, subset_id + "0", split_level + 1, noisy_count_0
        )
        self._build_clustering(
            subset_1, subset_id + "1", split_level + 1, noisy_count_1
        )

    def _split(self, current, split_level, noisy_count):

        emptiness_scores, centreness_scores = self._compute_scores(current, noisy_count)

        scores = centreness_scores * self.w_c + emptiness_scores * self.w_e

        dimension, position = self._exponential_mechanism(
            noisy_count, split_level, scores
        )

        return position, int(dimension)

    def _compute_scores(self, current, noisy_count):
        d = self.data[current].shape[1]

        emptiness_scores = np.zeros(self.num_intervals * d)
        centreness_scores = np.zeros(self.num_intervals * d)

        if noisy_count <= 0:
            return emptiness_scores, centreness_scores

        argsort_data = np.argsort(self.data[current], axis=0)

        intervals_left_border = np.arange(
            self.bounds[0],
            self.bounds[1] + self.interval_size,
            self.interval_size,
        )

        for dim in range(d):
            dim_data = self.data[current][:, dim]

            if len(intervals_left_border) > self.num_intervals:
                intervals_left_border = intervals_left_border[: self.num_intervals + 1]

            empt_scores = self._emptiness(dim_data, intervals_left_border, noisy_count)
            cent_scores = self._centerness(
                dim_data,
                intervals_left_border,
                argsort_data[:, dim],
                noisy_count,
            )

            prev_i = dim * self.num_intervals
            next_i = (dim + 1) * self.num_intervals

            emptiness_scores[prev_i:next_i] = empt_scores
            centreness_scores[prev_i:next_i] = cent_scores

        return emptiness_scores, centreness_scores

    def _centerness(
        self, dim_data, intervals_left_border, arg_sort_dim_data, noisy_count
    ):
        intervals_center = intervals_left_border[:-1] + self.interval_size / 2

        ps = np.searchsorted(dim_data, intervals_center, sorter=arg_sort_dim_data)

        @np.vectorize
        def c(n, q, t, p):
            if abs(n / 2 - p) >= n / 2 - n * q:
                return ((n / 2 - abs(p - n / 2)) * t) / (n * q)

            else:
                return (t - 2 * q) / (1 - 2 * q) + (
                    (n / 2 - abs(p - (n / 2))) * (1 - t)
                ) / (n / 2 - n * q)

        return c(noisy_count, self.q, self.t, ps)

    def _emptiness(self, dim_data, intervals_left_border, noisy_count):

        if noisy_count <= 0:
            return 1

        num_points_per_interval = histogram1d(
            dim_data, bins=len(intervals_left_border) - 1, range=self.bounds
        )

        emptiness_score = 1 - (num_points_per_interval * 1 / noisy_count)

        emptiness_score[emptiness_score < 0] = 0
        emptiness_score[emptiness_score > 1] = 1

        return emptiness_score

    def _averaging(self):

        num_dims = self.data.shape[1]

        dpm_centers = np.zeros((len(self.clusters), num_dims))

        for i, ids in enumerate(self.clusters):

            if len(ids) <= 0:
                dpm_centers[i] = np.zeros(num_dims)
                continue

            non_dpm_center = np.sum(self.data[ids], axis=0)

            for j in range(num_dims):
                dpm_centers[i][j] = GaussianAnalytic(
                    epsilon=self.epsilon_average,
                    delta=self.delta_average,
                    sensitivity=self.max_norm,
                ).randomise(non_dpm_center[j])
            dpm_centers[i] /= self.cluster_noisy_count[i]

        return dpm_centers

    def _exponential_mechanism(self, noisy_count, split_level, scores):

        n_split_noisy = noisy_count + self._shifted_laplace_offset(split_level)
        n_split_noisy = max(n_split_noisy, 1)

        delta_u = self.w_c * ((self.t / self.q) / n_split_noisy) + self.w_e * (
            1 / n_split_noisy
        )

        exp_mech = Exponential(
            epsilon=self.epsilon_exp_per_split[split_level],
            sensitivity=delta_u,
            utility=scores.flatten().tolist(),
            monotonic=False,
        )

        selected_index = exp_mech.randomise()

        d = self.data.shape[1]

        dimension = int(
            (selected_index % (self.num_intervals * d)) // self.num_intervals
        )

        pos_index = int(
            (selected_index % (self.num_intervals * d)) % self.num_intervals
        )

        position = (
            self.bounds[0] + pos_index * self.interval_size + self.interval_size / 2
        )

        return dimension, position

    def _laplace_mechanism(self, split_level, n):
        mechanism = Laplace(
            epsilon=self.epsilon_count_per_split[split_level], sensitivity=1
        )
        n_t = mechanism.randomise(n)
        return n_t

    def _shifted_laplace_offset(self, split_level):
        return np.log(2 * self.delta_count) / self.epsilon_count_per_split[split_level]
