from typing import Callable, Any, Union

import numpy as np
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt
import pandas as pd
import logging


class Core(object):
    @staticmethod
    def vis_plot(x, x_d, logprob, ylabel='Percentage/PDF', title='', vlines=[], vline_colors=[], vline_labels=[], hist=True,
                 reverse=False):
        """
        plot probability density function (PDF) and histogram for comparison
        """
        if hist:
            pd.DataFrame(x).rename(columns={0: 'Histogram'}).plot.hist(bins=20, alpha=0.5, color="goldenrod",
                                                                       **{"density": True},
                                                                       figsize=(18, 16)
                                                                       )
        else:
            plt.figure(figsize=(8, 2), dpi=100)
        plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label='Probability Density Function')
        # plt.plot(x, np.full_like(x, -0.005), '|k', markeredgewidth=1)
        for vl, c, l in zip(vlines, vline_colors, vline_labels):
            plt.axvline(x=vl, color=c, linestyle='--', label=f'{l}:{vl}')

        if len(vlines) > 0:
            if not reverse:
                plt.axvspan(vlines[-1], x_d[-1], facecolor='b', alpha=0.05)
            else:
                plt.axvspan(x_d[0], vlines[-1], facecolor='b', alpha=0.05)

        plt.legend()
        plt.ylabel('')
        plt.title(title)
        plt.yticks([])
        # plt.ylim((0, .03))
        # plt.xlim((10, 40))

    @staticmethod
    def grid_search_bandwith(x):
        """
        search for the best bandwith for KDE
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import KFold

        bandwidths = np.linspace(0, 1, 20)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=KFold(5)
                            )
        grid.fit(x);
        return grid.best_params_['bandwidth']

    @staticmethod
    def get_transformer(X):
        """
        tried for transform input data before KDE
        - didn't work, not in use
        """
        from sklearn.preprocessing import PowerTransformer
        from sklearn.preprocessing import Normalizer
        # bc = PowerTransformer(method="box-cox")
        yj = PowerTransformer(method="yeo-johnson")
        X_trans_bc = yj.fit(X)
        # norm_t = Normalizer()
        # norm_t = norm_t.fit(X_trans_bc.transform(X))
        return X_trans_bc
        # .transform(X)
        # logging.info(min(X_trans_bc), max(X_trans_bc), np.std(X_trans_bc))
        # return X_trans_bc

    @staticmethod
    def get_threshold_index(threshold, low_bound, is_discrete, prev_val_offset, step, boundary_offset):
        """
        Calculate threshold value index
        """
        threshold_index = int((threshold - low_bound) / step)
        if threshold == low_bound + boundary_offset:
            logging.info(f'adjust boundary {threshold}')
            threshold_index = int((threshold - low_bound - boundary_offset) / step)
        elif is_discrete:
            # discrete values will lead to PDF shape like pulses,
            # it's important to start with the valley between the pulse you want to include and the one before that
            # the following is to find the valley index
            threshold_index_prev = int((threshold - prev_val_offset - low_bound) / step)
            threshold_index = int(threshold_index_prev + (threshold_index - threshold_index_prev) / 2)
        return threshold_index

    @staticmethod
    def search_for_zero_mass_index(kde, min_v, n_samples=100):
        """
        Search near zero probability mass for boundary adjustment
        """
        bins = np.linspace(min_v - 10, min_v, n_samples)
        pd_vals = np.exp(kde.score_samples(bins.reshape(-1, 1)))
        first_zero_idx = np.max(np.where(pd_vals < 1e-10))
        return first_zero_idx, bins[first_zero_idx]

    @staticmethod
    def kde_estimate(X, bandwidth=1, kernel='gaussian', search_bandwidth=True):
        """
        Kernel density estimation to get probability
        """
        # KDE estimate
        from sklearn.neighbors import KernelDensity
        best_bw = bandwidth
        if search_bandwidth:
            best_bw = Core.grid_search_bandwith(X)
            logging.info(f'learned best bandwidth {best_bw}')

        kde = KernelDensity(bandwidth=best_bw, kernel=kernel)
        kde.fit(X)

        return kde, best_bw

    @staticmethod
    def deterioration_index(X, low_bound, up_bound, threshold,
                            n_samples=10000, plot_title='',
                            is_discrete=False, prev_discrete_value_offset=1,
                            weight_sum_steps=10, reverse=False,
                            bandwidth=1, kernel='gaussian', search_bandwidth=True,
                            do_plot=True
                            ):
        """
        obtain deterioration index
        X - the random sample of measurements
        low_bound/up_nbound - the boundary values of the measurement
        n_samples - number of bins to use for probability calculation. default is 2000.
        plot_title - the title of the plot, if generates plot. default is empty string
        is_discrete - whether the random sample is discrete. NB: this might be overwritten based on bandwidth learned.
          Small bandwidths will always bring out pulse like PDFs. default is False.
        prev_discrete_value_offset - the difference between the threshold and the previous legitimate value. default is 1.
        weight_sum_steps - the number of bins for weighted sum of k-step cutoffs, default is 20
        reverse - for calculating p(X<threshold), i.e., the smaller the measure value the more severe a patient is.
          default is False
        bandwidth - default bandwidth to use if not search bandwidth, default is 1
        kernel - the kernel to use for KDE, default is gaussian.
        search_bandwidth - whether to use grid search to find optimal bandwidth for X. default is True
        do_plot - whether to generate plots, default is True
        """

        # estimate density function
        kde, fitted_bandwith = Core.kde_estimate(X, bandwidth=bandwidth, kernel=kernel,
                                                 search_bandwidth=search_bandwidth)

        # detect pulse like PDF
        if fitted_bandwith < .1:
            # force is_discrete because the bestfitted_bandwith_bw would lead to one
            logging.info(f'FORCED to be discrete because the bandwidth is {fitted_bandwith}')
            is_discrete = True
        elif fitted_bandwith > .7:
            # force is_discrete to be false because the fitted_bandwith would lead to one
            logging.info(f'FORCED to be NOT discrete because the bandwidth is {fitted_bandwith}')
            is_discrete = False

        # automatically adjust on boundaries
        _, adjusted_min = Core.search_for_zero_mass_index(kde, low_bound)
        logging.info(f'adjusted min val {adjusted_min}')
        boundary_offset = low_bound - adjusted_min
        orig_low_bound = low_bound
        low_bound -= boundary_offset
        up_bound += boundary_offset

        # use learned KDE estimator to get probability
        bins = np.linspace(low_bound, up_bound, n_samples)
        kd_vals = kde.score_samples(bins.reshape(-1, 1))  # Get PDF values for each x
        step_width = (up_bound - low_bound) / n_samples  # get the step
        prob = np.exp(kd_vals) * step_width  # get the approximate prob at each point using the integral of the PDF

        if do_plot:
            tidx = Core.get_threshold_index(threshold, low_bound, is_discrete, prev_discrete_value_offset, step_width,
                                            boundary_offset=boundary_offset)
            Core.vis_plot(X, bins, kd_vals, title=plot_title,
                          vlines=[threshold, round(tidx * step_width + low_bound, 2)], vline_colors=['r', 'b'],
                          vline_labels=['Threshold', 'Boudary-adjusted'],
                          hist=False, reverse=reverse)

        # severity quantification

        if reverse:
            s = low_bound
            e = min(threshold, up_bound)
        else:
            s = max(threshold, low_bound)
            e = up_bound
        # 1. binary like multimorbidity num > 3, yes or no
        sq1 = Core.stepped_severity(prob, s, e, 1, low_bound, step_width, is_discrete, boundary_offset,
                                    prev_val_offset=prev_discrete_value_offset, reverse=reverse)
        # 2. stepped quantification that considers higher/lower the value, more severe the patients are
        sqs = Core.stepped_severity(prob, s, e, weight_sum_steps, low_bound, step_width, is_discrete, boundary_offset,
                                    prev_val_offset=prev_discrete_value_offset, reverse=reverse)

        return {'overall-prob': round(prob.sum(), 4), 'one-step': round(sq1, 4), 'k-step': round(sqs, 6), '|X|': len(X)}

    @staticmethod
    def stepped_severity(probs, s, e, steps, low_bound, step_width, is_discrete, boundary_offset, prev_val_offset=1,
                         reverse=False) -> float:
        """
        To quantify severity by considering higher values as more severe.
        This is done by weighted sum by integrating probs from a threshold - s
        """

        bins = np.linspace(s, e, steps)
        # bin_probs = [probs[get_threshold_index(t, low_bound, is_discrete, step_width):].sum()  for t in bins]
        bin_probs = []
        for i, t in enumerate(bins):
            idx1 = Core.get_threshold_index(t, low_bound, is_discrete, prev_val_offset, step_width, boundary_offset)
            if i < len(bins) - 1:
                idx2 = Core.get_threshold_index(bins[i + 1], low_bound, is_discrete, prev_val_offset, step_width,
                                                boundary_offset)
                bin_probs.append(probs[idx1:idx2].sum())
            else:
                bin_probs.append(probs[idx1:].sum()) if not reverse else bin_probs.append(
                    probs[idx1:Core.get_threshold_index(e,
                                                        low_bound,
                                                        is_discrete,
                                                        prev_val_offset,
                                                        step_width,
                                                        boundary_offset)].sum())
        s = 0

        # weight functions
        # weight_function = lambda x: log(x+2, 2)
        weight_function: Callable[[Any], Union[int, Any]] = lambda x: x + 1

        w = 0
        for i, p in enumerate(bin_probs):
            if not reverse:
                s += weight_function(i) * p
            else:
                s += weight_function(len(bin_probs) - i) * p
            w += weight_function(i)
        return s / w

    @staticmethod
    def db_ineq(di1, di2) -> float:
        """
        quantify inequality
        """
        return di1['k-step'] / di2['k-step'] - 1


class Util(object):
    @staticmethod
    # get the random sample for obtaining deterioration index
    def get_random_sample(df, feature, feature_gen_fun=None):
        if feature_gen_fun is not None:
            X = df.apply(feature_gen_fun, axis=1).to_numpy().reshape(-1, 1)
        else:
            X = df[feature].to_numpy().reshape(-1, 1)
        return X

    @staticmethod
    def compare_two_groups(df1, df2, feature, cohort_name1, cohort_name2, di_label, threshold,
                           bandwidth=1, is_discrete=False, search_bandwidth=True, do_plot=True,
                           feature_gen_fun=None, reverse=False
                           ):
        """
        # obtain the inequality quantification
        df1 - data frame of the first group
        df2 - data frame of the second group
        feature - the feature name of the measurement, not used if feature_gen_fun is not none
        cohort_name1 - the name of the first cohort to be displayed on the plot
        cohort_name2 - the name of the second cohort to be displayed on the plot
        di_label - the label of the deterioration index to be displayed on the plot
        threshold - the threshold for abnormality starting point. This could be a list
          if two cohorts require different thresholds. For example, male/female might have
          different normal ranges for some measurements. The first element will be used for the first one.
        """
        X1 = Util.get_random_sample(df1, feature, feature_gen_fun=feature_gen_fun)
        X2 = Util.get_random_sample(df2, feature, feature_gen_fun=feature_gen_fun)
        # it is very important to use the same min/max values as the k-step weighting needs to
        # put the same weight for the same level of deterioration
        min_v = min([np.min(X1), np.min(X2)])
        max_v = max([np.max(X1), np.max(X2)])

        threshold1 = threshold
        threshold2 = threshold
        if type(threshold) is list:
            threshold1 = threshold[0]
            threshold2 = threshold[1]
        c1_di = Core.deterioration_index(X1, min_v, max_v, reverse=reverse, threshold=threshold1, bandwidth=bandwidth,
                                         is_discrete=is_discrete,
                                         plot_title=f'{cohort_name1}| {di_label}', search_bandwidth=search_bandwidth,
                                         do_plot=do_plot)
        c2_di = Core.deterioration_index(X2, min_v, max_v, reverse=reverse, threshold=threshold2, bandwidth=bandwidth,
                                         is_discrete=is_discrete,
                                         plot_title=f'{cohort_name2}| {di_label}', search_bandwidth=search_bandwidth,
                                         do_plot=do_plot)
        ineq = Core.db_ineq(c1_di, c2_di)
        # print(f'{cohort_name1} vs {cohort_name2} inequality on {di_label} is {ineq:.2%}')
        return c1_di, c2_di, ineq

    @staticmethod
    def area(w_data):
        """
        calculate the area under curve - do NOT do interpolation
        """
        prev = None
        area = 0
        decision_area = 0
        n_points = 0
        for r in w_data:
            if prev is not None:
                a = (r[1] + prev[1]) * (r[0] - prev[0]) / 2  # * r[2]
                area += a
                if prev[0] >= 0.5:
                    decision_area += a
                    n_points += 1
            prev = r

        if prev is not None:
            a = (r[1] + prev[1]) * (r[0] - prev[0]) / 2  # * r[2]
            area += a
            if prev[0] >= 0.5:
                decision_area += a
                n_points += 1

        return area, decision_area

    @staticmethod
    def vis_DA_indices(data, label):
        """
        plot dot-line for approximating a DA curve
        """
        w_data = data[np.where(data[:, 1] > 0)][:, [0, 2, 1]]
        a, decision_area = Util.area(w_data)
        plt.plot(w_data[:, 0], w_data[:, 1], '-')
        plt.plot(w_data[:, 0], w_data[:, 1], 'o', label=label)
        return a, decision_area, w_data

    @staticmethod
    def viz(d1, d2, g1_label, g2_label, deterioration_label, allocation_label, config):
        """
        do DA curve visualisation
        """
        if 'style' in config:
            plt.style.use(config['style'])
        font_size = config['font_size'] if 'font_size' in config else 12
        if 'fig_size' in config:
            fig = plt.figure(figsize=config['fig_size'], dpi=200)

        # do some clearning: remove those empty points
        d1 = np.delete(d1, np.where(d1[:, 1] == 0), axis=0)
        d2 = np.delete(d2, np.where(d2[:, 1] == 0), axis=0)
        # make two datasets even in terms of max x val
        x_min = min(np.max(d1[:, 0]), np.max(d2[:, 0]))
        d1 = np.delete(d1, np.where(d1[:, 0] > x_min), axis=0)
        d2 = np.delete(d2, np.where(d2[:, 0] > x_min), axis=0)

        # automatically set x/y limits for better viz
        # x_max = max(np.max(d1[:, 0]), np.max(d2[:, 0]))
        y_max = max(np.max(d1[:, 2]), np.max(d2[:, 2]))

        plt.xlim(0, x_min * 1.05)
        plt.ylim(0, y_max * 1.05)

        # do plots
        a1, da1, _ = Util.vis_DA_indices(d1, g1_label)
        a2, da2, _ = Util.vis_DA_indices(d2, g2_label)

        # generate output
        # print('{0}\t{1:.2%}\t{2:.2%}\t{3:.2%}'.format(deterioration, white_d_ratio, non_white_d_ratio,
        #                                      (non_white_d_ratio - white_d_ratio)/white_d_ratio))
        print('AUC\t{0:.6f}\t{1:.6f}\t{2:.2%}'.format(a1, a2, (a2 - a1) / a1))
        print('Decision AUC\t{0:.6f}\t{1:.6f}\t{2:.2%}'.format(da1, da2, (da2 - da1) / da1))

        # figure finishing up
        plt.xlabel(allocation_label, fontsize=font_size)
        plt.ylabel(deterioration_label, fontsize=font_size)

        # plot decision region
        plt.plot([0.5, 0.5], [0, 1], '--', lw=.8, color='g')
        plt.axvspan(0.5, 1, facecolor='b', alpha=0.1)

        plt.legend(fontsize=font_size, loc='best')