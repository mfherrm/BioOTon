import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit

from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# from cython_kriging2 import simple_krig_var
from .cython_kriging2 import simple_krig_var

# machine learning
from sklearn.model_selection import train_test_split

# plotting
import seaborn as sns
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio

sns.set_style("whitegrid")
pio.renderers.default = "browser"
pio.templates.default = "plotly_white"

# All code in this file is either part of or adapted from https://github.com/julianslz/Fair-Train-Test 

#######################################
# Functions
#######################################
@jit(nopython=True)
def setup_rotmat(c0, cc, ang, nst):
    PI = np.pi
    DTOR = PI / 180.0

    # The first time around, re-initialize the cosine matrix for the variogram
    # structures
    rotmat = np.zeros((4, nst))
    maxcov = c0
    azmuth = (90.0 - ang[0]) * DTOR
    rotmat[0, ...] = np.cos(azmuth)
    rotmat[1, ...] = np.sin(azmuth)
    rotmat[2, ...] = -1 * np.sin(azmuth)
    rotmat[3, ...] = np.cos(azmuth)

    maxcov += np.sum(cc)

    return rotmat, maxcov


def discretize(array, bins, rango=None):
    """
    Function to discretize the kriging variance into bins.

    :param array:
    :type array: ndarray
    :param bins:
    :type bins: int
    :param rango:
    :type rango: list
    :return:
    """
    hist, bin_edges = np.histogram(array, bins=bins, range=rango)
    probability = hist / len(array)
    return probability, bin_edges


def kriging_variance_rw(rw_dict_model):
    """
    Compute the kriging variance at the test locations.
    :param rw_dict_model:
    :return:
    """
    test_kvar = simple_krig_var(**rw_dict_model)
    return test_kvar


def pdf_distance(pdf_realizations, target_pdf):
    """
    Computes the Wasserstein distance, Jensen-Shannon distance, and MSE between two kriging variance distributions.
    :param pdf_realizations:
    :param target_pdf: Array. A numpy array that holds the kriging variance distribution of the real-world
    (target distribution)
    :return: Array. A 2D array of size [3xn], where the three stands for WS, JS, MSE distances. On the other hand, n
    stands for the number of samples kvar_array has
    """
    divergence = None
    # 2 JS, MSE, and n realizations each
    if isinstance(pdf_realizations, list):
        divergence = np.zeros((2, len(pdf_realizations)))

        with tqdm(total=len(pdf_realizations)) as pbar:
            for realization in range(len(pdf_realizations)):
                fair_distance = GetDivergence(
                    pdf_realizations[0].to_numpy()[:, -1],
                    target_pdf,
                    26,
                    [np.min(pdf_realizations[0].to_numpy()[:, -1]), np.max(pdf_realizations[0].to_numpy()[:, -1])]
                )
                divergence[0, realization] = fair_distance.js_divergence()
                divergence[1, realization] = fair_distance.mse_distance()
                pbar.update(1)

    elif isinstance(pdf_realizations, np.ndarray):
        divergence = np.zeros((2, pdf_realizations.shape[1]))

        with tqdm(total=pdf_realizations.shape[1]) as pbar:
            for realization in range(pdf_realizations.shape[1]):
                fair_distance = GetDivergence(
                    pdf_realizations[:, realization],
                    target_pdf,
                    26,
                    [np.min(pdf_realizations[:, realization]), np.max(pdf_realizations[:, realization])]
                )
                divergence[0, realization] = fair_distance.js_divergence()
                divergence[1, realization] = fair_distance.mse_distance()
                pbar.update(1)

    return divergence


def spatial_config_and_kvar(training, rw_set, vmap, xrange, yrange, xdir, ydir):
    """
    Plot the available data for training and real-world use of the model. Furthermore, plot the kriging variance map.
    :param training:
    :param rw_set:
    :param vmap:
    :param xrange:
    :param yrange:
    :return:
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('Original dataset', 'Kriging variance'),
        horizontal_spacing=0.05,
        shared_yaxes=True
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=training[xdir],
            y=training[ydir],
            name='Available data',
            marker_symbol='circle',
            showlegend=True,
            marker=dict(
                color='LightSkyBlue',
                size=10,
                line=dict(
                    color='black',
                    width=1
                )
            ),
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=rw_set[xdir],
            y=rw_set[ydir],
            name='Planned real-world use of the model',
            marker_symbol='diamond',
            showlegend=True,
            marker=dict(
                color='firebrick',
                size=10,
                line=dict(
                    color='black',
                    width=1
                )
            ),
        ),
        row=1, col=1
    )

    fig.add_trace(go.Contour(
        z=np.flipud(vmap),
        x=xrange,
        y=yrange,
        colorbar=dict(
            # thickness=25,
            title="Kriging variance",
            x=1.0
        ),
        zmin=0.0,
        zmax=1.0,
    ), row=1, col=2)

    fig.add_trace(go.Scatter(x=rw_set[xdir], y=rw_set[ydir],
                             mode='markers',
                             marker=dict(color="#9c1733", showscale=False, size=10,
                                         symbol='diamond', line=dict(color='black', width=1),
                                         opacity=1.),
                             showlegend=False),
                  row=1, col=2)

    fig.add_trace(go.Scatter(x=training[xdir], y=training[ydir],
                             mode='markers',
                             marker=dict(color='LightSkyBlue', showscale=False, size=4,
                                         symbol='circle', opacity=0.4),
                             showlegend=False),
                  row=1, col=2)

    for i in range(2):
        fig.update_xaxes(title_text="X (m)", range=[-20, 1020], row=1, col=i + 1, showgrid=False, ticks='')
        fig.update_yaxes(range=[-20, 1020], row=1, col=i + 1, showgrid=False, ticks='')

    fig.update_yaxes(title_text="Y (m)", row=1, col=1)

    fig.update_layout(
        width=900,
        height=500,
        legend=dict(
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.15
        ),
        margin=dict(
            l=0, r=80, t=20, b=0, pad=0
        )
    )

    fig.show()

    return fig


def plot_3_realizations(fair_train, fair_test, rand_train, rand_test, spatial_cv, real_world_set, realiz, xdir, ydir,
                        xmin, ymin, xmax, ymax):
    fig = make_subplots(
        rows=1,
        cols=2,  # 3
        # subplot_titles=('Spatial fair train-test split', 'Validation set approach', 'Spatial cross-validation'),
        subplot_titles=('Spatial fair train-test split', 'Validation set approach'),
        horizontal_spacing=0.02,
        shared_yaxes=True
    )
    column = 1
    sets = ['Training', 'Test', 'Planned RW use']
    colores = ['LightSkyBlue', 'goldenrod', 'firebrick']
    markers = ['circle', 'triangle-up', 'diamond']
    sizes = [8, 10, 10]
    # fair and random plots
    # for train, validation in zip([fair_train[realiz], rand_train[realiz]], [fair_test[realiz], rand_test[realiz]]):
    for train, validation in zip([fair_train[realiz], rand_train[realiz]], [fair_test[realiz], rand_test[realiz]]):
        datasets = [train, validation, real_world_set]
        if column == 1:
            legendshow = False
        else:
            legendshow = True

        for i, set_i in enumerate(datasets):
            fig.add_trace(
                go.Scatter(
                    mode='markers',
                    x=set_i[xdir],
                    y=set_i[ydir],
                    name=sets[i],
                    marker_symbol=markers[i],
                    showlegend=legendshow,
                    marker=dict(
                        color=colores[i],
                        size=sizes[i],
                        line=dict(
                            color='black',
                            width=1
                        )
                    ),
                ),
                row=1, col=column
            )
        column += 1

    # for i in range(5):
    #     train_all_but_one = spatial_cv[0].query("kfold == @i")
    #
    #     fig.add_trace(
    #         go.Scatter(
    #             mode='markers',
    #             x=train_all_but_one[xdir],
    #             y=train_all_but_one[ydir],
    #             name='Fold ' + str(i + 1),
    #             marker=dict(
    #                 # color='LightSkyBlue',
    #                 size=8,
    #                 line=dict(
    #                     color='black',
    #                     width=1
    #                 )
    #             ),
    #         ),
    #         row=1, col=3
    #     )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=real_world_set[xdir],
            y=real_world_set[ydir],
            name='Planned RW use',
            marker_symbol='diamond',
            showlegend=False,
            marker=dict(
                color='firebrick',
                size=10,
                line=dict(
                    color='black',
                    width=1
                )
            ),
        ),
        row=1, col=2  # col=3
    )

    # Update axis properties
    for i in range(2):  # 3
        fig.update_xaxes(title_text="X (m)", range=[xmin, xmax], row=1, col=i + 1, showgrid=False, ticks='',
                         showticklabels=False)
        fig.update_yaxes(range=[ymin, ymax], row=1, col=i + 1, showgrid=False, ticks='', showticklabels=False)

    fig.update_yaxes(title_text="Y (m)", row=1, col=1)
    fig.update_layout(
        width=900,
        height=500,
        legend_title_text='<b>Set<b>',
        # legend_title_side="top"
    )

    titulo = "Final set configurations for realization " + str(realiz)
    fig.update_layout(title_text="<b>" + titulo + "<b>", title_x=0.5)

    fig.show()

    return fig


def variogram_arrays(vario):
    nst = vario.get("nst")
    cc = np.zeros(nst)
    aa = np.zeros(nst)
    it = np.zeros(nst)
    ang = np.zeros(nst)
    anis = np.zeros(nst)
    nug = vario.get("nug")
    sill = nug
    cc[0] = vario.get("cc1")
    sill = sill + cc[0]
    it[0] = vario.get("it1")
    ang[0] = vario.get("azi1")
    aa[0] = vario.get("hmaj1")
    anis[0] = vario.get("hmin1") / vario.get("hmaj1")
    if nst == 2:
        cc[1] = vario.get("cc2")
        sill = sill + cc[1]
        it[1] = vario.get("it2")
        ang[1] = vario.get("azi2")
        aa[1] = vario.get("hmaj2")
        anis[1] = vario.get("hmin2") / vario.get("hmaj2")

    return cc, aa, it, ang, anis, sill


class GetDivergence:
    def __init__(self, array_state_obs, array_expected, bins, rango):
        self.array_state_obs = array_state_obs
        self.array_expected = array_expected
        self.bins = bins
        self.rango = rango

    def mse_distance(self):
        """
        Compute the mean squared error distance to compare two distributions.
        :return:
        """
        array_expected = self.array_expected
        bins = self.bins
        rango = self.rango
        array_state_obs = self.array_state_obs

        p, bins_p = discretize(array_expected, bins=bins, rango=rango)
        q, _ = discretize(array_state_obs, bins=bins_p)

        divergence = (np.square(p - q)).mean()

        return divergence

    def kl_divergence(self):
        """
        Compute the Kullback–Leibler divergence between two distributions.
        :return:
        """
        array_expected = self.array_expected
        bins = self.bins
        rango = self.rango
        array_state_obs = self.array_state_obs

        p, bins_p = discretize(array_expected, bins=bins, rango=rango)
        q, _ = discretize(array_state_obs, bins=bins_p)

        divergence = kl_div(p, q)
        divergence = np.where(divergence == np.inf, np.nan, divergence)
        max_diverg = np.nanmax(divergence)
        divergence = np.where(divergence == np.nan, max_diverg, divergence)

        return np.sum(divergence)

    def js_divergence(self):
        """
        Compute the Jensen-Shannon divergence between two distributions.
        :return:
        """
        array_expected = self.array_expected
        bins = self.bins
        rango = self.rango
        array_state_obs = self.array_state_obs

        p, bins_p = discretize(array_expected, bins=bins, rango=rango)
        q, _ = discretize(array_state_obs, bins=bins_p)

        divergence = jensenshannon(p, q)

        return divergence

    def ws_distance(self):
        """
        Compute the Wasserstein distance between two distributions.
        :return:
        """
        array_expected = self.array_expected
        bins = self.bins
        rango = self.rango
        array_state_obs = self.array_state_obs

        p, bins_p = discretize(array_expected, bins=bins, rango=rango)
        q, _ = discretize(array_state_obs, bins=bins_p)

        divergence = wasserstein_distance(p, q)

        return divergence


class SpatialFairSplit:
    def __init__(self, available_data, real_world_set, vario_model, number_bins=15, test_size=0.20, xdir='X',
                 ydir='Y'):
        self.available_data = available_data
        self.rw_set = real_world_set
        self.number_bins = number_bins
        if test_size <= 0 or test_size > 0.25:
            self.test_size = 0.25
        else:
            self.test_size = test_size

        self._xdir = xdir
        self._ydir = ydir

        self._cc, self._aa, self._it, self._ang, self._anis, self.sill = variogram_arrays(vario_model)
        self._nst = vario_model.get('nst')
        self._nug = vario_model.get('nug')

        # Compute the rotational matrices for kriging
        self._rotmat, self._maxcov = setup_rotmat(self._nug, self._cc, self._ang, self._nst)

        # Kriging variance of real world
        rw_dict_model = self._dictionary_assigner(self.available_data, self.rw_set)
        self.rw_krig_var = kriging_variance_rw(rw_dict_model)

        self._weights = np.ones_like(self.rw_krig_var) / len(self.rw_krig_var)
        self._probability, self._bins, _ = plt.hist(
            self.rw_krig_var, bins=number_bins, density=False, weights=self._weights
        )

        kvariance_set = self._kvar_one_at_a_time(self.available_data)
        self.available_data = self.available_data.merge(kvariance_set, how="left", on="UWI")

        # number of samples in training and test sets
        self._test_samples = int(len(self.available_data) * test_size)
        self._train_samples = len(self.available_data) - self._test_samples

    def _dictionary_assigner(self, train_set, test_set):
        """
        It creates a complete dictionary required for kriging variance computation.

        :return: A complete dictionary used for computing the kriging variance.
        """
        dictionary_model = {
            'nst': self._nst, 'ndata': train_set.shape[0], 'nest': test_set.shape[0],
            'anis': self._anis, "cc": self._cc, "aa": self._aa, "it": self._it,
            "ang": self._ang, "nug": self._nug,
            'x_train': train_set[self._xdir].to_numpy(), 'y_train': train_set[self._ydir].to_numpy(),
            'x_test': test_set[self._xdir].to_numpy(), 'y_test': test_set[self._ydir].to_numpy(),
            "rotmat": self._rotmat, "maxcov": self._maxcov
        }

        return dictionary_model

    def _kvar_one_at_a_time(self, dataset):
        """
        Compute the kriging variance one sample at a time with itself.
        :param dataset:
        :return:
        """
        kvariance_set = np.zeros((len(dataset), 2))
        for i, uwi in enumerate(dataset['UWI']):
            test_well = dataset.query("UWI == @uwi")
            train_wells = dataset.query("UWI != @uwi")
            model_dictionary = self._dictionary_assigner(train_wells, test_well)
            # kriging variance of testing
            well_kvar = simple_krig_var(**model_dictionary)
            # save the kriging variance
            kvariance_set[i, 0] = well_kvar
            # save the uwi
            kvariance_set[i, 1] = uwi

        # assign the kriging variance to the correct wells
        kvariance_set = pd.DataFrame(kvariance_set, columns=['kvar', 'UWI'])

        return kvariance_set

    def _get_fair_test_samples(self, dataset, seed):
        """
        Select the samples that have similar kriging variance as the target distribution
        :param dataset:
        :param seed: Seed for reproducibility. Integer
        :return:

        """
        dataset2 = dataset.copy()
        trials = 0
        counter = 0
        fair_test_samples = []
        test_samples = int(len(dataset) * self.test_size)
        np.random.seed(11150 * seed)
        while counter < test_samples or trials > (len(self.available_data)):
            random_bin_index = np.random.randint(0, len(self._probability))
            # the minimum value of kriging variance of the bin
            left_kvar = self._bins[random_bin_index]
            # the maximum value of kriging variance of the bin
            right_kvar = self._bins[random_bin_index + 1]
            # the wells that are inside the kriging variance
            subset_wells_bin = dataset2.query("kvar >= @left_kvar & kvar < @right_kvar")
            # if there are no subsets that fill the condition (e.g., extreme kvar), pass.
            if len(subset_wells_bin) == 0:
                pass
            else:
                # randomly choose a well within that subset
                one_well = subset_wells_bin.sample(n=1)
                # the probability of ocurrence of the chosen bin
                prob_at_bin = self._probability[random_bin_index]
                # draw a random number U~(0, max probability)
                np.random.seed(seed + trials)
                z = np.random.uniform(0, np.max(self._probability))

                if z <= prob_at_bin:
                    one_well_index = one_well.index
                    dataset2.drop(index=one_well_index, inplace=True)
                    dataset2.drop(columns=["kvar"], inplace=True)

                    # store the uwi of the fair test well
                    fair_test_samples.append(one_well['UWI'].to_numpy())
                    # update the kriging variance one-by-one because you removed one well
                    train_kvariance = self._kvar_one_at_a_time(dataset2)
                    # store the computed kriging variance into the dataframe
                    dataset2 = dataset2.merge(train_kvariance, how="left", on="UWI")
                    # update counter
                    counter += 1

            trials += 1
            # pbar.update(1)

        return fair_test_samples

    def _get_multiple_kriging_var(self, sub_train, sub_test):
        """
        Computes the kriging variance of n subtests using n subtrain datasets.
        :param sub_train: List. List containing DataFrames used as known Results for simple kriging.
        :param sub_test: List. List containing DataFrames with the locations where to compute the kriging variance
        :return: Array. An array containing n test samples kriging variances.
        """
        subtest_kvar = np.zeros((len(sub_test[0]), len(sub_test)))
        with tqdm(total=len(sub_test)) as pbar:
            for realization, (subtrain_i, subtest_i) in enumerate(zip(sub_train, sub_test)):
                model_dictionary = self._dictionary_assigner(subtrain_i, subtest_i)
                subtest_kvar[:, realization] = simple_krig_var(**model_dictionary)

                pbar.update(1)

        return subtest_kvar

    def fair_sets_realizations(self, realizations):
        """
        Compute the training and test sets using spatial fair split. Moreover, compute the kriging variance
        of the -n realizations- test sets.
        :param realizations:
        :return:
        """
        test_sets = []
        training_sets = []
        with tqdm(total=realizations) as pbar:
            for i in range(realizations):
                sub_test_uwi = self._get_fair_test_samples(self.available_data, seed=i)
                test_uwi = np.array(sub_test_uwi).tolist()
                test_uwi = [i[0] for i in test_uwi]
                test_sets.append(self.available_data[self.available_data['UWI'].isin(test_uwi)])
                training_sets.append(self.available_data[~self.available_data['UWI'].isin(test_uwi)])

                pbar.update(1)

        # get the kriging variances of all the subtest sets
        test_fair_kvar = self._get_multiple_kriging_var(training_sets, test_sets)

        return training_sets, test_sets, test_fair_kvar

    def _get_n_random_splits(self, realizations):
        """
        Definesn realizations different random tragitin-test-split Results sets.
        :param realizations: Integer. Number of splits to compute.
        :return: List. Two lists of train and test.
        """
        subtrain_random = []
        subtest_random = []
        for realization in range(realizations):
            x_train, y_train, = train_test_split(
                self.available_data,
                test_size=self.test_size,
                random_state=realization
            )
            subtrain_random.append(x_train)
            subtest_random.append(y_train)

        return subtrain_random, subtest_random

    def _get_n_spatial_splits(self, realizations):
        """
        Get n realizations of spatial cross-validation with 5 folds.
        :param realizations:
        :return:
        """
        training_sp = self.available_data.copy()
        subtrain_spatial = []
        scaler = StandardScaler()
        feat_std = scaler.fit_transform(training_sp[[self._xdir, self._ydir]])
        for i in range(realizations):
            kmeans = KMeans(n_clusters=5, random_state=i).fit(feat_std)
            training_sp['kfold'] = kmeans.labels_
            subtrain_spatial.append(training_sp)
        return subtrain_spatial

    def _spatial_kvar(self, spatial_sets):
        """
        Compute the kriging variance of the spatial cross-validation.
        :param spatial_sets:
        :return:
        """
        subtest_kvar = np.zeros((len(spatial_sets[0]), len(spatial_sets)))
        with tqdm(total=len(spatial_sets)) as pbar:
            for realization in range(len(spatial_sets)):
                train_set = spatial_sets[realization]
                kvar_temporal = []
                for kfold in range(5):
                    test_wells = train_set.query("kfold == @kfold")
                    train_wells = train_set.query("kfold != @kfold")
                    model_dictionary = self._dictionary_assigner(train_wells, test_wells)
                    kvar_temporal.append(simple_krig_var(**model_dictionary))

                subtest_kvar[:, realization] = np.concatenate(kvar_temporal).ravel()
                pbar.update(1)

        return subtest_kvar

    def create_other_sets(self, realizations):
        """
        Create n realizations of training and test sets using the validation set approach with random assignment and
        k-fold cross-validation.
        :param realizations:
        :return:
        """
        # compute n random sets
        train_random, test_random = self._get_n_random_splits(realizations)
        # compute the kriging variance of the random tests
        test_kvar_random = self._get_multiple_kriging_var(train_random, test_random)

        # compute n spatial sets
        train_spatial = self._get_n_spatial_splits(realizations)
        # compute the kriging variance of the spatial tests
        test_kvar_spatial = self._spatial_kvar(train_spatial)

        return train_random, test_random, test_kvar_random, train_spatial, test_kvar_spatial


class PublicationImages:
    def __init__(self, test_kvar_random, test_kvar_fair, test_kvar_spatial, rw_kvar):

        self.test_kvar_random = test_kvar_random
        self.test_kvar_fair = test_kvar_fair
        self.test_kvar_spatial = test_kvar_spatial
        self.rw_kvar = rw_kvar

        self.divergence_df = self._compute_divergence()

    def _compute_divergence(self):
        """
        Compute the distance between kvar distribution of target and subtests of fair and random
        :return:
        """
        fair_diverg = pdf_distance(self.test_kvar_fair, self.rw_kvar)
        random_diverg = pdf_distance(self.test_kvar_random, self.rw_kvar)
        diver_df = pd.DataFrame(np.concatenate((fair_diverg.T, random_diverg.T)), columns=['JS', 'MSE'])
        diver_df['Type'] = 'Spatial fair train-test split'
        diver_df.loc[fair_diverg.shape[1]:, 'Type'] = 'Validation set approach'

        return diver_df

    def divergence_violins(self, diver_metrics=None):
        """
        Plot the divergence metrics of the spatial fair train-test split and validation set approach using violing
        plots.
        :param diver_metrics:
        :return:
        """

        if diver_metrics is None:
            diver_metrics = ['Jensen-Shannon', 'Mean squared error']

        metrics = ['JS', 'MSE']

        fig, axs = plt.subplots(1, 2, figsize=(12, 7))
        for dist_metric in range(2):
            sns.violinplot(data=self.divergence_df, x="Type", y=metrics[dist_metric], inner="quartile", cut=0,
                           ax=axs[dist_metric])
            axs[dist_metric].set_title(diver_metrics[dist_metric], fontsize=15)
            axs[dist_metric].set_ylabel("Divergence", fontsize=14)
            axs[dist_metric].set_xlabel("Data split method", fontsize=14)

        for i in range(2):
            axs[i].grid(False)
            axs[i].xaxis.set_ticks_position('bottom')
            axs[i].xaxis.set_tick_params(direction='in')
            axs[i].yaxis.set_ticks_position('left')
            axs[i].yaxis.set_tick_params(direction='in')

        plt.suptitle(
            'Divergence of distribution realizations',
            fontweight='bold',
            fontsize=18,
        )

        plt.show()

        return fig

    def kde_plots(self, max_y):
        """
        Plot all the realizations of the probability distributions of the three cross-validation methods. Include the
        planned real-world use of the model distribution. All the PDFs are kriging variance
        :param max_y:
        :return:
        """

        fig, axs = plt.subplots(1, 2, figsize=(12, 7), sharey=True)  # 1, 3
        colores = (.388, .431, .392, 0.05)  # the last digit is the opacity

        # TODO AttributeError: 'list' object has no attribute 'shape' LINE 730
        for i in range(self.test_kvar_fair.shape[1]):
            sns.kdeplot(
                x=self.test_kvar_fair[:, i],
                ax=axs[0],
                clip=[0, 1],
                linewidth=0.2,
                color=colores
            )

            sns.kdeplot(
                x=self.test_kvar_random[:, i],
                ax=axs[1],
                clip=[0, 1],
                linewidth=0.2,
                color=colores
            )

            # sns.kdeplot(
            #     x=self.test_kvar_spatial[:, i],
            #     ax=axs[2],
            #     clip=[0, 1],
            #     linewidth=0.2,
            #     color=colores,
            # )

            # if i == self.test_kvar_spatial.shape[1] - 1:
            if i == self.test_kvar_random.shape[1] - 1:
                sns.kdeplot(
                    x=self.test_kvar_random[:, i],
                    ax=axs[1],  #[2]
                    clip=[0, 1],
                    linewidth=0.2,
                    label="Test distribution realizations",
                    color=colores
                )

        for row in range(2):  # 2
            sns.kdeplot(
                x=self.rw_kvar,
                ax=axs[row],
                linewidth=3,
                clip=[0, 1],
                label='Planned real-world use of the model distribution',
                color=(.784, .054, 0.172, 1))
            axs[row].set_xlabel('Kriging variance', fontsize=14)

        axs[0].set_title("Spatial fair train-test split", fontsize=14)
        axs[1].set_title("Validation set approach", fontsize=14)
        # axs[2].set_title("Spatial cross-validation", fontsize=14)
        fig.suptitle(
            'Comparison of kriging variance distributions',
            fontsize=18,
            fontweight='bold',
            # fontname="Times New Roman"
        )

        # handles, labels = axs[2].get_legend_handles_labels()
        handles, labels = axs[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2)

        for i in range(2):  # 3
            axs[i].set_ylim([0, max_y])
            axs[i].set_xlim([0, 1])
            axs[i].grid(False)
            axs[i].xaxis.set_ticks_position('bottom')
            axs[i].xaxis.set_tick_params(direction='in')
            axs[i].yaxis.set_ticks_position('left')
            axs[i].yaxis.set_tick_params(direction='in')

        # Adjusting the sub-plots
        plt.subplots_adjust(bottom=0.15)
        plt.show()

        return fig

    def updated_rw_kvar(self, training, real_world, dictionary_model, training_sets, max_y=3.5):

        # instantiate the class
        fair_cv = SpatialFairSplit(training, real_world, dictionary_model)

        fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 6))
        for column in range(2):
            updt_fair_kvar = np.zeros((len(self.rw_kvar[column, 0]), len(self.test_kvar_fair[column])))
            for i in range(len(self.test_kvar_fair[column])):
                dict_model = fair_cv.dictionary_assigner(training_sets[column][i], real_world[column])
                # kriging variance of testing
                updt_fair_kvar[:, i] = kriging_variance_rw(**dict_model)

            tempo = np.sort(updt_fair_kvar, axis=0)
            updated_test_kvar = np.mean(tempo, axis=1)

            sns.kdeplot(
                x=self.rw_kvar[:], label='Planned real-world use of the model', linewidth=3, ax=axs[column],
                color=(.784, .054, 0.172, 1.)
            )

            sns.kdeplot(
                x=updated_test_kvar[:], label='Expected conditional probability', linestyle="--", linewidth=3,
                ax=axs[column],
                color=(0.18, 0.19, 0.75, 1.0)
            )

            for j in range(updt_fair_kvar.shape[1]):
                if j != updt_fair_kvar.shape[1] - 1:
                    sns.kdeplot(
                        x=self.test_kvar_fair[column][:, j],
                        ax=axs[column],
                        color=(.388, .431, .392, 0.1)
                    )

                else:
                    sns.kdeplot(
                        x=self.test_kvar_fair[column][:, j],
                        ax=axs[column],
                        label="Test realizations",
                        color=(.388, .431, .392, 0.1)
                    )

            axs[column].set_title("Demonstration " + str(column + 1))
            axs[column].set_xlim([0, 1])
            axs[column].set_ylim([0, max_y])
            axs[column].xaxis.set_ticks_position('bottom')
            axs[column].yaxis.set_ticks_position('left')
            axs[column].xaxis.set_tick_params(direction='in')
            axs[column].yaxis.set_tick_params(direction='in')
            axs[column].grid(False)

        fig.suptitle(
            'Comparison of kriging variance distributions',
            fontsize=14,
            fontweight='bold',
            # fontname="Times New Roman"
        )

        axs[0].legend(loc='upper center', bbox_to_anchor=(1, -0.05),
                      fancybox=True, shadow=True, ncol=4)
        plt.show()
