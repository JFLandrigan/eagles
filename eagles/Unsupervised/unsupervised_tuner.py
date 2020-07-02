from eagles.Unsupervised.utils import plot_utils as pu
from eagles.Unsupervised.utils import cluster_eval_utils as ceu
from eagles.Unsupervised.utils import logger_utils as lu
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import logging

logger = logging.getLogger(__name__)

def find_max_sil(res_dict):
    max_ind = res_dict["scores"].argmax()
    num_clusters = res_dict["n_clusters"][max_ind]
    max_sil_score = res_dict["scores"][max_ind]

    return num_clusters, max_sil_score


def _init_method(model=None, params={}):

    if model is None:
        logger.warning("No model passed in")
        return

    if model == "kmeans":
        mod = KMeans(**params)
    elif model == "agglomerativeclustering":
        mod = AgglomerativeClustering(**params)
    elif model == "dbscan":
        mod = DBSCAN(**params)
    else:
        mod = model

    return mod


def find_optimal_clusters(
    data=None,
    ft_cols=[],
    cluster_method="kmeans",
    metric="min_sil",
    min_num_clusters=2,
    max_num_clusters=10,
    params={},
    scale=None,
    plot_dims=[],
    summary_stats=[],
    run_stat_comps=True,
    plot_scale=None,
    random_seed=None,
    log="log",
    log_name=None,
    log_path=None,
    log_note=None,
):

    if min_num_clusters == max_num_clusters:
        logger.warning("WARNING MIN AND MAX NUM CLUSTERS SHOULD NOT BE EQUAL")
        return

    if random_seed is None:
        random_seed = np.random.randint(1000, size=1)[0]
    print("Random Seed Value: " + str(random_seed))

    if len(ft_cols) == 0:
        ft_cols = [col for col in data.columns]

    data = data[ft_cols].copy(deep=True)

    if scale:
        if scale == "standard":
            scaler = StandardScaler()
            data = scaler.fit_transform(data[ft_cols])
        elif scale == "minmax":
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data[ft_cols])
        else:
            data = scale.fit_transfrom(data)

        data = pd.DataFrame(data)
        data.columns = ft_cols

    # if kmeans of agglom loop through to find the optimal clusters
    if cluster_method in ["kmeans", "agglomerativeclustering"]:

        res_dict = {"n_clusters": np.array([]), "scores": np.array([])}

        # loop through the number of clusters and create dictionary of num clusters with metrics
        for i in range(min_num_clusters, max_num_clusters, 1):

            params["n_clusters"] = i
            res_dict["n_clusters"] = np.append(res_dict["n_clusters"], i)
            model = _init_method(model=cluster_method, params=params)

            pred_labels = model.fit_predict(data[ft_cols])

            if metric in ["max_sil", "knee_sil"]:
                res_dict["scores"] = np.append(
                    res_dict["scores"], silhouette_score(data, pred_labels)
                )
            elif metric == "knee_wss":
                res_dict["scores"] = np.append(res_dict["scores"], model.inertia_)
            else:
                logger.warning("WARNING METRIC NOT SUPPORTED")
                return

    elif cluster_method in ["dbscan"]:
        model = _init_method(model=cluster_method, params=params)
        model.fit_predict(data[ft_cols])
    else:
        logger.warning("WARNING the clustering method is not supported")
        return

    # Once looped through and found the scores across the range of clusters then get final set based on the best score
    if cluster_method in ["kmeans", "agglomerativeclustering"]:

        if metric == "max_sil":
            opt_n_clusters, max_sil_score = find_max_sil(res_dict=res_dict)
            opt_n_clusters = int(opt_n_clusters)
            print("Best silhoutte score: " + str(max_sil_score))
        elif metric == "knee_wss":
            kn = KneeLocator(
                x=res_dict["n_clusters"],
                y=res_dict["scores"],
                curve="convex",
                direction="decreasing",
            )
            opt_n_clusters = int(kn.knee)
        elif metric == "knee_sil":
            kn = KneeLocator(
                x=res_dict["n_clusters"],
                y=res_dict["scores"],
                curve="concave",
                direction="increasing",
            )
            opt_n_clusters = int(kn.knee)

        pu.plot_score_curve(data=res_dict, metric=metric, opt_n_clusters=opt_n_clusters)

    elif cluster_method in ["dbscan"]:
        opt_n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)

    print("Optimal number of clusters: " + str(opt_n_clusters) + "\n")

    eval_clusters(
        data=data,
        n_clusters=opt_n_clusters,
        method=cluster_method,
        params=params,
        ft_cols=ft_cols,
        plot_dims=plot_dims,
        summary_stats=summary_stats,
        run_stat_comps=run_stat_comps,
        plot_scale=plot_scale,
        log=log,
        log_name=log_name,
        log_path=log_path,
        log_note=log_note,
    )

    return data


def eval_clusters(
    data=None,
    n_clusters=2,
    method=None,
    params={},
    scale=None,
    ft_cols=[],
    plot_dims=[],
    summary_stats=[],
    run_stat_comps=True,
    plot_scale=None,
    log="log",
    log_name=None,
    log_path=None,
    log_note=None,
):

    if len(ft_cols) == 0:
        ft_cols = [col for col in data.columns]

    data = data[ft_cols].copy(deep=True)

    if scale:
        if scale == "standard":
            scaler = StandardScaler()
            data = scaler.fit_transform(data[ft_cols])
        elif scale == "minmax":
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data[ft_cols])
        else:
            data = scale.fit_transfrom(data)

        data = pd.DataFrame(data)
        data.columns = ft_cols

    params["n_clusters"] = n_clusters
    model = _init_method(model=method, params=params)
    pred_labels = model.fit_predict(data[ft_cols])
    data["Cluster"] = model.labels_

    sil_score = silhouette_score(data, pred_labels)

    print("Silhouette Score: " + str(round(sil_score, 2)))
    if type(model).__name__ == "Pipeline":
        if type(model.named_steps["model"]).__name__ == "KMeans":
            print("WSS Total: " + str(round(model.named_steps["model"].inertia_, 2)) + "\n")
    elif method == "kmeans":
        print("WSS Total: " + str(round(model.inertia_, 2)) + "\n")

    if len(plot_dims) == 0:
        plot_dims = ft_cols + ["Cluster"]

    print("Number of Observations per Cluster")
    print(str(data["Cluster"].value_counts()) + "\n\n")

    base_cluster_stats = ceu.create_summary_table(
        data=data, plot_dims=plot_dims, summary_stats=summary_stats
    )
    base_cluster_stats = round(base_cluster_stats,2)
    print("Base Cluster Stats \n")
    print(base_cluster_stats.T)
    print("\n\n")

    if run_stat_comps:
        sig_test_results, post_hoc_comps = ceu.run_cluster_comps(
            data=data, ft_cols=ft_cols
        )
        if sig_test_results.shape[0] == 0:
            print("No significant differences found between clusters")
        else:
            print("Significance Testing Results \n")
            print(str(round(sig_test_results, 2)) + "\n\n")
            if post_hoc_comps.shape[0] == 0:
                print("No pairwise significant difference")
            else:
                print("Pairwise Differences \n")
                print(str(round(post_hoc_comps, 2)) + "\n\n")

    pu.plot_mean_cluster_scores(data=data, plot_scale=plot_scale)
    pu.plot_ft_relationships(data=data, plot_dims=plot_dims)

    if log:
        log_data = {
            "n_clusters":n_clusters
            ,"features":ft_cols
            ,"Silhouette Score":round(sil_score, 2)
            ,"data":data
            ,"params": model.get_params()
            ,"base_cluster_stats":round(base_cluster_stats,2)
        }

        if type(model).__name__ == "Pipeline":
            log_data["method"] = type(model).__name__
            pipe_steps = "Pipe steps: "
            for k in model.named_steps.keys():
                pipe_steps = pipe_steps + type(model.named_steps[k]).__name__ + " "
            log_data["pipe_steps"] = pipe_steps
        else:
            log_data["method"] = type(model).__name__

        if type(model).__name__ == "Pipeline":
            if type(model.named_steps["model"]).__name__ == "KMeans":
                log_data["WSS"] = round(model.named_steps["model"].inertia_,2)
        elif method == "kmeans":
            log_data["WSS"] = round(model.inertia_, 2)

        if log_note:
            log_data['note'] = log_note

        lu.log_results(fl_name=log_name, fl_path=log_path, log_data=log_data)


    return data
