import gc
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier, XGBRegressor

import CYP_inhibition_functions


def perform_hyperopt_tuning_rf(X, y, seed, pm, evals):
    # Reduce data for faster tuning

    def hyperopt_train_test(params):
        clf = RandomForestClassifier(**params, criterion='entropy', random_state=seed)
        return cross_val_score(clf, X, y, cv=5, n_jobs=-2).mean()

    space4rf = {'max_depth': hp.choice('max_depth', range(5, 30)),
                'max_features': hp.choice('max_features', range(1, X.shape[1])),
                'n_estimators': hp.choice('n_estimators', range(30, 1000)),
                'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 30))}

    best = 0

    best_score = 0
    best_params = None

    def f(params):
        nonlocal best_score
        nonlocal best_params
        acc = hyperopt_train_test(params)
        if acc > best_score:
            best_score = acc
            best_params = params
            print('new best:', best_score, params)
        return {'loss': -acc, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(f, space4rf, algo=tpe.suggest, max_evals=evals, trials=trials)
    return best_params, trials


def plot_hyperopt_results(best, trials, pm, model_name, name):
    parameters = best.keys()
    num_params = len(parameters)
    f, axes = plt.subplots(ncols=num_params, figsize=(15, 5))
    cmap = plt.cm.jet
    num_trials = len(trials.trials)

    for i, param in enumerate(parameters):
        xs = np.array([t['misc']['vals'][param] for t in trials.trials]).ravel()
        ys = [-t['result']['loss'] for t in trials.trials]

        for j in range(num_trials):
            color = cmap(float(j) / num_trials)
            axes[int(i)].scatter(xs[j], ys[j], s=20, linewidth=0.01, alpha=0.5, color=color)

        axes[int(i)].set_title(param)

    plt.savefig(pm['dir'] + '/' + pm['Project_name'] + pm['fig_dir'] + '/' + model_name + '_%s_hyp_search.png' % name)


def log_results(name, best, cv_score, test_score, accuracy, pm):
    with open(pm['output_filename'], 'a') as fileOutput:
        fileOutput.write(name + ' best params: ' + str(best) + '\n')
        fileOutput.write('5-fold cross validation: ' + str(cv_score) + '\n')
        fileOutput.write('Test Score: ' + str(test_score) + 'Accuracy Score: ' + str(accuracy) + '\n')


def log_best_params(name, best, pm):
    with open(pm['output_filename'], 'a') as fileOutput:
        fileOutput.write(name + ' best params: ' + str(best) + '\n')


def save_model_to_dir(model, pm, model_name, name):
    model_path = pm['dir'] + '/' + pm['Project_name'] + pm['model_dir'] + '/' + model_name + '_' + name + '_model.h5'
    if not os.path.isfile(model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)

def save_model_xgb(model, pm, model_name, name):
    model_path = pm['dir'] + '/' + pm['Project_name'] + pm['model_dir'] + '/' + model_name + '_' + name + '_model.json'
    if not os.path.isfile(model_path):
        model.save_model(model_path)



def plot_permutation_importance(importances_train, importances_test, tree_importance_sorted_idx, name, pm):
    # get size of the importance_train
    height = int(importances_train.shape[1] / 3)
    f, axs = plt.subplots(1, 2, figsize=(20, height))

    importances_test.plot.box(vert=False, whis=10, ax=axs[0])
    axs[0].set_title("Permutation Importances (test set)")
    axs[0].axvline(x=0, color="k", linestyle="--")
    axs[0].set_xlabel("Decrease in accuracy score")
    axs[0].figure.tight_layout()

    importances_train.plot.box(vert=False, whis=10, ax=axs[1])
    axs[1].set_title("Permutation Importances (train set)")
    axs[1].axvline(x=0, color="k", linestyle="--")
    axs[1].set_xlabel("Decrease in accuracy score")
    axs[1].figure.tight_layout()

    plt.savefig(pm['dir'] + '/' + pm['Project_name'] + pm['fig_dir'] + '/RF_%s_importance.png' % name)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def calculate_and_plot_importances(X_train, y_train, name, pm):
    n_runs = 10
    feature_importances_runs = []

    for i in range(n_runs):
        rf = RandomForestClassifier(random_state=i)
        rf.fit(X_train, y_train)
        feature_importances_runs.append(rf.feature_importances_)

    feature_importances_mean = np.mean(feature_importances_runs, axis=0)
    feature_importances_std = np.std(feature_importances_runs, axis=0)

    tree_importance_sorted_idx = np.argsort(feature_importances_mean)[::-1]

    fig, ax = plt.subplots(figsize=(10, 40))
    ax.barh(range(X_train.shape[1]), feature_importances_mean[tree_importance_sorted_idx],
            xerr=feature_importances_std[tree_importance_sorted_idx], capsize=5)
    ax.set_yticklabels(X_train.columns[tree_importance_sorted_idx])
    ax.set_yticks(range(X_train.shape[1]))
    ax.set_title("Random Forest Feature Importances (%s)" % name)
    plt.tight_layout()
    plt.savefig(pm['dir'] + '/' + pm['Project_name'] + pm['fig_dir'] + '/RF_%s_tree_importance.png' % name)

    return tree_importance_sorted_idx


def select_top_features(X, tree_importance_sorted_idx, n_top_features):
    selected_features = X.columns[tree_importance_sorted_idx][:n_top_features]
    X_selected = X[selected_features]
    return X_selected, selected_features


def select_top_features_perm(importances_train, importances_test, n_top_features):
    importances_train = importances_train.reindex(importances_train.mean().sort_values(ascending=False).index, axis=1)
    importances_test = importances_test.reindex(importances_test.mean().sort_values(ascending=False).index, axis=1)
    # print 50 most left columns names of the importance_train and importance_test
    top_importances_train = importances_train.columns[:n_top_features]
    top_importances_test = importances_test.columns[:n_top_features]
    # make top_importances_train dataframes
    importances_train = importances_train[top_importances_train]
    importances_test = importances_test[top_importances_test]

    return importances_train, importances_test


def prepare_and_run_permutation_importance(X_train, y_train, X_test, y_test, model, pm, seed):
    test_size = 1-0.85
    X_tune_train, _, y_tune_train, _ = train_test_split(X_train, y_train, test_size=test_size, random_state=seed, shuffle=False)
    X_tune_test, _, y_tune_test, _ = train_test_split(X_test, y_test, test_size=test_size, random_state=seed, shuffle=False)

    result_train = permutation_importance(model, X_tune_train, y_tune_train, n_repeats=pm['perm_repeats'],
                                          random_state=seed, n_jobs=4)
    result_test = permutation_importance(model, X_tune_test, y_tune_test, n_repeats=pm['perm_repeats'],
                                         random_state=seed, n_jobs=4)

    return result_train, result_test

def Do_RandomForest_FS(X, y, X_test, y_test, name, pm, scored, seed):
    # Perform hyperparameter tuning
    test_size = 1 - 0.8
    if pm['evaluate_features'] == 0 :
        X_tune, _, y_tune, _ = train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=False)
        best_params, trials = perform_hyperopt_tuning_rf(X_tune, y_tune, seed, pm, 7)
        print('Tuning Done, Best params: ', best_params)
        # Train the model
        rf = RandomForestClassifier(**best_params, criterion='entropy', random_state=seed, n_jobs=-1)
        rf.fit(X, y)

        initial_name = name + '_initial'
        # Calculate feature importances
        tree_importance_sorted_idx = calculate_and_plot_importances(X, y, initial_name, pm)
        print('Feature Importance Done')

        # Save and plot the results
        plot_hyperopt_results(best_params, trials, pm, 'RF', initial_name)

        del X_tune, y_tune,
        gc.collect()

        # Select top features
        n_top_features = 100
        X_selected, selected_features = select_top_features(X, tree_importance_sorted_idx, n_top_features)
        X_selected_test, selected_features = select_top_features(X_test, tree_importance_sorted_idx, n_top_features)
        # Save the selected features
        # combine all selected features
        selected_features_all = pd.DataFrame(
            {'selected_features': selected_features})
        selected_features_all.to_csv(
            pm['dir'] + '/' + pm['Project_name'] + pm['fig_dir'] + '/RF_%s_selected_features.csv' % name)
    else:
        X_selected = X
        X_selected_test = X_test


    X_tune_s, _, y_tune_s, _ = train_test_split(X_selected, y, test_size=test_size+0.1, shuffle = True, random_state=seed)
    # Perform hyperparameter tuning again with selected features
    best_params_selected, trials_selected = perform_hyperopt_tuning_rf(X_tune_s, y_tune_s, seed, pm, pm['maxevals'])

    # Train the model with selected features
    rf_selected = RandomForestClassifier(**best_params_selected, criterion='entropy', random_state=seed, n_jobs=-1)
    rf_selected.fit(X_selected, y)
    scored['cv'] = str(cross_val_score(rf_selected, X_selected, y, scoring='accuracy', cv=5, n_jobs=-2))

    final_name = name + '_final'
    # Calculate feature importances
    calculate_and_plot_importances(X_selected, y, final_name, pm)
    print('2nd Feature Importance Done')

    # Save and plot the results
    plot_hyperopt_results(best_params_selected, trials_selected, pm, 'RF', final_name)

    del X_tune_s, y_tune_s
    gc.collect()

    # Calculate accuracy metrics
    test_score = rf_selected.score(X=X_selected_test, y=y_test)
    test_score_all_feat = rf.score(X=X_test, y=y_test)
    print('5-fold cross validation: ', scored['cv'])
    scores_pred_test = rf_selected.predict(X_selected_test)
    print('Test Score with all features: ', test_score_all_feat)
    scored['test_score'] = test_score
    print("Test accuracy with top features: ", test_score)

    testresults = CYP_inhibition_functions.testresults(y_test, scores_pred_test)
    score_rf = {**scored, **testresults}
    score_rf = pd.DataFrame.from_dict(score_rf, orient='index', columns=[name])
    return score_rf

def Do_RandomForest_FS_PI(X, y, X_test, y_test, name, pm, scored, seed):
    # Perform hyperparameter tuning
    test_size = 1 - 0.8
    X_tune, _, y_tune, _ = train_test_split(X, y, test_size=test_size, random_state=seed)
    best_params, trials = perform_hyperopt_tuning_rf(X_tune, y_tune, seed, pm, 7)
    print('Tuning Done, Best params: ', best_params)
    # Train the model
    rf = RandomForestClassifier(**best_params, criterion='entropy', random_state=seed, n_jobs=-1)
    rf.fit(X, y)

    # Calculate feature importances
    tree_importance_sorted_idx = calculate_and_plot_importances(X, y, name, pm)
    print('Feature Importance Done')

    result_train, result_test = prepare_and_run_permutation_importance(X, y, X_test, y_test, rf, pm, seed)
    print('Permutation Importance Done')
    importances_train = pd.DataFrame(result_train.importances.T, columns=X.columns)
    importances_test = pd.DataFrame(result_test.importances.T, columns=X_test.columns)

    # sort the importance_train from high to low
    top_importances_train, top_importances_test = select_top_features_perm(importances_train, importances_test, 200)
    # combine importance_train and importance_test
    importances_all = pd.concat([top_importances_train, top_importances_test], axis=1)
    # save the importance_all
    importances_all.to_csv(pm['dir'] + '/' + pm['Project_name'] + pm['fig_dir'] + '/RF_%s_importance_all.csv' % name)

    # Save and plot the results
    initial_name = name + '_initial'
    plot_hyperopt_results(best_params, trials, pm, 'RF', initial_name)
    #lot_permutation_importance(importances_train, importances_test, tree_importance_sorted_idx, initial_name, pm)


    del X_tune, y_tune, importances_train, importances_test
    gc.collect()

    # Select top features
    n_top_features = 100
    X_selected, selected_features = select_top_features(X, tree_importance_sorted_idx, n_top_features)
    X_selected_test, selected_features_test = select_top_features(X_test, tree_importance_sorted_idx, n_top_features)
    # Save the selected features
    # combine all selected features
    selected_features_all = pd.DataFrame(
        {'selected_features': selected_features, 'selected_features_test': selected_features_test})
    selected_features_all.to_csv(
        pm['dir'] + '/' + pm['Project_name'] + pm['fig_dir'] + '/RF_%s_selected_features.csv' % name)



    X_tune_s, _, y_tune_s, _ = train_test_split(X_selected, y, test_size=test_size+0.1, random_state=seed)
    #Perform hyperparameter tuning again with selected features
    best_params_selected, trials_selected = perform_hyperopt_tuning_rf(X_tune_s, y_tune_s, seed, pm, pm['maxevals'])

    # Train the model with selected features
    rf_selected = RandomForestClassifier(**best_params_selected, criterion='entropy', random_state=seed, n_jobs=-1)
    rf_selected.fit(X_selected, y)
    scored['cv'] = str(cross_val_score(rf_selected, X_selected, y, scoring='accuracy', cv=5, n_jobs=-2))

    # Calculate feature importances
    tree_importance_sorted_idx = calculate_and_plot_importances(X, y, name, pm)

    print('2nd Feature Importance Done')
    result_train, result_test = prepare_and_run_permutation_importance(X_selected, y, X_selected_test, y_test, rf_selected, pm, seed)
    print('2nd Permutation Importance Done')
    importances_train = pd.DataFrame(result_train.importances.T, columns=X_selected.columns)
    importances_test = pd.DataFrame(result_test.importances.T, columns=X_selected_test.columns)

    # Save and plot the results
    final_name = name + '_final'
    plot_hyperopt_results(best_params, trials, pm, 'RF', final_name)
    plot_permutation_importance(importances_train, importances_test, tree_importance_sorted_idx, final_name, pm)

    del X_tune_s, y_tune_s,# importances_train, importances_test
    gc.collect()

    # Calculate accuracy metrics
    test_score = rf_selected.score(X=X_selected_test, y=y_test)
    test_score_all_feat = rf.score(X=X_test, y=y_test)
    print('5-fold cross validation: ', scored['cv'])
    scores_pred_test = rf_selected.predict(X_selected_test)
    print('Test Score with all features: ', test_score_all_feat)
    scored['test_score'] = test_score
    print("Test accuracy with top features: ", test_score)

    testresults = CYP_inhibition_functions.testresults(y_test, scores_pred_test)
    score_rf = {**scored, **testresults}
    score_rf = pd.DataFrame.from_dict(score_rf, orient='index', columns=[name])
    return score_rf


def Do_XGradientBoost(X, y, testSet, scores_test, name, pm, scored, seed):
    X = sklearn.preprocessing.normalize(X)
    testSet_norm = sklearn.preprocessing.normalize(testSet)

    print(
        "|---------------------------------------------------\nStarting Hyp Search for XGBoost Classifier\n|---------------------------------------------------")

    if pm['hyp_tune'] == 1:
        X_tune, _, y_tune, _ = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=False)

        def hyperopt_train_test(params):
            model = XGBClassifier(**params, random_state=seed)
            return cross_val_score(model, X_tune, y_tune, cv=5, n_jobs=8).mean()

        space4xgb = {'n_estimators': scope.int(hp.quniform('n_estimators', 5, 300, 1)),
                     'learning_rate': hp.choice('learning_rate', [ 0.03, 0.06, 0.12, 0.25, 0.5]),
                     'max_depth': scope.int(hp.quniform('max_depth', 5, 20, 1)),
                     'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 20, 1)),
                     'gamma': scope.int(hp.quniform('gamma', 0, 9, 1))}

        best_score = 0
        best_params = None

        def f(params):
            nonlocal best_score
            nonlocal best_params
            acc = hyperopt_train_test(params)
            if acc > best_score:
                best_score = acc
                best_params = params
                print('new best:', best_score, params)
            return {'loss': -acc, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(f, space4xgb, algo=tpe.suggest, max_evals=pm['maxevals'], trials=trials)

        # Print the best parameters
        print("Best parameters:", best_params)

        log_best_params(name, best_params, pm)

        plot_hyperopt_results(best_params, trials, pm, 'XGB', name)

    else:
        best_params = {'gamma': 0, 'learning_rate': 0.25, 'max_depth': 20, 'min_child_weight': 4, 'n_estimators': 200}

    model = XGBClassifier(**best_params, random_state=seed)
    scored['cv'] = str(cross_val_score(model, X, y, scoring='accuracy', cv=5, n_jobs=8))
    model.fit(X, y)


    print('5-fold cross validation: ', scored['cv'])
    scores_pred_test = model.predict(testSet_norm)

    scored['score'] = model.score(X=testSet_norm, y=scores_test)
    accuracy = accuracy_score(scores_test, scores_pred_test)
    print('Test Score: ', scored, accuracy)

    save_model_xgb(model, pm, 'XGB', name)
    log_results(name, best_params, scored['cv'], scored['score'], accuracy, pm)

    testresults = CYP_inhibition_functions.testresults(scores_test, scores_pred_test)
    score_xgb = {**scored, **testresults}
    score_xgb = pd.DataFrame.from_dict(score_xgb, orient='index', columns=[name])
    return score_xgb

def Do_XGradientBoost_regression(X, y, testSet, scores_test, name, pm, scored, seed):
    X = sklearn.preprocessing.normalize(X)
    testSet_norm = sklearn.preprocessing.normalize(testSet)

    print(
        "|---------------------------------------------------\nStarting Hyp Search for XGBoost Regressor\n|---------------------------------------------------")

    if pm['hyp_tune'] == 1:
        X_tune, _, y_tune, _ = train_test_split(X, y, test_size=0.2, random_state=seed)

        def hyperopt_train_test(params):
            modelxgb = XGBRegressor(**params, random_state=seed)
            return -cross_val_score(modelxgb, X_tune, y_tune, scoring='neg_mean_absolute_error', cv=5, n_jobs=-2).mean()

        space4xgb = {'n_estimators': scope.int(hp.quniform('n_estimators', 5, 100, 1)),
                     'learning_rate': hp.choice('learning_rate', [0.015, 0.03, 0.06, 0.12, 0.25]),
                     'max_depth': scope.int(hp.quniform('max_depth', 10, 20, 1)),
                     'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 20, 1)),
                     'gamma': scope.int(hp.quniform('gamma', 0, 9, 1)),
                     'reg_alpha': scope.int(hp.quniform('reg_alpha', 20, 180, 1))}

        best_score = 10
        best_params = None

        def f(params):
            nonlocal best_score
            nonlocal best_params
            acc = hyperopt_train_test(params)
            print('acc:', acc)
            if acc < best_score:
                best_score = acc
                best_params = params
                print('new best:', best_score, params)
            return {'loss': acc, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(f, space4xgb, algo=tpe.suggest, max_evals=pm['maxevals'], trials=trials)

        # Print the best parameters
        print("Best parameters:", best_params)

        #log_best_params(name, best_params, pm)

        plot_hyperopt_results(best_params, trials, pm, 'XGB', name)

    model = XGBRegressor(**best_params, random_state=seed)
    scored['cv'] = str(cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-2))
    model.fit(X, y)


    print('5-fold cross validation: ', scored['cv'])
    scores_pred_test = model.predict(testSet_norm)

    from sklearn.metrics import mean_squared_error
    accuracy = (mean_squared_error(scores_test, scores_pred_test)) ** 0.5
    scored['score'] = accuracy
    print('Test rmse: ', accuracy)

    save_model_xgb(model, pm, 'XGB', name)
    #log_results(name, best_params, scored['cv'], scored['score'], accuracy, pm)

    score_xgb = {**scored}
    score_xgb = pd.DataFrame.from_dict(score_xgb, orient='index', columns=[name])
    return score_xgb


def Do_GradientBoost(X, y, testSet, scores_test, name, pm, scored, seed):
    X = sklearn.preprocessing.normalize(X)
    testSet_norm = sklearn.preprocessing.normalize(testSet)

    print(
        "|---------------------------------------------------\nStarting Hyp Search for Gradient Boosting Classifier\n|---------------------------------------------------")

    if pm['hyp_tune'] == 1:
        X_tune, _, y_tune, _ = train_test_split(X, y, test_size=0.2, random_state=seed)

        def hyperopt_train_test(params):
            model_gb = GradientBoostingClassifier(**params, random_state=seed)
            return cross_val_score(model_gb, X_tune, y_tune, cv=5, n_jobs=-2).mean()

        space4gb = {'n_estimators': scope.int(hp.quniform('n_estimators', 5, 300, 1)),
                    'learning_rate': hp.choice('learning_rate', [0.015, 0.03, 0.06, 0.12, 0.25]),
                    'max_features': scope.int(hp.quniform('max_features', 10, 19, 1)),
                    'max_depth': scope.int(hp.quniform('max_depth', 1, 40, 1)),
                    'min_samples_split': scope.int(hp.quniform('min_samples_split', 3, 300, 1))}

        best_score = 0
        best_params = None

        def f(params):
            nonlocal best_score
            nonlocal best_params
            acc = hyperopt_train_test(params)
            if acc > best_score:
                best_score = acc
                best_params = params
                print('new best:', best_score, params)
            return {'loss': -acc, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(f, space4gb, algo=tpe.suggest, max_evals=pm['maxevals'], trials=trials)

        # Print the best parameters
        print("Best parameters:", best_params)

        log_best_params(name, best_params, pm)
        plot_hyperopt_results(best_params, trials, pm, 'GB', name)

    model = GradientBoostingClassifier(**best_params, random_state=seed)
    model.fit(X, y)

    scored['cv'] = str(cross_val_score(model, X, y, scoring='accuracy', cv=5, n_jobs=-2))
    print('5-fold cross validation: ', scored['cv'])
    scores_pred_test = model.predict(testSet_norm)

    scored['score'] = model.score(X=testSet_norm, y=scores_test)
    accuracy = accuracy_score(scores_test, scores_pred_test)
    print('Test Score: ', scored, accuracy)

    save_model_to_dir(model, pm, 'GB', name)
    log_results(name, best_params, scored['cv'], scored['score'], accuracy, pm)

    testresults = CYP_inhibition_functions.testresults(scores_test, scores_pred_test)
    score = {**scored, **testresults}
    score = pd.DataFrame.from_dict(score, orient='index', columns=[name])
    return score


def Do_KNN(X, y, testSet, scores_test, name, pm, scored, seed):
    from sklearn.neighbors import KNeighborsClassifier
    X = sklearn.preprocessing.normalize(X)
    testSet_norm = sklearn.preprocessing.normalize(testSet)

    print(
        "|---------------------------------------------------\nStarting Hyp Search for KNN\n|---------------------------------------------------")

    if pm['hyp_tune'] == 1:
        X_tune, _, y_tune, _ = train_test_split(X, y, test_size=0.2, random_state=seed)

        def hyperopt_train_test(params):
            model_knn = KNeighborsClassifier(**params)
            model_knn.random_state = seed
            return cross_val_score(model_knn, X_tune, y_tune, cv=5, n_jobs=-2).mean()

        space4knn = {'weights': hp.choice('weights', ['uniform', 'distance']),
                     'p': hp.choice('p', [1, 2]),
            'n_neighbors': hp.choice('n_neighbors', list(range(5, 301))),
            'leaf_size': hp.choice('leaf_size', list(range(1, 51))), }

        best = 0

        def f(params):
            nonlocal best
            acc = hyperopt_train_test(params)
            if acc > best:
                best = acc
                print('new best:', best, params)
            return {'loss': -acc, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(f, space4knn, algo=tpe.suggest, max_evals=pm['maxevals'], trials=trials)

        if best['weights'] == 0:
            best['weights'] = 'uniform'
        if best['weights'] == 1:
            best['weights'] = 'distance'

        log_best_params(name, best, pm)

        plot_hyperopt_results(best, trials, pm, 'KNN', name)

    model = KNeighborsClassifier(**best)
    model.fit(X, y)

    scored['cv'] = str(cross_val_score(model, X, y, scoring='accuracy', cv=5, n_jobs=-2))
    print('5-fold cross validation: ', scored['cv'])
    scores_pred_test = model.predict(testSet_norm)

    scored['score'] = model.score(X=testSet_norm, y=scores_test)
    accuracy = accuracy_score(scores_test, scores_pred_test)
    print('Test Score: ', scored, accuracy)

    save_model_to_dir(model, pm, 'KNN', name)

    testresults = CYP_inhibition_functions.testresults(scores_test, scores_pred_test)
    score = {**scored, **testresults}
    score = pd.DataFrame.from_dict(score, orient='index', columns=[name])
    return score





def Do_LR(X, y, testSet, scores_test, name, pm, scored, seed):
    X = sklearn.preprocessing.normalize(X)
    testSet_norm = sklearn.preprocessing.normalize(testSet)

    print(
        "|---------------------------------------------------\nStarting Hyp Search for LR\n|---------------------------------------------------")

    if pm['hyp_tune'] == 1:
        X_tune, _, y_tune, _ = train_test_split(X, y, test_size=0.2, random_state=seed)

        def hyperopt_train_test(params):
            model_lr = LogisticRegression(**params, max_iter=400, verbose=False)
            model_lr.random_state = seed
            return cross_val_score(model_lr, X_tune, y_tune, cv=5, n_jobs=-2).mean()

        space4lr = {'penalty': hp.choice('penalty', ['l2', 'none']), 'C': hp.choice('C', range(1, 20)), }

        best = 0

        def f(params):
            nonlocal best
            acc = hyperopt_train_test(params)
            if acc > best:
                best = acc
                print('new best:', best, params)
            return {'loss': -acc, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(f, space4lr, algo=tpe.suggest, max_evals=pm['maxevals'], trials=trials)

        if best['penalty'] == 0:
            best['penalty'] = 'l2'
        if best['penalty'] == 1:
            best['penalty'] = 'none'

        log_best_params(name, best, pm)

        plot_hyperopt_results(best, trials, pm, 'LR', name)

    model = LogisticRegression(**best, max_iter=400)
    model.fit(X, y)

    scored['cv'] = str(cross_val_score(model, X, y, scoring='accuracy', cv=5, n_jobs=-2))
    print('5-fold cross validation: ', scored['cv'])
    scores_pred_test = model.predict(testSet_norm)

    scored['score'] = model.score(X=testSet_norm, y=scores_test)
    accuracy = accuracy_score(scores_test, scores_pred_test)
    print('Test Score: ', scored, accuracy)

    save_model_to_dir(model, pm, 'LR', name)
    log_results(name, best, scored['cv'], scored['score'], accuracy, pm)

    testresults = CYP_inhibition_functions.testresults(scores_test, scores_pred_test)
    score = {**scored, **testresults}
    score = pd.DataFrame.from_dict(score, orient='index', columns=[name])
    return score


def Do_DNN(X, y, testSet, scores_test, name, pm, scored, seed):
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, BatchNormalization, Activation
    from tensorflow.keras.optimizers import Adam
    from keras_tuner.tuners import Hyperband

    from tensorflow import keras
    X = sklearn.preprocessing.normalize(X)
    testSet_norm = sklearn.preprocessing.normalize(testSet)

    print(
        "|---------------------------------------------------\nStarting Hyp Search for DNN\n|---------------------------------------------------")

    number_of_variables = X.shape[1]

    def model_builder(hp):
        hp_units1 = hp.Int('units1', min_value=32, max_value=1024, step=32)
        hp_units2 = hp.Int('units2', min_value=32, max_value=832, step=32)
        hp_units3 = hp.Int('units3', min_value=32, max_value=512, step=32)

        model = keras.Sequential()
        model.add(Dense(units=hp_units1, input_shape=(number_of_variables,),
                        activity_regularizer=tf.keras.regularizers.l2(
                            hp.Choice('Regularization', values=[0.1, 1e-2, 1e-3]))))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(keras.layers.Dropout(hp.Choice('Dropout', values=[0.1, 1e-2, 1e-3])))
        model.add(Dense(units=hp_units2, activity_regularizer=tf.keras.regularizers.l2(
            hp.Choice('Regularization', values=[0.1, 1e-2, 1e-3]))))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(keras.layers.Dropout(hp.Choice('Dropout', values=[0.1, 1e-2, 1e-3])))
        model.add(Dense(units=hp_units3, activity_regularizer=tf.keras.regularizers.l2(
            hp.Choice('Regularization', values=[0.1, 1e-2, 1e-3]))))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(units=2, activation='softmax'))
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    tuner = Hyperband(model_builder, objective='val_accuracy', max_epochs=15, factor=2, directory='my_dir',
                      project_name='intro_to_kt', seed=seed, hyperband_iterations=2, overwrite=True)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    X_tune, _, y_tune, _ = train_test_split(X, y, test_size=0.2, random_state=seed)
    tuner.search(X_tune, y_tune, epochs=(pm['maxevals'] + 2), validation_split=0.2, callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model_dnn = tuner.hypermodel.build(best_hps)
    history = model_dnn.fit(X, y, epochs=50, validation_split=0.2)

    print('Best hyper-parameters :')
    tuner.results_summary(num_trials=1)
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(X, y, epochs=best_epoch, validation_split=0.2)

    scores_pred_test = hypermodel.predict(testSet_norm)
    bi = np.argmax(scores_pred_test, axis=1)
    scores_pred_test = pd.DataFrame(bi)

    scored['score'] = hypermodel.evaluate(testSet_norm, scores_test)[1]
    print("[test loss, test accuracy]:", scored)

    save_model_to_dir(hypermodel, pm, 'DNN', name)
    log_results(name, best_hps, None, scored['score'], None, pm)

    testresults = CYP_inhibition_functions.testresults(scores_test, scores_pred_test)
    scored = {**scored, **testresults}
    scored = pd.DataFrame.from_dict(scored, orient='index', columns=[name])

    return scored
