"""Test the actual Task 3 notebook functions and preserve the existing Task 1/2 code."""
import ast
import json
import os
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

ROOT = Path(os.environ.get('TASK3_REPO', Path(__file__).resolve().parents[1]))
NAME = 'Poon_Robatche-Claive_Salanon_Xu_Assignment1.ipynb'
NB_PATH = Path(os.environ.get('TASK3_NOTEBOOK', ROOT / 'notebooks' / NAME))
NB = json.loads(NB_PATH.read_text())
CODE = lambda c: ''.join(c['source'])

@pytest.fixture(scope='module')
def functions():
    import matplotlib
    matplotlib.use('Agg')
    from sklearn.linear_model import Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.exceptions import ConvergenceWarning
    namespace = dict(np=np, pd=pd, warnings=warnings, Lasso=Lasso,
                     RandomForestRegressor=RandomForestRegressor, MLPRegressor=MLPRegressor,
                     ConvergenceWarning=ConvergenceWarning)
    for cell in NB['cells']:
        if 'task3-definitions' in cell['metadata'].get('tags', []):
            exec(compile(CODE(cell), str(NB_PATH), 'exec'), namespace)
    return namespace

@pytest.fixture(scope='module')
def world(functions):
    days = pd.bdate_range('2000-01-03', periods=310)
    rng = np.random.default_rng(123)
    raw = pd.DataFrame(rng.normal(0, .02, (310, 110)), index=days)
    members = pd.DataFrame(True, index=days, columns=raw.columns)
    members.iloc[265:268, 0] = False
    sigma = functions['volatility'](raw)
    adjusted = raw / sigma.shift()
    target = adjusted.clip(-5, 5).shift(-1).where(members)
    target = target.sub(target.mean(axis=1), axis=0)
    dates = pd.Series(days, index=days).shift(-1)
    features, _ = functions['build_features'](adjusted.clip(-5, 5), sigma, members)
    return dict(days=days, raw=raw, members=members, sigma=sigma, adjusted=adjusted,
                target=target, dates=dates, features=features)

def test_ewma_equations_burn_in_and_lag(functions):
    raw = pd.DataFrame({'a': [.1, -.2, np.nan, .05]})
    sigma = functions['volatility'](raw, beta1=.5, beta2=.75, initial_variance=.04, burn_in=2)
    mean1 = .25*.1
    variance1 = .5*(.1-mean1)**2 + .5*.04
    mean2 = .25*(-.2) + .75*mean1
    variance2 = .5*(-.2-mean2)**2 + .5*variance1
    assert np.isnan(sigma.iloc[0, 0])
    assert sigma.iloc[1, 0] == pytest.approx(np.sqrt(variance2))
    assert sigma.iloc[2, 0] == sigma.iloc[1, 0]
    adjusted = raw / sigma.shift()
    assert adjusted.iloc[3, 0] == pytest.approx(.05/np.sqrt(variance2))

@pytest.mark.parametrize('position', [130, 260, 289])
def test_features_do_not_change_when_future_is_removed(functions, world, position):
    w = world
    cut = w['days'][position]
    raw, members = w['raw'].loc[:cut], w['members'].loc[:cut]
    sigma = functions['volatility'](raw)
    features, _ = functions['build_features']((raw/sigma.shift()).clip(-5, 5), sigma, members)
    for name in features:
        pd.testing.assert_series_equal(features[name].loc[cut], w['features'][name].loc[cut])

def test_feature_ranks_membership_missingness_and_momentum(functions):
    days = pd.bdate_range('2000-01-03', periods=300)
    raw = pd.DataFrame(0., index=days, columns=['a', 'b', 'c', 'outside'])
    raw.iloc[-22, 0] = 5  # t-21 is included in momentum.
    raw.iloc[-21, 1] = 5  # t-20 is excluded.
    members = raw.notna()
    members['outside'] = False
    sigma = raw * 0 + 1
    features, _ = functions['build_features'](raw, sigma, members)
    assert features['momentum'].iloc[-1]['a'] == .5
    assert features['momentum'].iloc[-1]['b'] < .5
    assert features['momentum'].iloc[-1]['outside'] == 0
    ranks = functions['rank_members'](pd.DataFrame([[1., np.nan, 3.]]), pd.DataFrame([[True]*3]))
    assert ranks.iloc[0].tolist() == [-.5, 0., .5]

def test_refit_uses_only_strictly_earlier_outcomes(functions, world):
    w = world
    first = w['days'][270]
    _, records = functions['walk_forward'](
        w['features'], w['target'], w['members'], w['dates'], {'model':'lasso','alpha':.0001},
        first, w['days'][290], train_days=60, refit_every=10, max_rows=2000)
    assert (records.last_training_outcome < records.refit).all()
    assert records.iloc[0].last_training_feature == w['days'][268]
    assert records.iloc[0].last_training_outcome == w['days'][269]
    assert records.feature_days.eq(60).all()

def test_current_day_outcome_cannot_enter_fit(functions, world, monkeypatch):
    w = world
    seen = []
    class Probe:
        def predict(self, X): return X[:, 0]
    def capture(spec, X, y):
        seen.append((X.copy(), y.copy()))
        return Probe()
    monkeypatch.setitem(functions, 'fit_model', capture)
    t = w['days'][270]
    fn = functions['walk_forward']
    args = (w['features'], w['target'], w['members'], w['dates'], {'model':'lasso','alpha':.0001}, t, t)
    fn(*args, train_days=60)
    changed = w['target'].copy()
    changed.loc[w['days'][269]] += 1e6  # This row's outcome is realized on prediction day t.
    fn(w['features'], changed, *args[2:], train_days=60)
    np.testing.assert_array_equal(seen[0][0], seen[1][0])
    np.testing.assert_array_equal(seen[0][1], seen[1][1])

def test_walk_forward_predictions_survive_raw_data_truncation(functions, world):
    w = world
    spec = {'model':'lasso','alpha':.0001}
    full, _ = functions['walk_forward'](w['features'], w['target'], w['members'], w['dates'],
                                               spec, w['days'][270], w['days'][300], train_days=60, refit_every=10)
    cut = w['days'][285]
    raw, mem = w['raw'].loc[:cut], w['members'].loc[:cut]
    sigma = functions['volatility'](raw)
    adjusted = raw/sigma.shift()
    features, _ = functions['build_features'](adjusted.clip(-5, 5), sigma, mem)
    target = adjusted.clip(-5, 5).shift(-1).where(mem)
    target = target.sub(target.mean(axis=1), axis=0)
    dates = pd.Series(raw.index, index=raw.index).shift(-1)
    short, _ = functions['walk_forward'](features, target, mem, dates, spec, w['days'][270], cut,
                                               train_days=60, refit_every=10)
    pd.testing.assert_frame_equal(full.loc[:cut], short)

def boundary_world():
    days = pd.to_datetime(['2010-12-29','2010-12-30','2010-12-31','2011-01-03','2011-01-04','2011-01-05'])
    raw = pd.DataFrame(np.arange(606).reshape(6,101)/10000, index=days)
    members = raw.notna()
    dates = pd.Series(days,index=days).shift(-1)
    actual = raw.shift(-1) / np.linspace(.5, 3, 101)
    predictions = raw * 10
    return days, raw, members, dates, actual, predictions

def test_outcome_split_has_no_first_test_return_in_development(functions):
    days, raw, mem, dates, actual, pred = boundary_world()
    fn = functions['score_predictions']
    dev, _, _ = fn(pred, actual, mem, raw, dates, '2010-12-30','2010-12-31')
    test, _, _ = fn(pred, actual, mem, raw, dates, '2011-01-01','2013-12-31')
    assert dev.index.tolist() == list(days[1:3])
    assert test.index.tolist() == list(days[3:])
    assert not len(dev.index.intersection(test.index))
    changed = actual.copy()
    changed.loc['2010-12-31'] += 1e6
    again, _, _ = fn(pred, changed, mem, raw, dates, '2010-12-30','2010-12-31')
    pd.testing.assert_series_equal(dev, again)

def test_missing_day_and_fewer_than_100_scored_fail(functions):
    days, raw, mem, dates, actual, pred = boundary_world()
    fn = functions['score_predictions']
    with pytest.raises(ValueError,match='Fewer than 100'):
        fn(pred.drop(index=days[2]), actual, mem, raw, dates,'2011-01-01','2013-12-31')
    pred.loc[days[2],:1] = np.nan
    with pytest.raises(ValueError,match='Fewer than 100'):
        fn(pred, actual, mem, raw, dates,'2011-01-01','2013-12-31')
    with pytest.raises(ValueError,match='next trading day'):
        fn(pred, actual, mem, raw, dates + pd.Timedelta(days=1),'2011-01-01','2013-12-31')

def test_hand_calculated_pnl_ic_and_raw_market(functions):
    days = pd.bdate_range('2012-01-02',periods=5)
    pred = pd.DataFrame([[1,2,3],[2,0,-2],[3,1,2],[1,3,2],[0,1,2]],index=days,dtype=float)
    actual = pd.DataFrame([[.1,.2,.3],[-.1,0,.1],[.2,-.1,.05],[.3,.2,.1],[np.nan]*3],index=days)
    raw = pd.DataFrame([[.01,.02,.03],[.02,.03,.06],[-.04,.02,.01],[.01,.02,.06],[.02,.03,.01]],index=days)
    members = pred.notna()
    members.iloc[1,2] = False
    dates = pd.Series(days,index=days).shift(-1)
    pnl, ic, metrics = functions['score_predictions'](pred,actual,members,raw,dates,str(days[1].date()),str(days[-1].date()),min_predictions=2)
    positions = pred.where(members).sub(pred.where(members).mean(axis=1),axis=0)
    expected = np.array([.2,-.1,.3,-.1])
    np.testing.assert_allclose(pnl, expected, atol=1e-14)
    for i in range(4):
        ok = members.iloc[i]
        assert ic.iloc[i] == pytest.approx(spearmanr(positions.iloc[i][ok],actual.iloc[i][ok]).statistic)
    benchmark = raw.shift(-1).where(members).mean(axis=1).iloc[:4]
    assert metrics['market correlation'] == pytest.approx(np.corrcoef(expected,benchmark)[0,1])
    assert metrics['Sharpe'] == pytest.approx(expected.mean()/expected.std(ddof=1)*np.sqrt(252))

@pytest.mark.parametrize('values,expected,infinite', [
    ([2,2,2],np.inf,1),([0,0,0],np.nan,0),([1,-1,1],.5,0),
    ([1,2,np.nan,8,9],5.5,0),([1,np.nan,9],np.nan,0)])
def test_holding_gap_zero_and_infinite_cases(functions,values,expected,infinite):
    holding, count = functions['holding_period'](pd.DataFrame({'a':values}))
    assert np.isnan(holding) if np.isnan(expected) else holding == pytest.approx(expected)
    assert count == infinite

def test_constant_predictions_have_undefined_sharpe_and_ic(functions):
    days, raw, mem, dates, actual, pred = boundary_world()
    _, ic, metrics = functions['score_predictions'](pred*0+1,actual,mem,raw,dates,'2011-01-01','2013-12-31')
    assert np.isnan(metrics['Sharpe']) and ic.isna().all()

def test_tiny_constant_forecasts_do_not_create_numerical_market_exposure(functions):
    days, raw, mem, dates, actual, pred = boundary_world()
    pred = (pred*0+1e-10).astype('float32')
    mem.iloc[2, 0] = False
    pnl, ic, metrics = functions['score_predictions'](pred,actual,mem,raw,dates,'2011-01-01','2013-12-31')
    assert pnl.eq(0).all() and np.isnan(metrics['Sharpe']) and ic.isna().all()

def test_weekend_sentiment_is_delayed(functions):
    trading = pd.to_datetime(['2010-01-08','2010-01-11','2010-01-12'])
    sentiment = pd.DataFrame({'a':[2.,4.,6.,10.]},index=pd.date_range('2010-01-08',periods=4))
    aligned = functions['align_sentiment'](sentiment,trading)
    assert aligned.loc['2010-01-11','a'] == 4
    assert aligned.loc['2010-01-12','a'] == 10
    assert pd.Timestamp('2010-01-08') not in aligned.index

def test_sentiment_excludes_duplicate_ids(functions,tmp_path):
    calendar = pd.date_range('2010-01-01',periods=5)
    for name in ['bullish','bearish']:
        pd.DataFrame({'date':calendar,'10':1.,'20':2.}).to_csv(tmp_path/f'psych_{name}.csv',index=False)
    trading = pd.bdate_range('2010-01-01',periods=5)
    members = pd.DataFrame(True,index=trading,columns=['10','10_b','20'])
    features, coverage, _ = functions['build_sentiment_features'](tmp_path,members,['10'])
    assert features['sentiment_available']['10'].eq(0).all()
    assert features['sentiment_available']['10_b'].eq(0).all()
    assert features['sentiment_available']['20'].iloc[1] == 1

def test_loader_keeps_duplicate_securities_and_removes_empty_columns(functions,tmp_path):
    days = pd.bdate_range('2010-01-01',periods=4)
    pd.DataFrame([[.1,.2,np.nan]]*4,index=days,columns=['10','10','20']).to_csv(
        tmp_path/'eq_returns.csv',index_label='date')
    pd.DataFrame([[1,1,0]]*4,index=days,columns=['10','10','20']).to_csv(
        tmp_path/'eq_indicator.csv',index_label='date')
    pd.DataFrame({'permno':[10,10,20],'cname':['A','B','C'],'gics':[10,20,30]}).to_csv(
        tmp_path/'eq_stock_metadata.csv',index=False)
    for name in ['bullish','bearish']:
        pd.DataFrame({'date':days,'10':1.}).to_csv(tmp_path/f'psych_{name}.csv',index=False)
    raw, members, meta, cleaning = functions['load_data'](tmp_path)
    assert raw.columns.tolist()==['10','10_b']
    assert members.columns.equals(raw.columns) and meta.index.tolist()==['10','10_b']
    assert cleaning['empty_columns_removed']==1
    assert cleaning['ambiguous_sentiment_ids']==['10']

@pytest.mark.parametrize('model', [{'model':'lasso','alpha':.001}, {'model':'forest','depth':4,'leaf':200},
                                 {'model':'mlp','hidden':16,'epochs':2}])
def test_all_model_families_fit_and_predict(functions,model):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(500,5))
    y = X[:,0]*.01+rng.normal(size=500)
    fitted = functions['fit_model'](model,X,y)
    assert np.isfinite(fitted.predict(X[:10])).all()

def test_regression_sources_included_without_behavior_changes():
    for name in ['Task1_regression_analysis.py','task2_regression_elasticnet.py','task2_regression_lasso.py',
                 'task2_classification_mlp.py','task2_classification_svm.py','task2_classification_helpers.py']:
        source = (ROOT/'src'/name).read_text()
        if name.startswith('task2_classification'):
            source = '\n'.join(line for line in source.splitlines() if not line.startswith('from task2_'))
        included = '\n'.join(CODE(c) for c in NB['cells'] if c['metadata'].get('original_source')==name)
        assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(included)), name

def test_classification_experiments_preserved_except_local_import_bootstrap():
    original = json.loads(subprocess.check_output(['git','show',f'c975847:notebooks/{NAME}'],cwd=ROOT,text=True))
    for cell in NB['cells']:
        if 'original_notebook_cell' not in cell['metadata']:
            continue
        old = CODE(original['cells'][cell['metadata']['original_notebook_cell']])
        tree = ast.parse(old)
        tree.body = [n for n in tree.body if not (
            isinstance(n,ast.ImportFrom) and (n.module or '').startswith('task2_') or
            isinstance(n,ast.If) and 'os.chdir' in ast.get_source_segment(old,n))]
        assert ast.dump(tree) == ast.dump(ast.parse(CODE(cell)))

def test_notebook_is_standalone_and_has_one_task3_section():
    sections = [c for c in NB['cells'] if c['cell_type']=='markdown' and CODE(c).startswith('# Task 3 —')]
    assert len(sections)==1
    definitions=[]
    for cell in NB['cells']:
        if cell['cell_type']!='code': continue
        tree=ast.parse(CODE(cell))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert not (node.module or '').startswith(('task2_','task3_'))
        if 'task3' in cell['metadata'].get('tags',[]):
            assert '/Users/' not in CODE(cell)
            definitions.extend(n.name for n in tree.body if isinstance(n,ast.FunctionDef))
    assert len(definitions)==len(set(definitions))
