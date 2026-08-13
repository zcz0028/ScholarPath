from scholarpath.selector.adaptive_cutoff import AdaptiveCutoffModel, extract_cutoff_features, snap_k


def paper(score, cov=0.0, sources=1):
    return {"title":"x", "raw":{"day8_final_score":score,"day8_canonical_constraint_coverage":cov,"day8_raw_constraint_coverage":cov,"b4_source_count":sources}}


def test_extract_features_is_gold_free_and_stable():
    row={"qid":"q1","question":"graph neural networks for molecules","papers":[paper(.9,.8,2),paper(.7,.4,1),paper(.2,0,1)],"gold_papers":[{"title":"SHOULD NOT MATTER"}]}
    a=extract_cutoff_features(row)
    row["gold_papers"]=[{"title":"DIFFERENT GOLD"}]
    b=extract_cutoff_features(row)
    assert a==b
    assert a["score_top1"]==.9
    assert a["query_token_count"]==5


def test_snap_k_prefers_smaller_on_tie():
    assert snap_k(7,(6,8))==6
    assert snap_k(9,(5,10,20))==10


def test_model_predicts_without_gold_fields():
    feats=[{"x":0.0,"y":0.0},{"x":.1,"y":.1},{"x":5.0,"y":5.0},{"x":5.1,"y":5.1}]
    model=AdaptiveCutoffModel.fit(feats,[2,3,40,50],neighbor_count=2,allowed_ks=(1,2,3,5,10,20,40,50))
    k,reason=model.predict({"x":.05,"y":.05})
    assert k in (2,3)
    assert "neighbor_ks" in reason


def test_model_is_deterministic():
    feats=[{"x":0.0},{"x":1.0},{"x":2.0}]
    model=AdaptiveCutoffModel.fit(feats,[2,5,10],neighbor_count=2,allowed_ks=(2,5,10))
    assert model.predict({"x":.8})==model.predict({"x":.8})
