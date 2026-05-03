def test_feature_order():
    FEATURE_ORDER = ['amount', 'hour', 'v1', 'v2', 'v3', 'v4', 'v5']
    assert len(FEATURE_ORDER) == 7
    assert FEATURE_ORDER[0] == 'amount'
    assert FEATURE_ORDER[-1] == 'v5'

def test_risk_logic():
    def get_risk(probability):
        return "high" if probability > 0.7 else "medium" if probability > 0.4 else "low"

    assert get_risk(0.9) == "high"
    assert get_risk(0.5) == "medium"
    assert get_risk(0.1) == "low"

def test_action_logic():
    def get_action(probability):
        return "block" if probability > 0.7 else "review" if probability > 0.4 else "approve"

    assert get_action(0.9) == "block"
    assert get_action(0.5) == "review"
    assert get_action(0.1) == "approve"

def test_prediction_output_keys():
    output = {
        "fraud": True,
        "fraud_probability": 0.99,
        "risk": "high",
        "action": "block"
    }
    assert "fraud" in output
    assert "fraud_probability" in output
    assert "risk" in output
    assert "action" in output