"""Create a plain-language educational report from model output and input values."""


def generate_health_report(inputs, prediction, probability=None):
    """Return report sections without diagnosing or recommending treatment."""
    if prediction == 1:
        risk_summary = (
            "The model identified a higher-risk indicator based on the entered "
            "values."
        )
        professional_recommendation = (
            "Because the model returned a higher-risk indicator, consider "
            "discussing these results with a qualified healthcare professional. "
            "They can interpret them in the context of your overall health."
        )
    else:
        risk_summary = (
            "The model identified a lower-risk indicator based on the entered "
            "values. This result does not rule out health concerns."
        )
        professional_recommendation = (
            "If you have health concerns or questions about screening, consult "
            "a qualified healthcare professional, regardless of this result."
        )

    if probability is not None:
        risk_summary += (
            f" The model's estimated probability for the higher-risk class is "
            f"{probability * 100:.1f}%."
        )

    # Present the submitted measurements transparently; they are not diagnoses
    # or a claim about which value caused the model's prediction.
    important_factors = [
        f"{name}: {value:g}" for name, value in inputs.items()
    ]

    general_guidance = (
        "Use this result as general educational information only. Everyday "
        "habits such as balanced meals, regular physical activity, and routine "
        "health checkups can support overall wellbeing. A healthcare "
        "professional can offer guidance suited to your circumstances."
    )

    return {
        "risk_summary": risk_summary,
        "important_factors": important_factors,
        "general_guidance": general_guidance,
        "professional_recommendation": professional_recommendation,
        "disclaimer": (
            "This model is an educational risk-indicator system and is not a "
            "medical diagnosis. It cannot confirm or rule out a disease."
        ),
    }
